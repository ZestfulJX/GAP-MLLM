import os
import json
import copy
import torch
import base64
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from io import BytesIO
from typing import List
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.data.utils import load_and_preprocess_images
from qwen_vl.model.modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT
from collections import defaultdict
from pytorch3d.ops import box3d_overlap
from typing import Dict, Optional, Sequence, List, Tuple, Any, Union
from pytorch3d.transforms import euler_angles_to_matrix

# hyperparameters inherited from Qwen2.5-VL
min_pixels: int = 256 * 28 * 28
max_pixels: int = 1605632
max_num_frames: int = 32

# others
device="cuda:0" # npu:0

class EulerDepthInstance3DBoxes:
    """3D boxes of instances in Depth coordinates.

    We keep the "Depth" coordinate system definition in MMDet3D just for
    clarification of the points coordinates and the flipping augmentation.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (alpha=0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (alpha=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of y.
    Also note that rotation of DepthInstance3DBoxes is counterclockwise,
    which is reverse to the definition of the yaw angle (clockwise).

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, alpha, beta, gamma).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self,
                 tensor,
                 box_dim=9,
                 with_yaw=True,
                 origin=(0.5, 0.5, 0.5)):

        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32,
                                                     device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # (0, 0, 0) as a fake euler angle.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 3)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 3
        elif tensor.shape[-1] == 7:
            assert box_dim == 7
            fake_euler = tensor.new_zeros(tensor.shape[0], 2)
            tensor = torch.cat((tensor, fake_euler), dim=-1)
            self.box_dim = box_dim + 2
        else:
            assert tensor.shape[-1] == 9
            self.box_dim = box_dim
        self.tensor = tensor.clone()

        self.origin = origin
        if origin != (0.5, 0.5, 0.5):
            dst = self.tensor.new_tensor((0.5, 0.5, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)
        self.with_yaw = with_yaw

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __getitem__(self, item: Union[int, slice, np.ndarray, torch.Tensor]):
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1),
                                 box_dim=self.box_dim,
                                 with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def dims(self) -> torch.Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou', eps=1e-4):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`EulerInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`EulerInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str): Mode of iou calculation. Defaults to 'iou'.
            eps (bool): Epsilon. Defaults to 1e-4.

        Returns:
            torch.Tensor: Calculated 3D overlaps of the boxes.
        """
        assert isinstance(boxes1, EulerDepthInstance3DBoxes)
        assert isinstance(boxes2, EulerDepthInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        corners1 = boxes1.corners
        corners2 = boxes2.corners
        _, iou3d = box3d_overlap(corners1, corners2, eps=eps)
        return iou3d

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3),
                     axis=1)).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin
        assert self.origin == (0.5, 0.5, 0.5), \
            'self.origin != (0.5, 0.5, 0.5) needs to be checked!'
        corners_norm = corners_norm - dims.new_tensor(self.origin)
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate
        corners = rotation_3d_in_euler(corners, self.tensor[:, 6:])

        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

from scipy.spatial.transform import Rotation as R
import numpy as np

def transform_scanrefer_bbox(bbox, extrinsic=None):
    center = bbox[0: 3]
    sizes = bbox[3:6]
    rot = R.from_euler("zxy", np.array(bbox[6:9]))
    if extrinsic is not None:
        center = (extrinsic @ np.array([*center, 1]).reshape(4, 1)).reshape(4)[:3].tolist()
        mat = extrinsic[:3, :3] @ rot.as_matrix()
        rot = R.from_matrix(mat)
    zxy = list(rot.as_euler("zxy"))

    return center + sizes + zxy

def box_transformer(axis_align_matrix, cam2global, pred_bbox):
    # frame_idx = pred_dict["frame"]
    extrinsic = np.array(axis_align_matrix) @ np.array(cam2global)
    pred_bbox = transform_scanrefer_bbox(pred_bbox, extrinsic)
    return pred_bbox

def rotation_3d_in_euler(points, angles, return_mat=False, clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple):
            Vector of angles in shape (N, 3)
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if len(angles.shape) == 1:
        angles = angles.expand(points.shape[:1] + (3, ))
        # angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 2 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_mat_T = euler_angles_to_matrix(angles, 'ZXY')  # N, 3,3
    rot_mat_T = rot_mat_T.transpose(-2, -1)

    if clockwise:
        raise NotImplementedError('clockwise')

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.bmm(points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new
    
class VGLLM_Inference:
    def __init__(self, pretrained):
        # load the model
        config = AutoConfig.from_pretrained(pretrained)
        self.model = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        ).eval()
        
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        
        # load the tokenizer
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")

    def call_model(
        self,
        contexts,
        visuals,
        add_frame_index: bool=False,
        gen_kwargs: dict = {},
    ):
        res = []
        messages = []
        processed_visuals = []
        print('contexts:', contexts)
        for i, context in enumerate(contexts):
    
            message = [{"role": "system", "content": "You are a helpful assistant."}]
    
            if len(visuals) > 0:
                visual = visuals[i] if i < len(visuals) else None
                if isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                    image_content = []
                    image_count = 0
                    for v in visual:
                        base64_image = v.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        if add_frame_index:
                            image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                        image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        image_count += 1
                    message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})
    
            messages.append(message)
    
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        geometry_encoder_inputs = []
        image_inputs = []
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        for message in messages:
            vision_info = extract_vision_info(message)
            cur_geometry_encoder_inputs = []
            for ele in vision_info:
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        # fix memory leak issue while using BytesIO
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")
    
                else:
                    raise NotImplementedError("Unsupported vision info type")
    
                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image = load_and_preprocess_images([image])[0]
                cur_geometry_encoder_inputs.append(copy.deepcopy(image))
                _, height, width = image.shape
                # merge_size = 2
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
    
            geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device
        if getattr(self.model.config, "use_geometry_encoder", False):
            inputs["geometry_encoder_inputs"] = [feat.to(device) for feat in geometry_encoder_inputs]
        inputs = inputs.to(device)
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 2048
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
    
        pad_token_id = self.tokenizer.pad_token_id
    
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )
    
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, ans in enumerate(answers):
            answers[i] = ans
    
        for ans, context in zip(answers, contexts):
            res.append(ans)
    
        return res

pretrained = "/mnt/data1/zhangjiaxin/project/ckpts/gapmllm-3d-3b"

model = VGLLM_Inference(pretrained)

# visualization
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from visualize_tools.img_drawer import ImageDrawer
from visualize_tools.color_selector import ColorMap

color_selctor = ColorMap()

def save_xyzrgb_ply(
    pointcloud: np.ndarray,
    save_path: str = "output.ply"
) -> None:
    if pointcloud.ndim != 2 or pointcloud.shape[1] != 6:
        raise ValueError("(N,6) and (x,y,z,r,g,b)")
    
    with open(save_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pointcloud)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        
        for p in pointcloud:
            x, y, z = p[:3]
            r, g, b = p[3:6].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"save to {save_path}")

def save_3d_boxes_as_ply(
    vis_boxes: dict,  # [x,y,z,l,w,h,rz,rx,ry]
    save_path: str = "boxes.ply",
    points_per_edge: int = 20, 
    axis_align_matrix: List[List[float]] = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
) -> None:

    all_box_points = []

    try:
        axis_align_np = np.array(axis_align_matrix, dtype=np.float64)
        inv_axis_align = np.linalg.inv(axis_align_np) 
    except np.linalg.LinAlgError:
        print("axis_align_matrix不可逆，使用原坐标")
        inv_axis_align = np.eye(4, dtype=np.float64) 
    
    for item in vis_boxes:
        label = item["label"]
        box = item["bbox_3d"]
        color = color_selctor.get_color(label)
        if len(box) != 9:
            continue 

        x, y, z = box[0], box[1], box[2]
        l, w, h = box[3], box[4], box[5]
        rz, rx, ry = box[6], box[7], box[8]
        
        half_l, half_w, half_h = l/2, w/2, h/2
        vertices = np.array([
            [half_l, -half_w, -half_h],  # 0
            [half_l,  half_w, -half_h],  # 1
            [-half_l, half_w, -half_h],  # 2
            [-half_l, -half_w, -half_h], # 3
            [half_l, -half_w, half_h],   # 4
            [half_l,  half_w, half_h],   # 5
            [-half_l, half_w, half_h],   # 6
            [-half_l, -half_w, half_h]   # 7
        ])
        
        def rotate_z(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        def rotate_x(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        
        def rotate_y(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        R = rotate_y(ry) @ rotate_x(rx) @ rotate_z(rz)
        vertices_rotated = (R @ vertices.T).T

        vertices_translated = vertices_rotated + np.array([x, y, z])

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), 
            (4, 5), (5, 6), (6, 7), (7, 4), 
            (0, 4), (1, 5), (2, 6), (3, 7)  
        ]
        
        for (v1_idx, v2_idx) in edges:
            v1 = vertices_translated[v1_idx]
            v2 = vertices_translated[v2_idx]
            t = np.linspace(0, 1, points_per_edge, endpoint=True)
            edge_points = v1 + t[:, None] * (v2 - v1)  # (points_per_edge, 3)
            edge_points_with_color = np.hstack([edge_points, np.tile(color, (points_per_edge, 1))])
            all_box_points.append(edge_points_with_color)
    
    if not all_box_points:
        print("None boxes")
        return

    all_points = np.vstack(all_box_points)
    
    coords = all_points[:, :3]  # (N, 3)
    coords_hom = np.hstack([coords, np.ones((len(coords), 1))])  # (N, 4)
    transformed_coords_hom = (inv_axis_align @ coords_hom.T).T  # (N, 4)
    transformed_coords = transformed_coords_hom[:, :3] 

    transformed_all_points = np.hstack([transformed_coords, all_points[:, 3:6]])  # (N, 6)
    save_xyzrgb_ply(transformed_all_points, save_path)

def visualize_3d_video_object_detection(sample, prediction, save_path=None):

    results = prediction.strip('\n').strip("```").strip("json").strip()
    lines = results.split('\n')
    # print(lines)
    bbox_list = []
    for line in lines:
        line = line.strip().strip(",")
        # print(line)
        if "bbox_3d" not in line and "label" not in line:
            continue
        try:
            pred_box = eval(line)
            bbox_list.append(pred_box)
        except Exception as e:
            print(f"Error parsing prediction bbox: {line}, Error: {e}")

    gt_bbox_dict = defaultdict(list)
    for bbox in sample["boxes"]:
        gt_bbox_dict[bbox["label"]].append(bbox["bbox_3d"])

    images = sample["images"]
    np_images = []
    for i, img in enumerate(images):
        drawer = ImageDrawer(os.path.join(media_dir, img))
        for item in bbox_list:
            if isinstance(item["bbox_3d"], str):
                item["bbox_3d"] = eval(item["bbox_3d"])
            center = np.array(item["bbox_3d"][:3]).reshape(3, 1)
            extent = np.array(item["bbox_3d"][3:6]).reshape(3, 1)
            rot = R.from_euler('zxy', np.array(item["bbox_3d"][6:9])).as_matrix() 
            geo = o3d.geometry.OrientedBoundingBox(center, rot, extent)
            intrinsic = np.array([[1170.18798828125, 0.0, 647.75, 0.0], 
                                 [0.0, 1170.18798828125, 483.75, 0.0], 
                                 [0.0, 0.0, 1.0, 0.0], 
                                 [0.0, 0.0, 0.0, 1.0]])
            extrinsic = np.linalg.inv(np.array(sample["cam2global"][0])) @ np.array(sample["cam2global"][i])
            try:
                color = color_selctor.get_color(item["label"])
            except:
                color = (0, 255, 0)
    
            drawer.draw_box3d(
                geo,
                color, 
                item["label"], 
                extrinsic, 
                intrinsic,
            )
        np_images.append(copy.deepcopy(drawer.img / 255.0))

    os.makedirs(save_path, exist_ok=True)

    for i, img in enumerate(np_images):
        img_save_path = os.path.join(save_path, f"output_test_image_{i}.png")
        plt.imsave(img_save_path, img, dpi=300)

    save_3d_boxes_as_ply(bbox_list, os.path.join(save_path, "boxes.ply"))



    # gt
    sorted_items = sorted(
        gt_bbox_dict.items(),
        key=lambda x: 1 if x[0] == "pillow" else 0
    )

    np_images = []
    for i, img in enumerate(images):
        drawer = ImageDrawer(os.path.join(media_dir, img))
        for key, value in sorted_items:
            if key == "door":
                continue
            # print(key)
            # print(value)
            # if isinstance(value, str):
            #     value = eval(value)
            center = np.array(value[0][:3]).reshape(3, 1)
            extent = np.array(value[0][3:6]).reshape(3, 1)
            rot = R.from_euler('zxy', np.array(value[0][6:9])).as_matrix() 
            geo = o3d.geometry.OrientedBoundingBox(center, rot, extent)
            intrinsic = np.array([[1170.18798828125, 0.0, 647.75, 0.0], 
                                 [0.0, 1170.18798828125, 483.75, 0.0], 
                                 [0.0, 0.0, 1.0, 0.0], 
                                 [0.0, 0.0, 0.0, 1.0]])
            extrinsic = np.linalg.inv(np.array(sample["cam2global"][0])) @ np.array(sample["cam2global"][i])
            try:
                color = color_selctor.get_color(key)
            except:
                color = (0, 255, 0)
    
            drawer.draw_box3d(
                geo,
                color, 
                key, 
                extrinsic, 
                intrinsic,
            )
        np_images.append(copy.deepcopy(drawer.img / 255.0))

    os.makedirs(save_path, exist_ok=True)

    for i, img in enumerate(np_images):
        img_save_path = os.path.join(save_path, f"gt_image_{i}.png")
        plt.imsave(img_save_path, img, dpi=300)

def compute_ap(gt_bbox_dict, pred_bbox_dict, iou_threshold=0.25):
    used_gt = defaultdict(set)
    iou = 0
    for category in pred_bbox_dict:
        for bbox in pred_bbox_dict[category]:
            gt_box_match = -1
            max_iou = 0
            for i, gt_box in enumerate(gt_bbox_dict[category]):
                if i in used_gt[category]:
                    continue
                try:
                    iou = EulerDepthInstance3DBoxes.overlaps(
                        EulerDepthInstance3DBoxes(torch.tensor([bbox])),
                        EulerDepthInstance3DBoxes(torch.tensor([gt_box]))
                    )
                except Exception as e:
                    print(f"Error calculating IOU: {e}")
                    iou = 0
                if iou > max_iou:
                    max_iou = iou
                    gt_box_match = i
            
            if max_iou > iou_threshold:
                iou += max_iou
                used_gt[category].add(gt_box_match)
    if len(gt_bbox_dict) == 0:
        return 0
    return iou/len(gt_bbox_dict)

def miou_get(gt_sample, output):
    prediction = output
    results = prediction.strip('\n').strip("```").strip("json").strip()
    lines = results.split('\n')
    bbox_list = []
    for line in lines:
        line = line.strip().strip(",")
        # print(line)
        if "bbox_3d" not in line and "label" not in line:
            continue
        try:
            pred_box = eval(line)
            bbox_list.append(pred_box)
        except Exception as e:
            print(f"Error parsing prediction bbox: {line}, Error: {e}")

    pred_bbox_dict = defaultdict(list)
    gt_bbox_dict = defaultdict(list)

    for bbox in gt_sample["boxes"]:
        gt_bbox_dict[bbox["label"]].append(bbox["bbox_3d"])

    for bbox in bbox_list:
        try:
            pred_bbox = np.array(bbox["bbox_3d"], dtype=float)
            pred_bbox_dict[bbox["label"]].append(pred_bbox)
        except Exception as e:
            print(f"Error parsing prediction bbox: {bbox}, Error: {e}")

    iou = compute_ap(gt_bbox_dict, pred_bbox_dict)
    map25, map50 = compute_map25_map50(gt_bbox_dict, pred_bbox_dict)
    global_f1, cat_f1_dict = compute_f1_iou25(gt_bbox_dict, pred_bbox_dict)
    print(global_f1)
    print(iou)
    print(map25)
    print(map50)

def compute_3d_iou(box1, box2):
    try:
        return EulerDepthInstance3DBoxes.overlaps(
                        EulerDepthInstance3DBoxes(torch.tensor([box1])),
                        EulerDepthInstance3DBoxes(torch.tensor([box2]))
                    ).item()
    except:
        return 0.0

def compute_map25_map50(gt_bbox_dict, pred_bbox_dict):

    all_cats = set(gt_bbox_dict.keys()).union(set(pred_bbox_dict.keys()))
    # print(all_cats)
    if not all_cats:
        return 0.0, 0.0

    ap25_total, ap50_total = 0.0, 0.0
    for cat in all_cats:
        # print(cat)
        gt_boxes = gt_bbox_dict.get(cat, [])
        pred_boxes = pred_bbox_dict.get(cat, [])
        # print(gt_boxes)
        # print(pred_boxes)
        num_gt, num_pred = len(gt_boxes), len(pred_boxes)
        if num_gt == 0 or num_pred == 0:
            ap25, ap50 = 0.0, 0.0
        else:
            ap25 = _calc_single_ap(gt_boxes, pred_boxes, 0.1)
            ap50 = _calc_single_ap(gt_boxes, pred_boxes, 0.5)
            # print(ap25)
        ap25_total += ap25
        ap50_total += ap50

    map25 = ap25_total / len(all_cats)
    map50 = ap50_total / len(all_cats)
    return map25, map50

def _calc_single_ap(gt_boxes, pred_boxes, iou_th):
    used_gt = [False] * len(gt_boxes)
    tp, fp = [0]*len(pred_boxes), [0]*len(pred_boxes)

    for i, p_box in enumerate(pred_boxes):
        max_iou, match_idx = 0.0, -1
        for j, g_box in enumerate(gt_boxes):
            if not used_gt[j]:
                iou = compute_3d_iou(p_box, g_box)
                if iou > max_iou:
                    max_iou = iou
                    match_idx = j
        if max_iou > iou_th and match_idx != -1:
            tp[i] = 1
            used_gt[match_idx] = True
        else:
            fp[i] = 1

    cum_tp = [sum(tp[:k+1]) for k in range(len(pred_boxes))]
    cum_fp = [sum(fp[:k+1]) for k in range(len(pred_boxes))]
    precision = [t/(t+f) if t+f>0 else 0.0 for t,f in zip(cum_tp, cum_fp)]
    recall = [t/len(gt_boxes) for t in cum_tp]

    ap, max_p = 0.0, 0.0
    for r, p in sorted(zip(recall, precision), key=lambda x: x[0], reverse=True):
        max_p = max(max_p, p)
        ap += max_p * (r if recall.index(r)==0 else r - recall[recall.index(r)-1])
    return ap

def compute_f1_iou25(gt_bbox_dict, pred_bbox_dict, iou_threshold=0.25):

    all_cats = set(gt_bbox_dict.keys()).union(set(pred_bbox_dict.keys()))
    global_tp, global_fp, global_fn = 0, 0, 0
    cat_f1_dict = {}

    for cat in all_cats:
        gt_boxes = gt_bbox_dict.get(cat, [])
        pred_boxes = pred_bbox_dict.get(cat, [])
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)
        used_gt = [False] * num_gt  
        tp, fp = 0, 0

        for p_box in pred_boxes:
            max_iou, match_idx = 0.0, -1
            for j, g_box in enumerate(gt_boxes):
                if not used_gt[j]:
                    iou = compute_3d_iou(p_box, g_box)
                    if iou > max_iou:
                        max_iou = iou
                        match_idx = j
            if max_iou > iou_threshold and match_idx != -1:
                tp += 1
                used_gt[match_idx] = True
            else:
                fp += 1
        fn = num_gt - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        cat_f1_dict[cat] = f1

        global_tp += tp
        global_fp += fp
        global_fn += fn

    global_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = 2 * global_p * global_r / (global_p + global_r) if (global_p + global_r) > 0 else 0.0

    return global_f1, cat_f1_dict

def degrade_first_frame(
    img,
    mode="blur",
    blur_radius=13,
    occlusion_ratio=0.5,
    occlusion_color=(0, 0, 0),
    random_occlusion=True,
):
    if mode is None or mode == "none":
        return img

    degraded = img.copy()

    if mode in ["blur", "blur_occlusion"]:
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if mode in ["occlusion", "blur_occlusion"]:
        w, h = degraded.size
        occ_w = int(w * occlusion_ratio)
        occ_h = int(h * occlusion_ratio)

        if random_occlusion:
            x0 = random.randint(0, max(0, w - occ_w))
            y0 = random.randint(0, max(0, h - occ_h))
        else:
            x0 = (w - occ_w) // 2
            y0 = (h - occ_h) // 2

        x1 = x0 + occ_w
        y1 = y0 + occ_h

        draw = ImageDraw.Draw(degraded)
        draw.rectangle([x0, y0, x1, y1], fill=occlusion_color)

    return degraded

def process_3d_video_object_detection(sample, output_path):
    text = sample["conversations"][0]["value"].replace("<image>", "")
    image_files = sample["images"]
    images = [
        Image.open(
            os.path.join(media_dir, image_file)
        ).convert("RGB")
        for image_file in image_files
    ]

    print(text)
    output = model.call_model(
        contexts=[text],
        visuals=[images],
    )[0]

    print("Output:", output)
    miou_get(sample, output)

    visualize_3d_video_object_detection(sample, output, output_path)

demo_data_dir = "demo"
task = "3d_video_object_detection"
with open(os.path.join(demo_data_dir, task, "sample.json"))as f:
    data = json.load(f)
    
filtered_data = []
for item in data:
    first_img = item["images"][0]
    scene_name = first_img.split("/")[-2]
    img_name = first_img.split("/")[-1]
    filtered_data.append(item)

media_dir = "demo/3d_video_object_detection/media"

for data_item in filtered_data:
    first_img = data_item["images"][0]
    scene_name = first_img.split("/")[-2]
    img_name = first_img.split("/")[-1]
    output_path = os.path.join("vis/qwen3", scene_name+"-"+img_name)
    process_3d_video_object_detection(data_item, output_path)