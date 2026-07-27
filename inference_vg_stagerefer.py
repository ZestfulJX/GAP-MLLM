import os
import json
import copy
import torch
import base64
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from io import BytesIO
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.data.utils import load_and_preprocess_images
from qwen_vl.model.modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT
import numpy as np
from pytorch3d.ops import box3d_overlap
from typing import Dict, Optional, Sequence, List, Tuple, Any, Union
from pytorch3d.transforms import euler_angles_to_matrix
from tqdm import tqdm
import argparse
from collections import defaultdict
from terminaltables import AsciiTable
# hyperparameters inherited from Qwen2.5-VL
min_pixels: int = 256 * 28 * 28
max_pixels: int = 1605632
max_num_frames: int = 32

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

# Detect the device type.
def get_device_type():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device_type = get_device_type()
print(f"Using device: {device_type}")

class VGLLM_Inference:
    def __init__(self, pretrained, local_rank):
        self.local_rank = local_rank
        self.device_type = device_type
        
        # Set the device.
        if device_type == "npu":
            self.device = torch.device(f"npu:{local_rank}")
            torch.npu.set_device(self.device)
        elif device_type == "cuda":
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
        
        # load the model
        config = AutoConfig.from_pretrained(pretrained)
        
        # NPU-specific configuration.
        if device_type == "npu":
            print(getattr(config, "geometry_encoder_type", "none"))
            print(getattr(config, "geometry_encoder_path", "none"))
            self.model = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map={"": local_rank},
                attn_implementation="flash_attention_2",
                # NPU does not support flash_attention_2; use the default implementation.
            ).eval()
            
            # Move the model to NPU.
            self.model = self.model.to(self.device)
            
        else:
            self.model = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map={"": local_rank},
                attn_implementation="flash_attention_2",
            ).eval()
        
        # Use DDP only in multi-device settings.
        if device_type in ["npu", "cuda"] and torch.distributed.is_available():
            self.model = DDP(self.model, device_ids=[local_rank])
        
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        
        # load the processor
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
        
        # Move inputs to the corresponding device.
        if device_type == "npu":
            inputs = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
            
        if getattr(self.model.module.config if hasattr(self.model, 'module') else self.model.config, "use_geometry_encoder", False):
            if device_type == "npu":
                inputs["geometry_encoder_inputs"] = [feat.npu() for feat in geometry_encoder_inputs]
            else:
                inputs["geometry_encoder_inputs"] = [feat.to(self.device) for feat in geometry_encoder_inputs]
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 4096
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
    
        pad_token_id = self.tokenizer.pad_token_id
    
        # Get the underlying model if it is wrapped by DDP.
        model_to_generate = self.model.module if hasattr(self.model, 'module') else self.model
    
        cont = model_to_generate.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )
    
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, ans in enumerate(answers):
            answers[i] = ans
    
        for ans, context in zip(answers, contexts):
            res.append(ans)
    
        return res

def setup(rank, world_size, node_rank, nodes):
    """Set up the distributed inference environment."""
    # Get the master address and port from environment variables or parameters.
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Compute global rank.
    global_rank = node_rank * world_size + rank
    
    if device_type == "npu":
        dist.init_process_group("hccl", rank=global_rank, world_size=world_size*nodes)
    else:
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size*nodes)

def cleanup():
    """Clean up the distributed inference environment."""
    dist.destroy_process_group()


def run_inference(rank, world_size, node_rank, nodes, data_samples, pretrained_path, data_path, output_path):
    setup(rank, world_size, node_rank, nodes)
    
    # Recompute data sharding across all nodes.
    total_devices = world_size * nodes
    total_samples = len(data_samples)
    samples_per_device = total_samples // total_devices
    global_rank = node_rank * world_size + rank
    start_idx = global_rank * samples_per_device
    end_idx = start_idx + samples_per_device if global_rank < total_devices - 1 else total_samples


    local_samples = data_samples[start_idx:end_idx]
    
    print(f"Rank {rank} processing {len(local_samples)} samples")
    
    # Initialize the model.
    model = VGLLM_Inference(pretrained_path, rank)
    
    results = []
    
    if rank == 0:
        # Add the overall progress bar.
        pbar = tqdm(total=len(data_samples), desc="Overall Progress")
    
    for i, sample in enumerate(local_samples):
        # if rank == 0:  # Print progress only in the main process.
        #     print(f"Processing sample {start_idx + i + 1}/{len(data_samples)}")
        

        # Process 3D visual grounding.
        output = process_3d_visual_grounding(model, sample, data_path)

        prediction = output.strip('\n').strip("```").strip("json").strip()
        # frame_data = json.loads(prediction)['frame']
        try:
            prediction_dict = json.loads(prediction)
            frame_data = prediction_dict['frame'] if 'frame' in prediction_dict else 0
        except json.JSONDecodeError:
            frame_data = 0

        output = process_3d_visual_grounding(model, sample, data_path, frame_data)
        
        # Compute IoU.
        iou_score, pred_bbox = calculate_iou_for_sample(sample, output, frame_data)
        
        results.append({
            'sample_id': start_idx + i,
            'query': sample["prompt"].split("\n")[1],
            'prediction': output,
            'iou': iou_score,
            'gt_box': sample['gt_bbox'],
            'pred_bbox_global_axis': pred_bbox,
            "images": sample["images"]
        })

        if rank == 0:
            print(f"Sample {start_idx + i}: IOU = {iou_score:.4f}")
            pbar.update(world_size)  # Each GPU processes a subset of the data.
            pbar.set_postfix({"Current IOU": f"{iou_score:.4f}"})
                
    
    # Each process writes its local results to a temporary file.
    local_output_file = os.path.join(output_path, f"temp_rank_{rank}.json")
    with open(local_output_file, 'w') as f:
        # Convert non-serializable objects first if output contains any tensors.
        json.dump(results, f, indent=2)
    
    # Wait until all processes finish writing files.
    dist.barrier()
    
    # The main process merges results and restores the original order.
    if rank == 0:
        final_results = []
        # Read all temporary files.
        for r in range(world_size):
            rank_file = os.path.join(output_path, f"temp_rank_{r}.json")
            with open(rank_file, 'r') as f:
                final_results.extend(json.load(f))
            os.remove(rank_file)  # Clean up the temporary file.
        
        # Sort by sample_id.
        final_results.sort(key=lambda x: x['sample_id'])
        aggregated = refer_aggregate_results(final_results)
        print("Aggregated result:", aggregated)  # Print the aggregated overall metric.
        # Save results.
        output_file = "inference_results_with_output.json"
        with open(os.path.join(output_path, output_file), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        # Compute average IoU.
        valid_ious = [r['iou'] for r in final_results if 'iou' in r and isinstance(r['iou'], (int, float))]
        if valid_ious:
            avg_iou = sum(valid_ious) / len(valid_ious)
            print(f"Average IOU: {avg_iou:.4f},")
    
    cleanup()

def calculate_iou_for_sample(sample, prediction, frame_data):
    """Compute IoU for a single sample."""
    # Parse prediction results.
    try:
        prediction = prediction.strip('\n').strip("```").strip("json").strip()
        bbox_list = [eval(prediction)]
        
        # Get the ground-truth bounding box.
        gt_bbox = sample["gt_bbox"]
        
        # Compute IoU between each predicted box and the ground-truth box.
        ious = []
        for pred_bbox in bbox_list:
            try:
                extrinsic = np.array(sample["axis_align_matrix"]) @ np.array(sample["cam2global"][frame_data])
                pred_bbox = transform_scanrefer_bbox(pred_bbox["bbox_3d"], extrinsic)
                iou = EulerDepthInstance3DBoxes.overlaps(
                    EulerDepthInstance3DBoxes(torch.tensor([pred_bbox])),
                    EulerDepthInstance3DBoxes(torch.tensor([gt_bbox]))
                ).item()
            except:
                iou = 0
            ious.append(iou)
        
        # Return the maximum IoU.
        return max(ious) if ious else 0.0, pred_bbox
    except:
        return 0, [0]*9

def refer_aggregate_results(results):
    """
    Follow the 3D detection metric style and report IoU@0.25, IoU@0.50, and mIoU
    for the referring task. Metrics are printed but not saved.
    """
    total = len(results)
    sum_iou = 0.0
    iou25 = 0
    iou50 = 0

    for res in results:
        iou = res['iou']
        sum_iou += iou
        if iou >= 0.25:
            iou25 += 1
        if iou >= 0.50:
            iou50 += 1

    # Compute metrics.
    miou = sum_iou / total if total > 0 else 0.0
    acc25 = iou25 / total if total > 0 else 0.0
    acc50 = iou50 / total if total > 0 else 0.0

    # Print a table in the original style.
    print("\n====== Refer Expression Evaluation Results ======")
    table_data = [
        ["Total Samples", "mIOU", "IoU@0.25", "IoU@0.50"],
        [str(total), f"{miou:.4f}", f"{acc25:.4f}", f"{acc50:.4f}"]
    ]
    table = AsciiTable(table_data)
    print(table.table)

    # Detailed hit counts.
    detail_table = [
        ["IoU Threshold", "Correct Count", "Total"],
        [">= 0.25", str(iou25), str(total)],
        [">= 0.50", str(iou50), str(total)],
    ]
    detail = AsciiTable(detail_table)
    detail.title = "Detailed Count"
    print(detail.table)
    print("="*60 + "\n")

    return miou

def extract_text_description(input_text):
    # Find the position of "Text:".
    text_start = input_text.find("Text:")
    if text_start == -1:
        return None
    
    # Find where the content after "Text:" starts.
    text_content_start = text_start + len("Text:")
    
    # Find the start of the next major section, such as "Output a JSON".
    next_section_start = input_text.find("Output a JSON", text_content_start)
    
    if next_section_start != -1:
        # Extract the content between "Text:" and "Output".
        description = input_text[text_content_start:next_section_start].strip()
    else:
        # If no next section is found, extract everything after "Text:".
        description = input_text[text_content_start:].strip()
    
    return description

def process_3d_visual_grounding(model, sample, data_path, frame_data = None):
    """Process the 3D visual grounding task."""
    # text = sample["prompt"]
    # text = sample["conversations"][0]["value"].replace("<image>", "")
    if frame_data == None:
        text = sample["conversations"][0]["value"]
        description = extract_text_description(text)
        final_text = f"Frame-0: <image>Frame-1: <image>Frame-2: <image>Frame-3: <image>Frame-4: <image>Frame-5: <image>Frame-6: <image>Frame-7: <image>Frame-8: <image>Frame-9: <image>Frame-10: <image>Frame-11: <image>Frame-12: <image>Frame-13: <image>Frame-14: <image>Frame-15: <image>Frame-16: <image>Frame-17: <image>Frame-18: <image>Frame-19: <image>Frame-20: <image>Frame-21: <image>Frame-22: <image>Frame-23: <image>Frame-24: <image>Frame-25: <image>Frame-26: <image>Frame-27: <image>Frame-28: <image>Frame-29: <image>Frame-30: <image>Frame-31: <image>\nLocalize the first clear frame in the video showing the object described in the text.\nText: {description}\nOutput a JSON dictionary with the frame index in \"frame\".\n"

        # Insert the extracted description into the template.
        # new_text = template.format(description=description)
        image_files = sample["images"]
        images = [
            Image.open(
                os.path.join(data_path, image_file)
            ).convert("RGB")
            for image_file in image_files
        ]
        # text_list = new_text.split('\n')[1:]
        # final_text = '\n'.join(text_list)
        output = model.call_model(
            contexts=[final_text],
            visuals=[images],
            add_frame_index=True,
        )[0]
    else:
        text = sample["conversations"][0]["value"].replace("<image>", "")
        description = extract_text_description(text)
        step = 10
        image_files = sample["images"]
        selected_image_path = image_files[frame_data]
        filename = os.path.basename(selected_image_path)
        base_num = int(filename.split('.')[0])
        start_num = max(0, base_num - 90)
        end_num = base_num + 90
        target_nums = []
        target_nums.append(base_num)
        current_num = base_num + step
        while current_num <= end_num:
            target_nums.append(current_num)
            current_num += step
        current_num = base_num - step
        while current_num >= start_num:
            target_nums.append(current_num)  # Insert after base_num.
            current_num -= step
        stage2_images = []
        # String format of the original number, used for replacement.
        base_num_str = f"{base_num:05d}"  # 100 -> "00100"
        for num in target_nums:
            num_str = f"{num:05d}"  # Format as five digits with zero padding.
            target_path = selected_image_path.replace(base_num_str, num_str)
            stage2_images.append(target_path)
        images = []

        for image_file in stage2_images:
            img_path = os.path.join(data_path, image_file)
            
            if os.path.exists(img_path) and os.path.isfile(img_path):
                img = Image.open(img_path).convert("RGB")
                images.append(img)
        images_proc = "<image>" * len(images)
        final_text = f"{images_proc}Localize the object described in the text.\nText: {description}\nOutput a JSON dictionary with the matched object's 3D bounding box in \"bbox_3d\" in the camera coordinate system of the first frame.\nThe 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, z_size, yaw, pitch, roll].\n"
        output = model.call_model(
            contexts=[final_text],
            visuals=[images],
        )[0]

    return output

def get_available_devices():
    """Get the number of available devices."""
    if device_type == "npu":
        return torch.npu.device_count()
    elif device_type == "cuda":
        return torch.cuda.device_count()
    else:
        return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Current node rank")
    parser.add_argument("--world_size", type=int, default=None, help="GPUs per node")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master_port", type=str, default="12355", help="Master node port")
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint path")
    parser.add_argument("--data_path", type=str, default="data", help="Data path")
    parser.add_argument("--output_path", type=str,  help="Output path")
    args = parser.parse_args()
    
    # # # Set environment variables.
    # os.environ['MASTER_ADDR'] = args.master_addr
    # os.environ['MASTER_PORT'] = args.master_port
    
    # Get the number of devices per node.
    if args.world_size is None:
        args.world_size = get_available_devices()
    
    # Load data.
    with open(os.path.join("data/evaluation/scanrefer", "scanrefer_val_32frames.json")) as f:
        data = json.load(f)

    # with open(os.path.join("data/evaluation/scanrefer", "scanrefer_val_32frames_id_visible_frame.json")) as f:
    #     data = json.load(f)

    
    pretrained_path = args.ckpt_path
    
    print(f"Node {args.node_rank}: Starting with {args.world_size} devices")
    os.makedirs(args.output_path, exist_ok=True)

    if args.world_size > 1:
        mp.spawn(
            run_inference,
            args=(args.world_size, args.node_rank, args.nodes, data, pretrained_path, args.data_path, args.output_path),
            nprocs=args.world_size,
            join=True
        )
    else:
        run_inference(0, 1, args.node_rank, args.nodes, data, pretrained_path, args.data_path, args.output_path)

if __name__ == "__main__":
    main()
