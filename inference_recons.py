import os
import math
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
from scipy.spatial import cKDTree as KDTree

min_pixels: int = 256 * 28 * 28
max_pixels: int = 1605632
max_num_frames: int = 32

def get_device_type():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

if "HCCL_CONNECT_TIMEOUT" not in os.environ:
    os.environ["HCCL_CONNECT_TIMEOUT"] = "7200"  # Unit: seconds
    
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
            gen_kwargs["max_new_tokens"] = 1280
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
    """Run inference on each device and keep results aligned with the original data order."""
    setup(rank, world_size, node_rank, nodes)
    total_devices = world_size * nodes
    # Keep original sample indices during sharding so results can be realigned later.
    total_samples = len(data_samples)
    global_rank = node_rank * world_size + rank
    # Round-robin assignment: rank 0 takes 0, world_size, 2*world_size; rank 1 takes 1, world_size+1, ...
    local_indices = [i for i in range(global_rank, total_samples, total_devices)]
    # Select samples for the current GPU by original index.
    local_samples = [data_samples[i] for i in local_indices]
    
    # Print assignment information for verification.
    start_idx = local_indices[0] if local_indices else None
    end_idx = local_indices[-1] if local_indices else None
    count = len(local_samples)
    
    print(f"Rank {global_rank} processing samples {start_idx}-{end_idx} ({count} samples)")
    model = VGLLM_Inference(pretrained_path, rank)
    results = []  # Store records associated with the original index and output.
    
    if global_rank == 0:
        pbar = tqdm(total=total_samples, desc="Overall Progress")
    
    for i, (sample, original_idx) in enumerate(zip(local_samples, local_indices)):
        # 1. Run inference and save the output.
        output = process_3d_video_object_detection(model, sample, data_path)
        
        # 2. Compute the score.
        ret = calculate_iou_for_sample(sample, output)
        
        # 3. Record the original index for later sorting, along with output and score.
        results.append({
            "original_index": original_idx,  # Keep this sample's original position in data_samples.
            "gt": sample["conversations"][1]["value"],
            "output": output,                # Inference output.
            "score": ret
        })
        
        if global_rank == 0:
            pbar.update(total_devices)
    
    # Each process writes its local results to a temporary file.
    local_output_file = os.path.join(output_path, f"temp_rank_{global_rank}.json")
    with open(local_output_file, 'w') as f:
        # Convert non-serializable objects first if output contains any tensors.
        json.dump(results, f, indent=2)
    
    # Wait until all processes finish writing files.
    dist.barrier()
    
    # The main process merges results and restores the original order.
    if global_rank == 0:
        final_results = []
        # Read all temporary files.
        for r in range(total_devices):
            rank_file = os.path.join(output_path, f"temp_rank_{r}.json")
            with open(rank_file, 'r') as f:
                final_results.extend(json.load(f))
            os.remove(rank_file)  # Clean up the temporary file.

        # Sort by original index to match data_samples exactly.
        final_results.sort(key=lambda x: x["original_index"])
        aggregated = score_aggregate_results(final_results, data_samples)
        print("Aggregated result:", aggregated)  # Print the aggregated overall metric.
        # Optionally remove original indices if they do not need to be retained.
        # for item in final_results:
        #     item.pop("original_index", None)
        
        # Save final results.
        output_file = os.path.join(output_path, "inference_results_with_output.json")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"Results saved to: {output_file} (order matches data_samples)")
        
    
    cleanup()

def compute_ap(gt, pred):
    if pred == []:
        return 0.0
    pred_point = pred[0]['pointmap']
    gt_point = gt[0]['pointmap']
    dist = math.sqrt((pred_point[0]-gt_point[0])**2 + 
                     (pred_point[1]-gt_point[1])**2 + 
                     (pred_point[2]-gt_point[2])**2)
    # Option 1: normalized score. D_max can be adjusted.
    D_max = 1
    score = max(0.0, 1.0 - dist/D_max)
    return score


def calculate_iou_for_sample(doc, results):
    results = results.strip('\n').strip("```").strip("json").strip()
    lines = results.split('\n')
    # print(lines)
    pred = []
    for line in lines:
        line = line.strip().strip(",")
        # print(line)
        # if "frame" not in line and "pointmap" not in line:
        #     continue
        if "pointmap" not in line:
            continue
        try:
            pred_pointmap = eval(line)
            pred.append(pred_pointmap)
        except Exception as e:
            print(f"Error parsing prediction bbox: {line}, Error: {e}")

    gt_result = doc["conversations"][1]["value"]
    gt_result = gt_result.strip('\n').strip("```").strip("json").strip()
    gt_lines = gt_result.split('\n')
    # print(lines)
    gt = []
    for line in gt_lines:
        line = line.strip().strip(",")
        # print(line)
        if "pointmap" not in line:
            continue
        try:
            gt_pointmap = eval(line)
            gt.append(gt_pointmap)
        except Exception as e:
            print(f"Error parsing prediction bbox: {line}, Error: {e}")

    ret = compute_ap(gt, pred)
    for g, p in zip(gt, pred):
        print("GT:", g, " PRED:", p, f" score: {ret:.4f}")
    return ret

def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median

# def score_aggregate_results(results):

#     all_results = []
#     for result in results:
#         all_results.append(result['score'])

#     mean_score = sum(all_results) / len(all_results)
#     # print(f"Average score: {mean_score:.4f}")
#     return mean_score

def score_aggregate_results(results, data_samples):
    """
    Evaluate point-cloud reconstruction per scene with raw and aligned metrics,
    then report the global mean.
    results: inference result list containing original_index.
    data_samples: full original sample list matched by original_index.
    """
    # 1. Attach each inference result to its original sample and extract the scene.
    scene_groups = {}

    for i, res in enumerate(results):
        sample = data_samples[i]  # Order is already aligned, so direct indexing is valid.
        first_img = sample["images"][0]
        scene = first_img.split("/")[-2]
        
        if scene not in scene_groups:
            scene_groups[scene] = []
        scene_groups[scene].append(res)

    # 2. Compute metrics scene by scene.
    scene_metrics = {}
    valid_scenes = []

    for scene, sample_results in scene_groups.items():

        gt_list = []
        pred_list = []
        for s_res in sample_results:
            try:
                # Parse the GT point cloud.
                gt_str = s_res["gt"].strip().strip("```").strip("json").strip()
                gt_dict = eval(gt_str)
                gt_pc = np.array(gt_dict["pointmap"], dtype=np.float32)

                # Parse the predicted point cloud.
                pred_str = s_res["output"].strip().strip("```").strip("json").strip()
                pred_dict = eval(pred_str)
                pred_pc = np.array(pred_dict["pointmap"], dtype=np.float32)

                gt_list.append(gt_pc)
                pred_list.append(pred_pc)
            except Exception as e:
                print(f"[{scene}] Failed to parse sample: {e}")
                continue

        if len(gt_list) == 0:
            continue

        gt_all = np.vstack(gt_list)
        pred_all = np.vstack(pred_list)

        # Raw metrics.
        acc_metric, acc_med_metric = accuracy(gt_all, pred_all)
        comp_metric, comp_med_metric = completion(gt_all, pred_all)
        overall_metric = (acc_metric + comp_metric) / 2

        # Metrics after Umeyama alignment.
        X, Y = pred_all.T, gt_all.T
        c, R, t = umeyama(X, Y)
        pred_aligned = (c * R @ X + t).T.astype(np.float32)

        acc_align, acc_med_align = accuracy(gt_all, pred_aligned)
        comp_align, comp_med_align = completion(gt_all, pred_aligned)
        overall_align = (acc_align + comp_align) / 2

        scene_metrics[scene] = {
            "acc_metric": acc_metric,
            "comp_metric": comp_metric,
            "overall_metric": overall_metric,
            "acc_align": acc_align,
            "comp_align": comp_align,
            "overall_align": overall_align,
        }
        valid_scenes.append(scene)

    # 3. Compute the global mean over all scenes.
    metric_keys = ["acc_metric", "comp_metric", "overall_metric",
                   "acc_align", "comp_align", "overall_align"]
    global_mean = {}
    for k in metric_keys:
        vals = [scene_metrics[s][k] for s in valid_scenes]
        global_mean[k] = np.mean(vals)

    # 4. Print the table.
    print("\n" + "="*90)
    print(" 3D Point Cloud Reconstruction (PER SCENE + GLOBAL MEAN) ")
    print("="*90)
    table_rows = [["Scene", "Acc(M)", "Comp(M)", "Overall(M)", "Acc(A)", "Comp(A)", "Overall(A)"]]
    for sc in valid_scenes:
        m = scene_metrics[sc]
        table_rows.append([
            sc,
            f"{m['acc_metric']:.3f}", f"{m['comp_metric']:.3f}", f"{m['overall_metric']:.3f}",
            f"{m['acc_align']:.3f}", f"{m['comp_align']:.3f}", f"{m['overall_align']:.3f}"
        ])
    # Global mean row.
    g = global_mean
    table_rows.append([
        "GLOBAL MEAN",
        f"{g['acc_metric']:.3f}", f"{g['comp_metric']:.3f}", f"{g['overall_metric']:.3f}",
        f"{g['acc_align']:.3f}", f"{g['comp_align']:.3f}", f"{g['overall_align']:.3f}"
    ])
    print(AsciiTable(table_rows).table)
    print("="*90 + "\n")

    # Keep compatibility with the old logic by returning global aligned overall.
    return global_mean["overall_align"]

def sample_meta(scene, data_path):
    pixel_coords = scene["pixel_coords"]
    images = scene["images"]
    image_file = []
    view_idx, scaled_pixel_x, scaled_pixel_y = pixel_coords[0], pixel_coords[1], pixel_coords[2]
    for idx, img_path in enumerate(images):
        # Build the full image path.
        full_img_path = os.path.join(data_path, img_path)
        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Image not found: {full_img_path}")
        # Open and mark the image.
        with Image.open(full_img_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            if view_idx == idx:
                cross_size = 10  # Cross arm length.
                thickness = 3    # Line thickness in pixels.
                
                # Draw the thick horizontal line with vertical offsets.
                for dy in range(-thickness + 1, thickness):
                    for dx in range(-cross_size, cross_size + 1):
                        x = scaled_pixel_x + dx
                        y = scaled_pixel_y + dy
                        if 0 <= x < width and 0 <= y < height:
                            img.putpixel((x, y), (255, 0, 0))
                
                # Draw the thick vertical line with horizontal offsets.
                for dx in range(-thickness + 1, thickness):
                    for dy in range(-cross_size, cross_size + 1):
                        x = scaled_pixel_x + dx
                        y = scaled_pixel_y + dy
                        if 0 <= x < width and 0 <= y < height:
                            img.putpixel((x, y), (255, 0, 0))
            image_file.append(img)
    return scene, image_file

def process_3d_video_object_detection(model, scene, data_path):
    sample, images = sample_meta(scene, data_path)
    text = sample["conversations"][0]["value"].replace("<image>", "")
    output = model.call_model(
        contexts=[text],
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
    parser.add_argument("--data_path", type=str, help="Data path")
    parser.add_argument("--output_path", type=str,  help="Output path")
    args = parser.parse_args()

    # Set environment variables.
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # Get the number of devices per node.
    if args.world_size is None:
        args.world_size = get_available_devices()
    
    # Load data.

    with open(os.path.join("data/evaluation/recons", "scannet_recons_val_4frames_full_image.json")) as f:
        data = json.load(f)
    

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
