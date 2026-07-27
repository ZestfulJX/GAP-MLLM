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
import sys
from pathlib import Path
# hyperparameters inherited from Qwen2.5-VL
project_root = Path(__file__).parent.parent.parent
print(project_root)
sys.path.append(str(project_root))

min_pixels: int = 256 * 28 * 28
max_pixels: int = 1605632
max_num_frames: int = 32

if "HCCL_CONNECT_TIMEOUT" not in os.environ:
    os.environ["HCCL_CONNECT_TIMEOUT"] = "3600"  # Unit: seconds

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

def setup(rank, world_size):
    """Set up the distributed inference environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if device_type == "npu":
        # Use the HCCL backend for NPU.
        dist.init_process_group("hccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed inference environment."""
    dist.destroy_process_group()

def run_inference(rank, world_size, data_samples, pretrained_path, data_path, output_path):
    """Run inference on each device and keep results aligned with the original data order."""
    setup(rank, world_size)
    
    # Use round-robin assignment so each GPU receives a balanced sample count.
    total_samples = len(data_samples)
    
    # Round-robin assignment: rank 0 takes 0, world_size, 2*world_size; rank 1 takes 1, world_size+1, ...
    local_indices = [i for i in range(rank, total_samples, world_size)]
    # Select samples for the current GPU by original index.
    local_samples = [data_samples[i] for i in local_indices]
    
    # Print assignment information for verification.
    start_idx = local_indices[0] if local_indices else None
    end_idx = local_indices[-1] if local_indices else None
    count = len(local_samples)
    
    print(f"Rank {rank} processing samples {start_idx}-{end_idx} ({count} samples)")
    
    model = VGLLM_Inference(pretrained_path, rank)
    results = []  # Store records associated with the original index.
    
    if rank == 0:
        pbar = tqdm(total=total_samples, desc="Overall Progress")
    
    for i, (sample, original_idx) in enumerate(zip(local_samples, local_indices)):
        # 1. Run inference and save the output.
        output = process_3d_video_caption(model, sample, data_path)
        # print(output)
        # 2. Compute the metric payload.
        result = scan2cap_process_results(sample, output)
        
        # 3. Record the original index for later sorting.
        results.append({
            "original_index": original_idx,  # Keep this sample's original position in data_samples.
            "result": result
        })
        
        if rank == 0:
            pbar.update(world_size)
    
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
        
        # Sort by original index to match data_samples exactly.
        final_results.sort(key=lambda x: x["original_index"])
        aggregated = scan2cap_aggregate_results(final_results)
        print("Aggregated result:", aggregated)  # Print the aggregated overall metric.
        
        # Save final results.
        output_file = os.path.join(output_path, "inference_results_with_output.json")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"Results saved to: {output_file} (order matches data_samples)")
    
    cleanup()

def scan2cap_process_results(doc, results):
    doc["pred_response"] = results if doc["iou"] >= 0.5 else ""
    doc["gt_response"] = doc["annotations"]
    return {"scan2cap_score": doc}

def scan2cap_aggregate_results(results):

    from lmms_eval.tasks.scan2cap.caption_eval.bleu.bleu import Bleu
    from lmms_eval.tasks.scan2cap.caption_eval.rouge.rouge import Rouge
    # from lmms_eval.tasks.scan2cap.caption_eval.meteor.meteor import Meteor
    from lmms_eval.tasks.scan2cap.caption_eval.cider.cider import Cider

    cider = Cider()
    bleu = Bleu()
    # meteor = Meteor()
    rouge = Rouge()

    res, gts = {}, {}
    for i, item in enumerate(results):
        # print(item)
        res[i] = ['sos ' + item["result"]["scan2cap_score"]['pred_response'].replace('.', ' . ').replace(',', ' , ').lower() + ' eos' ]
        gts[i] = ['sos ' + it.replace('.', ' . ').replace(',', ' , ').lower() + ' eos' for it in item["result"]["scan2cap_score"]['gt_response']]

    cider_score = cider.compute_score(gts, res)
    bleu_score = bleu.compute_score(gts, res)
    # meteor_score = meteor.compute_score(gts, res)
    rouge_score = rouge.compute_score(gts, res)

    table_data = [
        ["Metric", "Score"],
        ["CIDER", f"{cider_score[0]*100:.2f}"],
        ["BLEU-4", f"{bleu_score[0][-1]*100:.2f}"],
        # ["METEOR", f"{meteor_score[0]*100:.2f}"],
        ["ROUGE", f"{rouge_score[0]*100:.2f}"],
        ["Data Num", f"{len(res)}"]
    ]


    table = AsciiTable(table_data)
    table.title = "Evaluation Metrics"
    print(table.table)

    return cider_score[0]*100

def process_3d_video_caption(model, sample, data_path):
    text = sample["conversations"][0]["value"].replace("<image>", "")
    image_files = sample["images"]
    images = [
        Image.open(
            os.path.join(data_path, image_file)
        ).convert("RGB")
        for image_file in image_files
    ]
    
    output = model.call_model(
        contexts=[text],
        visuals=[images],
    )[0]

    # print("Output:", output)
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

    with open(os.path.join("data/evaluation/scan2cap", "scan2cap_val_32frames.json")) as f:
        data = json.load(f)
    
    pretrained_path = args.ckpt_path
    
    print(f"Node {args.node_rank}: Starting with {args.world_size} devices")
    os.makedirs(args.output_path, exist_ok=True)
    # data = data[:24]
    if args.world_size > 1:
        mp.spawn(
            run_inference,
            args=(args.world_size, data, pretrained_path, args.data_path, args.output_path),
            nprocs=args.world_size,
            join=True
        )
    else:
        run_inference(0, 1, data, pretrained_path, args.data_path, args.output_path)

if __name__ == "__main__":
    main()
