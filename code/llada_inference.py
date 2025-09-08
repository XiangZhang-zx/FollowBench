#!/usr/bin/env python3

import argparse
import os
import json
import torch
from datetime import timedelta
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm, trange
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

# Import Fast-dLLM cache components and modified model
try:
    import sys
    sys.path.append('/home/xz2649/projects/Fast-dLLM/llada')
    from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
    from model.modeling_llada import LLaDAModelLM  # Use Fast-dLLM's modified LLaDA model
    FAST_DLLM_AVAILABLE = True
    USE_FAST_DLLM_MODEL = True
    print("✅ Fast-dLLM LLaDA cache and model imported successfully")
except ImportError as e:
    print(f"⚠️ Fast-dLLM LLaDA cache not available: {e}")
    FAST_DLLM_AVAILABLE = False
    USE_FAST_DLLM_MODEL = False

    # Fallback to original generate function
    try:
        sys.path.append('/home/xz2649/projects/dLLM-cache')
        from utils import generate
        print("✅ Fallback to dLLM-cache generate function")
    except ImportError:
        print("❌ No generate function available")

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    '''
    if temperature == 0.:
        return logits  # Skip noise when temperature is 0
    
    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    
    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()
    
    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
        
    return num_transfer_tokens.to(torch.int64)

@torch.no_grad()
def generate(model, prompt, steps=64, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Optimized version of the generate function.
    '''
    # Use mixed precision for faster computation
    with torch.cuda.amp.autocast(enabled=True):
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        # Adjust steps if needed
        steps_per_block = max(1, steps // num_blocks)

        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length
            
            block_mask_index = (x[:, start_idx:end_idx] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                
                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    
                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == 'low_confidence':
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == 'random':
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]

        return x

class Generator():
    def __init__(self, model, tokenizer, use_cache=True, dual_cache=False, threshold=None, factor=None, accelerator=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        self.dual_cache = dual_cache
        self.threshold = threshold
        self.factor = factor
        self.accelerator = accelerator
        self.kwargs = kwargs
    
    def generate(self, inputs):
        batch_size = 1
        responses = []
        for i in trange(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            model_inputs = self.tokenizer(
                batch,
                padding=True,
                padding_side="left",
                truncation=False,
                return_tensors="pt"
            ).to(self.model.device)

            # Use Fast-dLLM cache functions if available
            if FAST_DLLM_AVAILABLE and self.use_cache:
                if self.dual_cache:
                    try:
                        generated_ids, nfe = generate_with_dual_cache(
                            self.model,
                            model_inputs.input_ids,
                            threshold=self.threshold,
                            factor=self.factor,
                            **self.kwargs
                        )
                    except TypeError as e:
                        # Fallback: filter out unsupported parameters
                        print(f"⚠️ Dual cache failed with full kwargs, trying without cfg_scale: {e}")
                        cache_kwargs = {k: v for k, v in self.kwargs.items() if k != 'cfg_scale'}
                        generated_ids, nfe = generate_with_dual_cache(
                            self.model,
                            model_inputs.input_ids,
                            threshold=self.threshold,
                            factor=self.factor,
                            **cache_kwargs
                        )
                else:
                    try:
                        generated_ids, nfe = generate_with_prefix_cache(
                            self.model,
                            model_inputs.input_ids,
                            threshold=self.threshold,
                            factor=self.factor,
                            **self.kwargs
                        )
                    except TypeError as e:
                        # Fallback: filter out unsupported parameters
                        print(f"⚠️ Prefix cache failed with full kwargs, trying without cfg_scale: {e}")
                        cache_kwargs = {k: v for k, v in self.kwargs.items() if k != 'cfg_scale'}
                        generated_ids, nfe = generate_with_prefix_cache(
                            self.model,
                            model_inputs.input_ids,
                            threshold=self.threshold,
                            factor=self.factor,
                            **cache_kwargs
                        )
            else:
                # Fallback to regular generate
                generated_ids = generate(self.model, model_inputs.input_ids, **self.kwargs)

            # 在生成后同步所有进程，避免NCCL超时
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            responses.extend(response)
        return responses, None, None

def merge_gpu_results(args, constraint_type):
    """合并多个GPU的结果文件"""
    import glob
    import time

    # 等待更长时间确保所有GPU完成写入
    print("⏳ Waiting for all GPU processes to complete...")
    time.sleep(5)

    # 查找所有GPU的输出文件
    pattern = os.path.join(args.api_output_path,
                          args.model_path.replace('/', '_'),
                          f"{constraint_type}_constraint_gpu*.jsonl")

    # 多次尝试查找文件，确保所有GPU都完成
    max_attempts = 10
    for attempt in range(max_attempts):
        gpu_files = glob.glob(pattern)
        if gpu_files:
            break
        print(f"🔍 Attempt {attempt + 1}: Looking for GPU result files...")
        time.sleep(1)

    if not gpu_files:
        print(f"⚠️ No GPU result files found for {constraint_type}")
        return

    print(f"📁 Found {len(gpu_files)} GPU result files: {[os.path.basename(f) for f in gpu_files]}")

    # 合并文件
    merged_file = os.path.join(args.api_output_path,
                              args.model_path.replace('/', '_'),
                              f"{constraint_type}_constraint.jsonl")

    all_results = []
    for gpu_file in sorted(gpu_files):
        print(f"📖 Reading {os.path.basename(gpu_file)}...")
        with open(gpu_file, 'r', encoding='utf-8') as f:
            file_results = []
            for line in f:
                if line.strip():
                    file_results.append(json.loads(line))
            all_results.extend(file_results)
            print(f"   Found {len(file_results)} results")

    # 写入合并后的文件
    with open(merged_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"🔗 Merged {len(all_results)} results from {len(gpu_files)} GPU files")
    print(f"📁 Merged file: {merged_file}")

@torch.inference_mode()
def llada_inference(args):
    """LLaDA diffusion inference for FollowBench"""

    # 设置分布式进程组超时时间为2小时，避免NCCL超时
    ipg = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[ipg])

    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"🚀 Loading LLaDA model: {args.model_path}")

    # Load LLaDA model - use Fast-dLLM's modified version if available
    if USE_FAST_DLLM_MODEL:
        print("🚀 Using Fast-dLLM's modified LLaDA model with KV cache support")
        model = LLaDAModelLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        print("⚠️ Using original LLaDA model (KV cache may not work)")
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Fast-dLLM cache is automatically managed in the generate functions
    if FAST_DLLM_AVAILABLE and args.use_cache:
        print("🚀 Fast-dLLM cache enabled for LLaDA model")
        if args.dual_cache:
            print("   Using dual cache mode")
        else:
            print("   Using prefix cache mode")
    else:
        if not FAST_DLLM_AVAILABLE:
            print("⚠️ Fast-dLLM cache not available for LLaDA, using fallback")
        else:
            print("⚠️ Fast-dLLM cache disabled by --use_cache=False")

    # 手动移动模型到设备，避免accelerate.prepare()干扰Fast-dLLM缓存
    if USE_FAST_DLLM_MODEL and FAST_DLLM_AVAILABLE:
        print("🚀 Using manual device placement for Fast-dLLM compatibility")
        model = model.to(accelerator.device)
    else:
        print("⚠️ Using accelerate.prepare() for fallback model")
        model = accelerator.prepare(model)

    # Create generator with LLaDA-specific parameters
    gen_params = {
        'steps': args.diffusion_steps,
        'gen_length': args.max_new_tokens,
        'block_length': args.block_length,
        'temperature': args.temperature,
        'cfg_scale': args.cfg_scale,
        'remasking': 'low_confidence',
        'mask_id': 126336  # LLaDA mask token ID
    }
    
    generator = Generator(model, tokenizer, use_cache=(FAST_DLLM_AVAILABLE and args.use_cache), dual_cache=args.dual_cache, threshold=args.threshold, factor=args.factor, accelerator=accelerator, **gen_params)
    
    print(f"✅ LLaDA model loaded with parameters: {gen_params}")

    # 等待所有GPU完成模型加载
    accelerator.wait_for_everyone()

    # Process each constraint type
    for constraint_type in args.constraint_types:
        print(f"\n🔍 Processing {constraint_type} constraints...")
        
        # Load input data
        input_file = os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl")
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # 限制样本数量
        if args.limit_samples > 0:
            data = data[:args.limit_samples]
            print(f"限制处理样本数量为: {len(data)}")

        print(f"Processing {len(data)} samples...")

        # Shard data across GPUs for parallel processing
        total_samples = len(data)
        samples_per_gpu = total_samples // accelerator.num_processes
        start_idx = accelerator.process_index * samples_per_gpu
        if accelerator.process_index == accelerator.num_processes - 1:
            # Last GPU handles remaining samples
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_gpu

        # Get data shard for this GPU
        data_shard = data[start_idx:end_idx]
        print(f"GPU {accelerator.process_index}: Processing samples {start_idx}-{end_idx-1} ({len(data_shard)} samples)")

        # Create output directory (use slash separator like AR models)
        output_dir = os.path.join(args.api_output_path, args.model_path)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{constraint_type}_constraint_gpu{accelerator.process_index}.jsonl")

        # Prepare inputs for batch processing
        inputs = [item['prompt_new'] for item in data_shard]
        
        # Generate responses
        responses, _, _ = generator.generate(inputs)
        
        # Save outputs in FollowBench format (compatible with llm_eval.py)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for item, response in zip(data_shard, responses):
                # Clean up response - remove any artifacts or repetitions
                cleaned_response = response.strip()

                api_output = {
                    "prompt": item['prompt_new'],  # Use "prompt" key for compatibility
                    "choices": [{"message": {"content": cleaned_response}}]
                }
                out_f.write(json.dumps(api_output, ensure_ascii=False) + '\n')
        
        print(f"✅ {constraint_type}: {len(data)} samples completed")
        print(f"Output saved to: {output_file}")

        # 等待所有进程完成
        accelerator.wait_for_everyone()

        # 合并所有GPU的结果 (只在主进程中执行)
        if accelerator.is_main_process:
            merge_gpu_results(args, constraint_type)

        # Clear GPU cache after each constraint type
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLaDA diffusion model inference for FollowBench')
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to LLaDA model")
    parser.add_argument("--constraint_types", nargs='+', type=str, 
                       default=['content', 'situation', 'style', 'format', 'example', 'mixed'],
                       help="Constraint types to evaluate")
    parser.add_argument("--api_input_path", type=str, default="api_input", 
                       help="Path to API input files")
    parser.add_argument("--api_output_path", type=str, default="api_output_vllm",
                       help="Path to save API output files")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="Maximum new tokens to generate")
    parser.add_argument("--diffusion_steps", type=int, default=1024,
                       help="Number of diffusion steps for LLaDA")
    parser.add_argument("--block_length", type=int, default=1024,
                       help="Block length for LLaDA generation")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for generation")
    parser.add_argument("--limit_samples", type=int, default=0,
                       help="限制处理的样本数量，0表示处理全部")

    # Fast-dLLM Cache parameters
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="Enable Fast-dLLM cache for acceleration")
    parser.add_argument("--dual_cache", action="store_true", default=False,
                       help="Enable dual cache mode for Fast-dLLM")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold for cache optimization")
    parser.add_argument("--factor", type=float, default=None,
                       help="Factor for cache optimization")
    parser.add_argument("--cfg_scale", type=float, default=0.0,
                       help="Classifier-free guidance scale")

    args = parser.parse_args()
    
    llada_inference(args)
