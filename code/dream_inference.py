#!/usr/bin/env python3

import argparse
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

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
def dream_inference(args):
    """Custom inference for Dream diffusion model"""
    
    accelerator = Accelerator()
    
    print(f"🌟 Loading Dream model: {args.model_path}")
    
    # Load Dream model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Prepare model with accelerate
    model = accelerator.prepare(model)
    
    print(f"✅ Dream model loaded on {accelerator.device}")
    
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
        
        # Process each sample
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, item in enumerate(tqdm(data_shard, desc=f"Dream {constraint_type} GPU{accelerator.process_index}")):
                try:
                    instruction = item['prompt_new']
                    
                    # Format input for Dream Base Model - 使用简单格式
                    print(f"🔄 Dream Base Model: 使用简单格式（不使用chat template）")
                    # 直接使用原始instruction，不添加对话标记
                    inputs = tokenizer(instruction, return_tensors='pt')
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                    
                    # Generate response with Dream's diffusion process
                    # Check both the model and unwrapped model for the method
                    actual_model = getattr(model, 'module', model)  # Handle DDP wrapping

                    if hasattr(actual_model, 'diffusion_generate'):
                        with torch.no_grad():
                            generated_ids = actual_model.diffusion_generate(
                                inputs=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                steps=args.diffusion_steps,  # Use 'steps' instead of 'diffusion_steps'
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=0.95,
                                alg=args.alg,  # Add algorithm parameter
                                alg_temp=args.alg_temp  # Add algorithm temperature
                            )

                        # Extract only the newly generated tokens
                        new_tokens = generated_ids[0][len(inputs['input_ids'][0]):]
                        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                        # Clean up the output
                        if '<|end_of_text|>' in generated_text:
                            generated_text = generated_text.split('<|end_of_text|>')[0]
                        if '<|endoftext|>' in generated_text:
                            generated_text = generated_text.split('<|endoftext|>')[0]
                        generated_text = generated_text.strip()
                    else:
                        # Debug info
                        available_methods = [attr for attr in dir(actual_model) if 'generate' in attr.lower()]
                        generated_text = f"Dream diffusion_generate method not available. Available: {available_methods}"
                    
                    # Format output to match FollowBench API format
                    api_output = {
                        "prompt": instruction,  # 统一使用 "prompt" 字段名
                        "choices": [{"message": {"content": generated_text}}],
                        "generation": generated_text  # Add generation field for compatibility
                    }
                    
                    out_f.write(json.dumps(api_output) + '\n')
                    
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    # Write error output
                    error_output = {
                        "prompt": item.get('prompt_new', ''),  # 统一使用 "prompt" 字段名
                        "choices": [{"message": {"content": f"Error: {str(e)}"}}]
                    }
                    out_f.write(json.dumps(error_output) + '\n')
                    continue
        
        print(f"✅ {constraint_type}: {len(data)} samples completed")
        print(f"Output saved to: {output_file}")

        # 等待所有进程完成
        accelerator.wait_for_everyone()

        # 合并所有GPU的结果 (只在主进程中执行)
        if accelerator.is_main_process:
            merge_gpu_results(args, constraint_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dream model inference for FollowBench')
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to Dream model")
    parser.add_argument("--constraint_types", nargs='+', type=str, 
                       default=['content', 'situation', 'style', 'format', 'example', 'mixed'],
                       help="Constraint types to evaluate")
    parser.add_argument("--api_input_path", type=str, default="api_input", 
                       help="Path to API input files")
    parser.add_argument("--api_output_path", type=str, default="api_output_vllm",
                       help="Path to save API output files")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="Maximum new tokens to generate")
    parser.add_argument("--diffusion_steps", type=int, default=2048,
                       help="Number of diffusion steps for Dream")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for generation")
    parser.add_argument("--alg", type=str, default="entropy",
                       help="Remasking strategy: origin, maskgit_plus, topk_margin, entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0,
                       help="Algorithm temperature for confidence-based strategies")
    parser.add_argument("--limit_samples", type=int, default=0,
                       help="限制处理的样本数量，0表示处理全部")

    args = parser.parse_args()

    # Print parameters for debugging
    print(f"🔧 Dream Inference Parameters:")
    print(f"  - Model path: {args.model_path}")
    print(f"  - Diffusion steps: {args.diffusion_steps}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Constraint types: {args.constraint_types}")

    dream_inference(args)
