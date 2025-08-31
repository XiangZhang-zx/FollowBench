#!/usr/bin/env python3

import argparse
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

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

        # Create output directory
        output_dir = os.path.join(args.api_output_path, args.model_path.replace('/', '_'))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{constraint_type}_constraint_gpu{accelerator.process_index}.jsonl")
        
        # Process each sample
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, item in enumerate(tqdm(data_shard, desc=f"Dream {constraint_type} GPU{accelerator.process_index}")):
                try:
                    instruction = item['prompt_new']
                    
                    # Format input for Dream
                    messages = [{'role': 'user', 'content': instruction}]
                    inputs = tokenizer.apply_chat_template(
                        messages, return_tensors='pt', return_dict=True, add_generation_prompt=True
                    )
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                    
                    # Generate response with Dream's diffusion process
                    if hasattr(model, 'generate'):
                        output = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=args.max_new_tokens,
                            diffusion_steps=args.diffusion_steps,
                            temperature=args.temperature,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        
                        generated_text = tokenizer.decode(
                            output[0][len(inputs['input_ids'][0]):], 
                            skip_special_tokens=True
                        )
                    else:
                        generated_text = "Dream generation method not available"
                    
                    # Format output to match FollowBench API format
                    api_output = {
                        "prompt_new": instruction,
                        "choices": [{"message": {"content": generated_text}}],
                        "generation": generated_text  # Add generation field for compatibility
                    }
                    
                    out_f.write(json.dumps(api_output) + '\n')
                    
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    # Write error output
                    error_output = {
                        "prompt_new": item.get('prompt_new', ''),
                        "choices": [{"message": {"content": f"Error: {str(e)}"}}]
                    }
                    out_f.write(json.dumps(error_output) + '\n')
                    continue
        
        print(f"✅ {constraint_type}: {len(data)} samples completed")
        print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dream model inference for FollowBench')
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to Dream model")
    parser.add_argument("--constraint_types", nargs='+', type=str, 
                       default=['content', 'situation', 'style', 'format', 'example', 'mixed'],
                       help="Constraint types to evaluate")
    parser.add_argument("--api_input_path", type=str, default="api_input", 
                       help="Path to API input files")
    parser.add_argument("--api_output_path", type=str, default="api_output", 
                       help="Path to save API output files")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                       help="Maximum new tokens to generate")
    parser.add_argument("--diffusion_steps", type=int, default=256, 
                       help="Number of diffusion steps for Dream")
    parser.add_argument("--temperature", type=float, default=0.2, 
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    dream_inference(args)
