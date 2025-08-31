#!/usr/bin/env python3

import json
import os

def fix_dream_output_format():
    """Fix Dream output format to ensure proper matching with data"""
    
    print("🔧 Fixing Dream output format for evaluation...")
    
    constraint_types = ["content", "mixed"]
    
    for constraint_type in constraint_types:
        print(f"\n📋 Processing {constraint_type} constraints...")
        
        # Load original data
        data_file = f"data/{constraint_type}_constraints.json"
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Load Dream output
        output_file = f"api_output_vllm/Dream-org/Dream-v0-Base-7B/{constraint_type}_constraint.jsonl"
        
        if not os.path.exists(output_file):
            print(f"❌ Output file not found: {output_file}")
            continue
            
        with open(output_file, 'r') as f:
            output = [json.loads(line) for line in f if line.strip()]
        
        print(f"Data entries: {len(data)}")
        print(f"Output entries: {len(output)}")
        
        # Create a mapping from prompt to response
        prompt_to_response = {}
        for item in output:
            prompt = item.get('prompt', '')
            response = item.get('choices', [{}])[0].get('message', {}).get('content', '')
            prompt_to_response[prompt] = response
        
        # Match data with output and add generation field
        matched = 0
        for i, data_item in enumerate(data):
            instruction = data_item['instruction']
            
            if instruction in prompt_to_response:
                data_item['generation'] = prompt_to_response[instruction]
                matched += 1
            else:
                # If no exact match, add empty generation
                data_item['generation'] = ''
                print(f"Warning: No match found for data item {i}")
        
        print(f"✅ Matched {matched}/{len(data)} entries")
        
        # Save the fixed data with generation field
        fixed_output_file = f"api_output_vllm/Dream-org/Dream-v0-Base-7B/{constraint_type}_constraint_fixed.jsonl"
        
        with open(fixed_output_file, 'w') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')
        
        print(f"📁 Fixed output saved to: {fixed_output_file}")
        
        # Also create a backup of the original data with generation field
        data_with_gen_file = f"data_gpt4_discriminative_eval_input/Dream-org/Dream-v0-Base-7B/{constraint_type}_constraint.jsonl"
        os.makedirs(os.path.dirname(data_with_gen_file), exist_ok=True)
        
        with open(data_with_gen_file, 'w') as f:
            for item in data:
                if 'generation' in item:
                    f.write(json.dumps(item) + '\n')
        
        print(f"📁 Data with generation saved to: {data_with_gen_file}")
    
    print("\n🎉 Dream format fixing completed!")

if __name__ == "__main__":
    fix_dream_output_format()
