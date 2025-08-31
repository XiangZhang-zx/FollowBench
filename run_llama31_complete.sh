#!/bin/bash

# Complete FollowBench evaluation for Llama 3.1
# Based on the original run_ar_models_followbench.sh script

set -e  # Exit on any error

# Configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

# Model configuration
MODEL_NAME="llama31"
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B"

# VLLM parameters
NUM_GPUS=4
TENSOR_PARALLEL_SIZE=2

echo "🦙 Complete FollowBench Evaluation for Llama 3.1"
echo "================================================"
echo "Model: $MODEL_PATH"
echo "GPUs: $NUM_GPUS (Tensor Parallel: $TENSOR_PARALLEL_SIZE)"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lm-eval-torch26

# Function to run model inference
run_model_inference() {
    local model_name=$1
    local model_path=$2
    
    echo "📊 Running inference for $model_name ($model_path)"
    echo "---------------------------------------------------"
    
    # Create model-specific output directory
    mkdir -p "api_output_vllm/$model_name"
    
    # Run VLLM inference with all constraint types
    python code/model_inference_vllm.py \
        --model-path "$model_path" \
        --gpus "0,1,2,3" \
        --num-gpus $TENSOR_PARALLEL_SIZE \
        --constraint_types content situation style format mixed \
        --data_path "data" \
        --api_output_path "api_output_vllm/$model_name" \
        --temperature 0.7 \
        --max-tokens 1024
    
    echo "✅ Inference completed for $model_name"
}

# Function to analyze results
analyze_results() {
    local model_name=$1
    
    echo "📊 Analyzing results for $model_name"
    echo "-----------------------------------"
    
    python3 -c "
import json
import os
from collections import defaultdict

print('🔍 Llama 3.1 FollowBench Results Analysis')
print('=' * 50)

model_name = '$model_name'
base_path = f'api_output_vllm/{model_name}'

# Find the actual model path directory
model_dirs = []
if os.path.exists(base_path):
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            model_dirs.append(item_path)

if not model_dirs:
    print(f'❌ No model directories found in {base_path}')
    exit(1)

# Use the first (and likely only) model directory
model_dir = model_dirs[0]
print(f'📁 Using model directory: {model_dir}')

constraint_types = ['content', 'situation', 'style', 'format', 'mixed']
results_summary = {}

for constraint_type in constraint_types:
    # Try different file extensions
    possible_files = [
        f'{model_dir}/{constraint_type}_constraint.jsonl',
        f'{model_dir}/{constraint_type}_constraints.jsonl',
        f'{model_dir}/{constraint_type}.jsonl'
    ]
    
    file_path = None
    for pf in possible_files:
        if os.path.exists(pf):
            file_path = pf
            break
    
    if file_path:
        print(f'\\n📋 {constraint_type.upper()} CONSTRAINTS:')
        print(f'   File: {file_path}')
        
        try:
            # Load JSONL file
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            total_examples = len(data)
            successful_responses = 0
            response_lengths = []
            level_distribution = defaultdict(int)
            
            for item in data:
                # Extract response content
                response = ''
                if 'choices' in item and item['choices']:
                    try:
                        response = item['choices'][0]['message']['content']
                    except (KeyError, IndexError):
                        pass
                elif 'response' in item:
                    response = item['response']
                elif 'content' in item:
                    response = item['content']
                
                # Check if response is valid
                if response and response.strip():
                    successful_responses += 1
                    response_lengths.append(len(response))
                
                # Extract level information from original data
                level = item.get('level', 'unknown')
                level_distribution[level] += 1
            
            success_rate = (successful_responses / total_examples * 100) if total_examples > 0 else 0
            avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
            
            print(f'   Total examples: {total_examples}')
            print(f'   Successful responses: {successful_responses} ({success_rate:.1f}%)')
            print(f'   Average response length: {avg_length:.0f} characters')
            print(f'   Level distribution: {dict(level_distribution)}')
            
            results_summary[constraint_type] = {
                'total': total_examples,
                'successful': successful_responses,
                'success_rate': success_rate,
                'avg_length': avg_length,
                'level_dist': dict(level_distribution)
            }
            
        except Exception as e:
            print(f'   ❌ Error processing file: {e}')
            results_summary[constraint_type] = {'total': 0, 'successful': 0, 'success_rate': 0}
    else:
        print(f'\\n❌ {constraint_type.upper()}: No output file found')
        results_summary[constraint_type] = {'total': 0, 'successful': 0, 'success_rate': 0}

# Summary analysis
print('\\n📊 PERFORMANCE SUMMARY:')
print('=' * 50)
print(f'{'Constraint Type':<15} {'Total':<8} {'Success':<8} {'Rate':<8} {'Avg Length':<12}')
print('-' * 55)

for constraint_type, stats in results_summary.items():
    print(f'{constraint_type:<15} {stats[\"total\"]:<8} {stats[\"successful\"]:<8} {stats[\"success_rate\"]:<7.1f}% {stats[\"avg_length\"]:<11.0f}')

# Hypothesis testing
print('\\n🎯 CONSTRAINT COMPLEXITY ANALYSIS:')
print('=' * 50)

single_constraints = ['content', 'situation', 'style', 'format']
single_rates = [results_summary[ct]['success_rate'] for ct in single_constraints if ct in results_summary and results_summary[ct]['total'] > 0]
single_avg = sum(single_rates) / len(single_rates) if single_rates else 0

mixed_success = results_summary.get('mixed', {}).get('success_rate', 0)

print(f'Single Constraints Average Success Rate: {single_avg:.1f}%')
print(f'Mixed Constraints Success Rate: {mixed_success:.1f}%')
print(f'Performance Drop: {single_avg - mixed_success:.1f} percentage points')

if single_avg > mixed_success:
    print('\\n✅ HYPOTHESIS SUPPORTED: Llama 3.1 shows performance drop on mixed constraints')
    print('   AR model struggles more with complex constraint combinations')
else:
    print('\\n❌ HYPOTHESIS NOT SUPPORTED: Mixed constraints performance is not significantly lower')

print('\\n✅ Llama 3.1 analysis completed!')
"
}

echo ""
echo "🚀 Starting Llama 3.1 evaluation..."
echo "==================================="

# Run the inference
run_model_inference "$MODEL_NAME" "$MODEL_PATH"

echo ""
echo "📊 Analyzing results..."
echo "======================"

# Analyze the results
analyze_results "$MODEL_NAME"

echo ""
echo "🎉 Complete evaluation finished!"
echo "==============================="
echo "📁 Results saved in: api_output_vllm/$MODEL_NAME/"
echo ""
echo "🔍 Next steps:"
echo "1. Compare with Dream model results"
echo "2. Analyze level-specific performance (0-5)"
echo "3. Examine specific constraint satisfaction patterns"
