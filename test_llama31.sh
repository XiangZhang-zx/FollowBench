#!/bin/bash

# FollowBench Test for Llama 3.1 - Single Model Evaluation
# This script demonstrates the constraint evaluation flow from single to multi

set -e  # Exit on any error

# Configuration
export CUDA_VISIBLE_DEVICES="0,1"  # Use 2 GPUs
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

# Model configuration
MODEL_NAME="llama31"
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B"

echo "🦙 FollowBench Evaluation for Llama 3.1"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Constraint evaluation flow: Single → Multi"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lm-eval-torch26

# Create output directories
mkdir -p "api_output_vllm/$MODEL_NAME"
mkdir -p "results/$MODEL_NAME"

echo "📊 Understanding FollowBench Constraint Types:"
echo "=============================================="

# Analyze constraint types and their progression
python -c "
import json
import os

constraint_types = [
    'content_constraints.json',
    'situation_constraints.json', 
    'style_constraints.json',
    'format_constraints.json',
    'mixed_constraints.json'
]

print('📋 Constraint Type Analysis:')
print('----------------------------')

total_single = 0
for i, constraint_file in enumerate(constraint_types[:-1], 1):  # Exclude mixed
    if os.path.exists(f'data/{constraint_file}'):
        with open(f'data/{constraint_file}', 'r') as f:
            data = json.load(f)
        constraint_name = constraint_file.replace('_constraints.json', '').upper()
        print(f'{i}. {constraint_name}: {len(data)} examples (Single Constraint)')
        total_single += len(data)

# Mixed constraints (multi-constraint)
if os.path.exists('data/mixed_constraints.json'):
    with open('data/mixed_constraints.json', 'r') as f:
        mixed_data = json.load(f)
    print(f'5. MIXED: {len(mixed_data)} examples (Multi-Constraint)')

print(f'\\nTotal Single Constraints: {total_single}')
print(f'Total Multi Constraints: {len(mixed_data)}')
print(f'Total Examples: {total_single + len(mixed_data)}')

print('\\n🔄 Evaluation Flow:')
print('Single Constraint Types → Mixed Constraint Types')
print('(Content, Situation, Style, Format) → (Combined Constraints)')
"

echo ""
echo "🚀 Starting Llama 3.1 Inference..."
echo "=================================="

# Test with a smaller subset first
echo "📝 Running inference on FORMAT constraints (as example)..."

python code/model_inference_vllm.py \
    --model-path "$MODEL_PATH" \
    --gpus "0,1" \
    --num-gpus 2 \
    --constraint_types format \
    --data_path "data" \
    --api_output_path "api_output_vllm/$MODEL_NAME" \
    --temperature 0.7 \
    --max-tokens 512 \
    --debug

echo "✅ Format constraint inference completed!"

# Check the output
echo ""
echo "📊 Analyzing Results..."
echo "======================"

python -c "
import json
import os

output_dir = 'api_output_vllm/$MODEL_NAME'
if os.path.exists(f'{output_dir}/format_constraints.json'):
    with open(f'{output_dir}/format_constraints.json', 'r') as f:
        results = json.load(f)
    
    print(f'✅ Generated {len(results)} responses for FORMAT constraints')
    
    # Show a sample result
    if results:
        sample = results[0]
        print('\\n📝 Sample Result Structure:')
        print('Keys:', list(sample.keys()))
        
        print('\\n📋 Sample Input:')
        print('Instruction:', sample.get('instruction', 'N/A')[:200] + '...')
        
        print('\\n📤 Sample Output:')
        print('Response:', sample.get('response', 'N/A')[:200] + '...')
else:
    print('❌ No output file found')
"

echo ""
echo "🎯 Next Steps for Full Evaluation:"
echo "=================================="
echo "1. Single Constraint Evaluation:"
echo "   - Content constraints (semantic requirements)"
echo "   - Situation constraints (context-specific requirements)" 
echo "   - Style constraints (writing style requirements)"
echo "   - Format constraints (structural requirements)"
echo ""
echo "2. Multi-Constraint Evaluation:"
echo "   - Mixed constraints (combining multiple constraint types)"
echo "   - Tests model's ability to satisfy multiple requirements simultaneously"
echo ""
echo "3. Evaluation Metrics:"
echo "   - Constraint satisfaction rate"
echo "   - Response quality"
echo "   - Consistency across constraint types"

echo ""
echo "✅ Llama 3.1 test completed!"
echo "📁 Results saved in: api_output_vllm/$MODEL_NAME/"
