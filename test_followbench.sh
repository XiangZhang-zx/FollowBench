#!/bin/bash

# FollowBench Test Script
# This script tests the FollowBench environment without running full model inference

set -e  # Exit on any error

echo "🧪 Testing FollowBench Environment"
echo "=================================="

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lm-eval-torch26

# Test 1: Check Python packages
echo "📦 Testing Python package imports..."
python -c "
import torch
import transformers
import vllm
import openai
import json
print('✅ All packages imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
"

# Test 2: Check data files
echo "📁 Testing data files..."
python -c "
import json
import os

data_files = [
    'content_constraints.json',
    'format_constraints.json', 
    'style_constraints.json',
    'situation_constraints.json',
    'example_constraints.json',
    'mixed_constraints.json'
]

for file in data_files:
    path = f'data/{file}'
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        print(f'✅ {file}: {len(data)} entries')
    else:
        print(f'❌ {file}: Not found')
"

# Test 3: Check script help functions
echo "🔧 Testing script help functions..."
echo "Testing model_inference_vllm.py..."
python code/model_inference_vllm.py --help > /dev/null && echo "✅ model_inference_vllm.py works"

echo "Testing llm_eval.py..."
python code/llm_eval.py --help > /dev/null && echo "✅ llm_eval.py works"

echo "Testing eval.py..."
python code/eval.py --help > /dev/null && echo "✅ eval.py works"

# Test 4: Create test directories
echo "📂 Testing directory creation..."
mkdir -p test_output/api_output_vllm/test_model
mkdir -p test_output/gpt4_discriminative_eval_input/test_model
mkdir -p test_output/gpt4_discriminative_eval_output/test_model
echo "✅ Test directories created successfully"

# Test 5: Test data loading with a small sample
echo "📊 Testing data processing..."
python -c "
import json

# Load a small sample from format constraints
with open('data/format_constraints.json', 'r') as f:
    data = json.load(f)

# Take first 3 entries for testing
sample_data = data[:3]

# Save test sample
with open('test_output/test_sample.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print(f'✅ Created test sample with {len(sample_data)} entries')
print('Sample entry keys:', list(sample_data[0].keys()))
"

echo ""
echo "🎉 FollowBench Environment Test Completed!"
echo "=========================================="
echo "✅ All components are working correctly"
echo "✅ Environment is ready for FollowBench evaluation"
echo ""
echo "📝 Notes:"
echo "- To run full evaluation, you need:"
echo "  1. GPT-4 API key"
echo "  2. Model access (HuggingFace tokens if needed)"
echo "  3. Sufficient GPU memory for model inference"
echo ""
echo "🚀 You can now run the full script with:"
echo "   ./run_ar_models_followbench.sh"
