#!/bin/bash

# Complete FollowBench Evaluation for Dream Model
# This script runs evaluation on existing Dream outputs

set -e  # Exit on any error

echo "🌟 Complete FollowBench Evaluation for Dream Model"
echo "=================================================="
echo "This will run GPT-4o-mini evaluation on existing Dream outputs"
echo ""

# Configuration
MODEL_PATH="Dream-org/Dream-v0-Base-7B"
API_KEY="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"

# Activate environment
echo "🔧 Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lm-eval-torch26

# Check existing Dream outputs
echo ""
echo "📋 Checking existing Dream outputs..."
echo "===================================="

DREAM_OUTPUT_DIR="api_output_vllm/$MODEL_PATH"
if [ ! -d "$DREAM_OUTPUT_DIR" ]; then
    echo "❌ Dream output directory not found: $DREAM_OUTPUT_DIR"
    echo "Please run Dream model inference first!"
    exit 1
fi

# List available constraint files
echo "Available Dream output files:"
ls -la "$DREAM_OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "No .jsonl files found"

# Step 1: GPT-4o-mini Evaluation (using subset data)
echo ""
echo "🤖 Step 1: Running GPT-4o-mini evaluation on Dream outputs..."
echo "============================================================"
echo "Using data_subset to match available Dream outputs..."
echo ""

python code/llm_eval.py \
    --model_path "$MODEL_PATH" \
    --constraint_types content mixed \
    --data_path data_subset \
    --api_key "$API_KEY"

if [ $? -eq 0 ]; then
    echo "✅ GPT-4o-mini evaluation completed successfully!"
else
    echo "❌ GPT-4o-mini evaluation failed!"
    exit 1
fi

# Step 2: Final Result Generation
echo ""
echo "📈 Step 2: Generating final evaluation results..."
echo "================================================"
echo ""

python code/eval.py \
    --model_paths "$MODEL_PATH" \
    --constraint_types content mixed \
    --data_path data_subset

if [ $? -eq 0 ]; then
    echo "✅ Final evaluation completed successfully!"
else
    echo "❌ Final evaluation failed!"
    exit 1
fi

# Display results
echo ""
echo "🎉 Dream Evaluation Finished!"
echo "============================="
echo ""
echo "📁 Results saved in:"
echo "   - GPT-4o-mini evaluations: gpt4_discriminative_eval_output/$MODEL_PATH/"
echo "   - Final results: evaluation_result/$MODEL_PATH/"
echo ""

# Show a quick summary
RESULT_DIR="evaluation_result/$MODEL_PATH"
if [ -d "$RESULT_DIR" ]; then
    echo "📋 Quick Summary:"
    echo "=================="
    
    if [ -f "$RESULT_DIR/level_results.json" ]; then
        echo "✅ Level-wise results generated"
    fi
    
    if [ -f "$RESULT_DIR/constraint_results.json" ]; then
        echo "✅ Constraint-wise results generated"
    fi
    
    echo ""
    echo "🔍 To view detailed results:"
    echo "   cat evaluation_result/$MODEL_PATH/level_results.json"
    echo "   cat evaluation_result/$MODEL_PATH/constraint_results.json"
fi

echo ""
echo "🚀 Dream evaluation completed!"
echo "Ready for comparison with Llama 3.1 results."
