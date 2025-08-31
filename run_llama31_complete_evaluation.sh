#!/bin/bash

# Complete FollowBench Evaluation for Llama 3.1
# This script runs the full evaluation pipeline: inference -> GPT-4o-mini evaluation -> final results

set -e  # Exit on any error

echo "🦙 Complete FollowBench Evaluation for Llama 3.1"
echo "================================================"
echo "This will run:"
echo "1. Model inference on all constraint types"
echo "2. GPT-4o-mini evaluation"
echo "3. Final result generation with Level 0-5 analysis"
echo ""

# Configuration
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B"
API_KEY="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"

# Activate environment
echo "🔧 Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lm-eval-torch26

# Step 1: Model Inference
echo ""
echo "📊 Step 1: Running model inference on all constraint types..."
echo "============================================================"
echo "Constraint types: content, situation, style, format, example, mixed"
echo "This may take 30-60 minutes depending on GPU speed..."
echo ""

python code/model_inference_vllm.py \
    --model-path "$MODEL_PATH" \
    --constraint_types content situation style format example mixed \
    --gpus "0,1,2,3" \
    --num-gpus 4 \
    --temperature 0.7 \
    --max-tokens 1024

if [ $? -eq 0 ]; then
    echo "✅ Model inference completed successfully!"
else
    echo "❌ Model inference failed!"
    exit 1
fi

# Step 2: GPT-4o-mini Evaluation
echo ""
echo "🤖 Step 2: Running GPT-4o-mini evaluation..."
echo "============================================"
echo "This will evaluate constraint satisfaction using GPT-4o-mini..."
echo "Estimated time: 20-40 minutes depending on API speed..."
echo ""

python code/llm_eval.py \
    --model_path "$MODEL_PATH" \
    --constraint_types content situation style format mixed \
    --api_key "$API_KEY"

if [ $? -eq 0 ]; then
    echo "✅ GPT-4o-mini evaluation completed successfully!"
else
    echo "❌ GPT-4o-mini evaluation failed!"
    exit 1
fi

# Step 3: Final Result Generation
echo ""
echo "📈 Step 3: Generating final evaluation results..."
echo "================================================"
echo "This will generate Level 0-5 analysis and final metrics..."
echo ""

python code/eval.py \
    --model_paths "$MODEL_PATH" \
    --constraint_types content situation style format mixed

if [ $? -eq 0 ]; then
    echo "✅ Final evaluation completed successfully!"
else
    echo "❌ Final evaluation failed!"
    exit 1
fi

# Display results
echo ""
echo "🎉 Complete Evaluation Finished!"
echo "==============================="
echo ""
echo "📁 Results saved in:"
echo "   - Model outputs: api_output_vllm/$MODEL_PATH/"
echo "   - GPT-4o-mini evaluations: gpt4_discriminative_eval_output/$MODEL_PATH/"
echo "   - Final results: evaluation_result/$MODEL_PATH/"
echo ""
echo "📊 Key result files:"
echo "   - evaluation_result/$MODEL_PATH/level_results.json"
echo "   - evaluation_result/$MODEL_PATH/constraint_results.json"
echo ""

# Show a quick summary if results exist
RESULT_DIR="evaluation_result/$MODEL_PATH"
if [ -d "$RESULT_DIR" ]; then
    echo "📋 Quick Summary:"
    echo "=================="
    
    # Try to show some basic stats
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
echo "🚀 Evaluation pipeline completed successfully!"
echo "Ready for analysis and comparison with other models."
