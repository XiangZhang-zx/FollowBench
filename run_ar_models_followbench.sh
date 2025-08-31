#!/bin/bash

# FollowBench evaluation for 4 AR models
# This script runs the complete evaluation pipeline:
# 1. Model inference using VLLM
# 2. LLM-based evaluation using GPT-4
# 3. Final result aggregation

set -e  # Exit on any error

# Configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Adjust based on available GPUs
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

# Your GPT-4 API key (replace with your actual key)
GPT4_API_KEY="your_gpt4_api_key_here"

# Model configurations
declare -A MODELS=(
    ["deepseek"]="deepseek-ai/deepseek-llm-7b-base"
    ["llama"]="meta-llama/Meta-Llama-3.1-8B"
    ["mistral"]="mistralai/Mistral-7B-Instruct-v0.3"
    ["trillion"]="trillionlabs/Trillion-7B-preview"
)

# VLLM parameters
NUM_GPUS=4
TENSOR_PARALLEL_SIZE=2  # Adjust based on model size and GPU memory

# Create results directory
mkdir -p followbench_results
cd FollowBench/

echo "🚀 Starting FollowBench evaluation for AR models"
echo "================================================"

# Function to run model inference
run_model_inference() {
    local model_name=$1
    local model_path=$2
    
    echo "📊 Running inference for $model_name ($model_path)"
    echo "---------------------------------------------------"
    
    # Create model-specific output directory
    mkdir -p "api_output_vllm/$model_name"
    
    # Run VLLM inference
    python code/model_inference_vllm.py \
        --model_path "$model_path" \
        --gpus "0,1,2,3" \
        --num_gpus $TENSOR_PARALLEL_SIZE \
        --constraint_types content situation style format mixed \
        --api_input_path "data" \
        --api_output_path "api_output_vllm/$model_name" \
        --temperature 0.7 \
        --max_new_tokens 1024 \
        --seed 42
    
    echo "✅ Inference completed for $model_name"
}

# Function to run LLM evaluation
run_llm_evaluation() {
    local model_name=$1
    local model_path=$2
    
    echo "🤖 Running LLM evaluation for $model_name"
    echo "-------------------------------------------"
    
    # Create evaluation directories
    mkdir -p "gpt4_discriminative_eval_input/$model_name"
    mkdir -p "gpt4_discriminative_eval_output/$model_name"
    
    # Run GPT-4 based evaluation
    python code/llm_eval.py \
        --model_path "$model_name" \
        --api_key "$GPT4_API_KEY" \
        --constraint_types content situation style format mixed \
        --data_path "data" \
        --api_output_path "api_output_vllm" \
        --gpt4_discriminative_eval_input_path "gpt4_discriminative_eval_input" \
        --gpt4_discriminative_eval_output_path "gpt4_discriminative_eval_output" \
        --max_tokens 1024
    
    echo "✅ LLM evaluation completed for $model_name"
}

# Function to run final evaluation
run_final_evaluation() {
    echo "📈 Running final evaluation and result aggregation"
    echo "---------------------------------------------------"
    
    # Collect all model names
    model_list=""
    for model_name in "${!MODELS[@]}"; do
        model_list="$model_list $model_name"
    done
    
    # Run final evaluation
    python code/eval.py --model_paths $model_list
    
    echo "✅ Final evaluation completed"
    echo "📁 Results saved in evaluation_result/"
}

# Main execution
main() {
    echo "🔧 Installing dependencies..."
    pip install vllm openai tqdm
    
    # Run inference for all models
    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"
        run_model_inference "$model_name" "$model_path"
    done
    
    # Run LLM evaluation for all models
    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"
        run_llm_evaluation "$model_name" "$model_path"
    done
    
    # Run final evaluation
    run_final_evaluation
    
    echo "🎉 All evaluations completed successfully!"
    echo "📊 Check evaluation_result/ for final results"
}

# Execute main function
main "$@"
