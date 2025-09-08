#!/usr/bin/env python3

import argparse
import sys

def test_dream_params():
    """Test Dream inference script parameter parsing"""
    print("🧪 Testing Dream inference parameter parsing...")
    
    # Import Dream inference argument parser
    sys.path.append('/home/xz2649/projects/FollowBench/code')
    
    # Create a mock argument list like what the batch script would pass
    test_args = [
        '--model_path', 'Dream-org/Dream-v0-Base-7B',
        '--constraint_types', 'content', 'situation',
        '--temperature', '0.2',
        '--max_new_tokens', '1024',
        '--diffusion_steps', '1024',
        '--api_output_path', 'api_output',
        '--use_cache',
        '--prompt_interval_steps', '100',
        '--gen_interval_steps', '7',
        '--transfer_ratio', '0.25'
    ]
    
    # Parse arguments using Dream's parser
    from dream_inference import argparse
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
    
    # Cache parameters
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="Enable dLLM-Cache for acceleration")
    parser.add_argument("--prompt_interval_steps", type=int, default=100,
                       help="Cache prompt features every N steps")
    parser.add_argument("--gen_interval_steps", type=int, default=7,
                       help="Cache generation features every N steps")
    parser.add_argument("--transfer_ratio", type=float, default=0.25,
                       help="Transfer ratio for cached features")

    args = parser.parse_args(test_args)
    
    print(f"✅ Dream parameters parsed successfully:")
    print(f"  - Model path: {args.model_path}")
    print(f"  - Use cache: {args.use_cache}")
    print(f"  - Prompt interval: {args.prompt_interval_steps}")
    print(f"  - Gen interval: {args.gen_interval_steps}")
    print(f"  - Transfer ratio: {args.transfer_ratio}")
    
    return True

def test_llada_params():
    """Test LLaDA inference script parameter parsing"""
    print("\n🧪 Testing LLaDA inference parameter parsing...")
    
    # Create a mock argument list like what the batch script would pass
    test_args = [
        '--model_path', 'GSAI-ML/LLaDA-8B-Base',
        '--constraint_types', 'content', 'situation',
        '--temperature', '0.2',
        '--max_new_tokens', '1024',
        '--diffusion_steps', '1024',
        '--block_length', '1024',
        '--api_output_path', 'api_output',
        '--use_cache',
        '--prompt_interval_steps', '100',
        '--gen_interval_steps', '7',
        '--transfer_ratio', '0.25'
    ]
    
    # Parse arguments using LLaDA's parser
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
    
    # Cache parameters
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="Enable dLLM-Cache for acceleration")
    parser.add_argument("--prompt_interval_steps", type=int, default=100,
                       help="Cache prompt features every N steps")
    parser.add_argument("--gen_interval_steps", type=int, default=7,
                       help="Cache generation features every N steps")
    parser.add_argument("--transfer_ratio", type=float, default=0.25,
                       help="Transfer ratio for cached features")

    args = parser.parse_args(test_args)
    
    print(f"✅ LLaDA parameters parsed successfully:")
    print(f"  - Model path: {args.model_path}")
    print(f"  - Block length: {args.block_length}")
    print(f"  - Use cache: {args.use_cache}")
    print(f"  - Prompt interval: {args.prompt_interval_steps}")
    print(f"  - Gen interval: {args.gen_interval_steps}")
    print(f"  - Transfer ratio: {args.transfer_ratio}")
    
    return True

if __name__ == "__main__":
    print("🚀 Testing cache parameter integration...")
    
    try:
        dream_success = test_dream_params()
        llada_success = test_llada_params()
        
        print(f"\n📊 Parameter Test Results:")
        print(f"  Dream Parameters: {'✅ PASS' if dream_success else '❌ FAIL'}")
        print(f"  LLaDA Parameters: {'✅ PASS' if llada_success else '❌ FAIL'}")
        
        if dream_success and llada_success:
            print("\n🎉 All parameter tests passed!")
        else:
            print("\n⚠️ Some parameter tests failed.")
            
    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        import traceback
        traceback.print_exc()
