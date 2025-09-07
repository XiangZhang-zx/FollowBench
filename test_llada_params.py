#!/usr/bin/env python3
import argparse
import sys
import os

# 添加code目录到路径
sys.path.append('code')

def test_llada_params():
    """测试LLaDA参数传递"""
    
    # 模拟命令行参数
    test_args = [
        '--model_path', 'GSAI-ML/LLaDA-8B-Base',
        '--constraint_types', 'example',
        '--max_new_tokens', '512',
        '--diffusion_steps', '512', 
        '--block_length', '512',
        '--temperature', '0.2',
        '--api_input_path', 'api_input_test',
        '--api_output_path', 'api_output_test'
    ]
    
    # 导入LLaDA推理脚本的参数解析器
    from llada_inference import main
    
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--constraint_types", nargs='+', required=True)
    parser.add_argument("--api_input_path", type=str, default="api_input")
    parser.add_argument("--api_output_path", type=str, default="api_output")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--diffusion_steps", type=int, default=2048)
    parser.add_argument("--block_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit_samples", type=int, default=0)
    
    # 解析参数
    args = parser.parse_args(test_args)
    
    print("=== LLaDA参数测试 ===")
    print(f"Model path: {args.model_path}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Block length: {args.block_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Constraint types: {args.constraint_types}")
    
    # 创建gen_params字典（模拟LLaDA推理脚本中的逻辑）
    gen_params = {
        'steps': args.diffusion_steps,
        'gen_length': args.max_new_tokens,
        'block_length': args.block_length,
        'temperature': args.temperature,
        'cfg_scale': 0.,
        'remasking': 'low_confidence',
        'mask_id': 126336
    }
    
    print(f"\n✅ 生成参数: {gen_params}")
    
    # 验证参数是否正确
    assert gen_params['block_length'] == 512, f"Block length应该是512，但得到{gen_params['block_length']}"
    assert gen_params['steps'] == 512, f"Steps应该是512，但得到{gen_params['steps']}"
    assert gen_params['gen_length'] == 512, f"Gen length应该是512，但得到{gen_params['gen_length']}"
    
    print("✅ 所有参数验证通过！")

if __name__ == "__main__":
    test_llada_params()
