#!/usr/bin/env python3
import json
import os
import tempfile

def create_test_data():
    """创建只包含样本88的测试数据"""
    
    # 读取原始数据
    with open('data/example_constraints.json', 'r') as f:
        data = json.load(f)
    
    # 找到样本88
    sample_88 = data[88]
    
    print(f"=== 样本88信息 ===")
    print(f"Example ID: {sample_88['example_id']}")
    print(f"Level: {sample_88['level']}")
    print(f"字符数: {len(sample_88['instruction'])}")
    print(f"估算tokens: ~{len(sample_88['instruction'])//4}")
    print()
    
    # 创建测试用的单样本文件
    test_data = [sample_88]
    
    # 保存到临时文件
    with open('data/example_constraints_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("✅ 已创建测试数据文件: data/example_constraints_test.json")
    return sample_88

def test_ar_models():
    """测试AR模型"""
    print("\n=== 测试AR模型 ===")
    
    # 创建测试用的API输入
    os.system("python -c \"from code.utils import convert_to_api_input; convert_to_api_input('data', 'api_input_test', 'example')\"")

    # 测试Mistral
    print("\n🔍 测试 Mistral-7B-v0.3...")
    cmd = """python code/model_inference_vllm.py \
        --model-path "mistralai/Mistral-7B-v0.3" \
        --constraint_types example \
        --temperature 0.2 \
        --max-tokens 512 \
        --num-gpus 1 \
        --gpus "0" \
        --api_input_path "api_input_test" \
        --api_output_path "api_output_test" \
        --limit_samples 1"""
    
    result = os.system(cmd)
    if result == 0:
        print("✅ Mistral测试成功")
    else:
        print("❌ Mistral测试失败")
    
    # 测试DeepSeek
    print("\n🔍 测试 DeepSeek-7B-base...")
    cmd = """python code/model_inference_vllm.py \
        --model-path "deepseek-ai/deepseek-llm-7b-base" \
        --constraint_types example \
        --temperature 0.2 \
        --max-tokens 512 \
        --num-gpus 1 \
        --gpus "0" \
        --api_input_path "api_input_test" \
        --api_output_path "api_output_test" \
        --limit_samples 1"""
    
    result = os.system(cmd)
    if result == 0:
        print("✅ DeepSeek测试成功")
    else:
        print("❌ DeepSeek测试失败")

def test_llada():
    """测试LLaDA模型"""
    print("\n=== 测试LLaDA模型 ===")
    
    cmd = """python code/llada_inference.py \
        --model_path "GSAI-ML/LLaDA-8B-Base" \
        --constraint_types example \
        --temperature 0.2 \
        --max_new_tokens 512 \
        --diffusion_steps 512 \
        --block_length 512 \
        --api_input_path "api_input_test" \
        --api_output_path "api_output_test" \
        --limit_samples 1"""
    
    result = os.system(cmd)
    if result == 0:
        print("✅ LLaDA测试成功")
    else:
        print("❌ LLaDA测试失败")

def test_dream():
    """测试Dream模型"""
    print("\n=== 测试Dream模型 ===")
    
    cmd = """python code/dream_inference.py \
        --model_path "Dream-org/Dream-v0-Base-7B" \
        --constraint_types example \
        --temperature 0.2 \
        --max_new_tokens 512 \
        --diffusion_steps 512 \
        --api_input_path "api_input_test" \
        --api_output_path "api_output_test" \
        --limit_samples 1"""
    
    result = os.system(cmd)
    if result == 0:
        print("✅ Dream测试成功")
    else:
        print("❌ Dream测试失败")

def check_outputs():
    """检查输出结果"""
    print("\n=== 检查输出结果 ===")
    
    models = [
        "mistralai/Mistral-7B-v0.3",
        "deepseek-ai/deepseek-llm-7b-base", 
        "GSAI-ML/LLaDA-8B-Base",
        "Dream-org/Dream-v0-Base-7B"
    ]
    
    for model in models:
        output_file = f"api_output_test/{model}/example_constraint.jsonl"
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    line = f.readline()
                    if line.strip():
                        data = json.loads(line)
                        content = data['choices'][0]['message']['content']
                        print(f"\n✅ {model}:")
                        print(f"  输出长度: {len(content)} 字符")
                        print(f"  前100字符: {content[:100]}...")
                    else:
                        print(f"\n❌ {model}: 输出文件为空")
            except Exception as e:
                print(f"\n❌ {model}: 读取输出失败 - {e}")
        else:
            print(f"\n❌ {model}: 输出文件不存在")

if __name__ == "__main__":
    # 创建测试数据
    sample_88 = create_test_data()
    
    # 创建测试目录
    os.makedirs("api_input_test", exist_ok=True)
    os.makedirs("api_output_test", exist_ok=True)
    
    # 运行测试
    test_ar_models()
    test_llada() 
    test_dream()
    
    # 检查结果
    check_outputs()
    
    print("\n=== 测试完成 ===")
    print("如果某个模型失败，可能是因为:")
    print("1. 输入太长，超过模型context限制")
    print("2. 内存不足")
    print("3. 模型加载失败")
    print("4. CUDA内存不足")
