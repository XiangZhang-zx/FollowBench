#!/usr/bin/env python3
"""
测试Base Model格式修改
验证去掉chat template后的prompt格式
"""

import sys
import os
sys.path.append('/home/xz2649/projects/FollowBench/code')

from model_inference_vllm import generate_prompts
from transformers import AutoTokenizer

def test_prompt_generation():
    """测试prompt生成"""
    print("🧪 测试Base Model Prompt生成")
    print("=" * 50)
    
    # 模拟FollowBench数据
    test_data = [
        {
            'prompt_new': 'Write a story about a cat. The story must be exactly 3 sentences long.'
        },
        {
            'prompt_new': 'List 5 animals. Each animal name must start with the letter "B".'
        },
        {
            'prompt_new': 'Explain photosynthesis in simple terms. Use exactly 2 paragraphs.'
        }
    ]
    
    print("📝 测试数据:")
    for i, example in enumerate(test_data, 1):
        print(f"  {i}. {example['prompt_new'][:50]}...")
    
    print("\n🔄 生成prompts...")
    
    # 测试不同tokenizer的情况
    tokenizers_to_test = [
        ('无tokenizer', None),
        ('DeepSeek Base', 'deepseek-ai/deepseek-llm-7b-base'),
        ('DeepSeek Chat', 'deepseek-ai/deepseek-llm-7b-chat'),
    ]
    
    for name, model_path in tokenizers_to_test:
        print(f"\n🔍 测试 {name}:")
        print("-" * 30)
        
        try:
            if model_path:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
                print(f"  Chat template: {'✅ 有' if has_template else '❌ 无'}")
            else:
                tokenizer = None
                print("  Chat template: ❌ 无tokenizer")
            
            # 生成prompts
            prompts = generate_prompts(test_data, tokenizer)
            
            print(f"  生成的prompts:")
            for i, prompt in enumerate(prompts, 1):
                print(f"    {i}. {repr(prompt[:60])}...")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")

def compare_old_vs_new_format():
    """比较旧格式vs新格式"""
    print("\n⚖️ 格式对比")
    print("=" * 50)
    
    test_prompt = "Write a story about a cat. The story must be exactly 3 sentences long."
    
    print("🔴 旧格式 (使用chat template):")
    try:
        tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', trust_remote_code=True)
        old_format = tokenizer.apply_chat_template(
            [{"role": "user", "content": test_prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"  {repr(old_format[:100])}...")
    except Exception as e:
        print(f"  错误: {e}")
    
    print("\n🟢 新格式 (简单格式):")
    new_format = test_prompt
    print(f"  {repr(new_format)}")
    
    print("\n📊 对比分析:")
    print("  旧格式特点:")
    print("    - 包含特殊标记 (<|im_start|>, <|im_end|>等)")
    print("    - 对话结构化格式")
    print("    - 适合instruct模型")
    print("  新格式特点:")
    print("    - 纯文本，无特殊标记")
    print("    - 直接的指令格式")
    print("    - 适合base模型")

def main():
    """主测试函数"""
    print("🎯 FollowBench Base Model格式测试")
    print("=" * 60)
    
    test_prompt_generation()
    compare_old_vs_new_format()
    
    print("\n✅ 修改总结:")
    print("=" * 50)
    print("1. ✅ 去掉了chat template的使用")
    print("2. ✅ 使用简单的原始prompt格式")
    print("3. ✅ 适合base model的训练数据格式")
    print("4. ✅ 避免了base model看到未见过的对话标记")
    
    print("\n🚀 现在可以运行FollowBench测试:")
    print("  python3 model_inference_vllm.py --model_path <base_model_path> ...")

if __name__ == "__main__":
    main()
