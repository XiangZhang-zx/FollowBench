import torch
import json
from transformers import AutoModel, AutoTokenizer

def test_dream_with_sample175():
    """测试Dream模型处理样本175"""
    print("=== 测试Dream模型 ===")

    # 读取样本175
    with open('data/example_constraints_175.json', 'r') as f:
        data = json.load(f)
    sample_content = data[0]['instruction']

    print(f"样本175长度: {len(sample_content)} 字符")
    print(f"估算tokens: ~{len(sample_content) // 4}")
    
    try:
        model_path = "Dream-org/Dream-v0-Base-7B"
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to("cuda").eval()

        # 直接使用样本175内容，不使用chat template（base model）
        inputs = tokenizer(sample_content, return_tensors="pt")
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")
        
        print(f"输入tokens: {input_ids.shape[1]}")


        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            output_history=True,
            return_dict_in_generate=True,
            steps=512,
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
        )
        generations = [
            tokenizer.decode(g[len(p) :].tolist())
            for p, g in zip(input_ids, output.sequences)
        ]

        result = generations[0].split(tokenizer.eos_token)[0]
        print(f"✅ Dream生成成功")
        print(f"生成长度: {len(result)} 字符")
        print(f"前100字符: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Dream测试失败: {e}")
        return False

def test_llada_with_sample175():
    """测试LLaDA模型处理样本175"""
    print("\n=== 测试LLaDA模型 ===")

    # 读取样本175
    with open('data/example_constraints_175.json', 'r') as f:
        data = json.load(f)
    sample_content = data[0]['instruction']

    print(f"样本175长度: {len(sample_content)} 字符")
    print(f"估算tokens: ~{len(sample_content) // 4}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
        model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.to("cuda").eval()
        
        # 直接使用样本175内容，不使用chat template（base model）
        inputs = tokenizer(sample_content, return_tensors="pt")
        input_ids = inputs.input_ids.to(device="cuda")
        
        print(f"输入tokens: {input_ids.shape[1]}")
        
        
        # 使用LLaDA的生成方法
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"✅ LLaDA生成成功")
        print(f"生成长度: {len(generated_text)} 字符")
        print(f"前100字符: {generated_text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LLaDA测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试样本175在扩散模型上的表现...")

    dream_success = test_dream_with_sample175()
    llada_success = test_llada_with_sample175()

    print(f"\n=== 测试结果 ===")
    print(f"Dream: {'✅ 成功' if dream_success else '❌ 失败'}")
    print(f"LLaDA: {'✅ 成功' if llada_success else '❌ 失败'}")

    if not dream_success and not llada_success:
        print("\n两个扩散模型都无法处理样本175，可能是因为输入过长")
    elif dream_success and llada_success:
        print("\n两个扩散模型都能处理样本175")
    else:
        print("\n只有一个扩散模型能处理样本175")
