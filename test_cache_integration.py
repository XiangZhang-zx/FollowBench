#!/usr/bin/env python3

import os
import sys
import torch
import json
from transformers import AutoModel, AutoTokenizer

# Add dLLM-cache to path
sys.path.append('/home/xz2649/projects/dLLM-cache')

def test_dream_cache():
    """Test Dream model with dLLM-Cache"""
    print("🧪 Testing Dream model with dLLM-Cache...")
    
    try:
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dllm_cache.hooks import register_cache_Dream
        from dataclasses import asdict
        
        # Load Dream model
        model_path = "Dream-org/Dream-v0-Base-7B"
        print(f"Loading Dream model: {model_path}")
        
        model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to("cuda").eval()
        
        # Initialize cache
        print("Initializing dLLM-Cache for Dream...")
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=100,
                    gen_interval_steps=7,
                    transfer_ratio=0.25,
                )
            )
        )
        register_cache_Dream(model, "model.layers")
        print("✅ Dream cache initialized successfully")
        
        # Test generation
        test_prompt = "Please write a Python function that calculates the factorial of a number."
        inputs = tokenizer(test_prompt, return_tensors='pt').to("cuda")
        
        # Reset cache
        feature_cache = dLLMCache()
        feature_cache.reset_cache(inputs['input_ids'].shape[1])
        
        print("Testing generation with cache...")
        with torch.no_grad():
            generated_ids = model.diffusion_generate(
                inputs=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                steps=256,
                max_new_tokens=128,
                temperature=0.2,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.0
            )
        
        # Decode response
        new_tokens = generated_ids[0][len(inputs['input_ids'][0]):]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"✅ Dream generation successful!")
        print(f"Generated text: {generated_text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Dream cache test failed: {e}")
        return False

def test_llada_cache():
    """Test LLaDA model with dLLM-Cache"""
    print("\n🧪 Testing LLaDA model with dLLM-Cache...")
    
    try:
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dllm_cache.hooks import register_cache_LLaDA
        from dataclasses import asdict
        
        # Import LLaDA generate function
        sys.path.append('/home/xz2649/projects/dLLM-cache')
        from utils import generate
        
        # Load LLaDA model
        model_path = "GSAI-ML/LLaDA-8B-Base"
        print(f"Loading LLaDA model: {model_path}")
        
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to("cuda").eval()
        
        # Initialize cache
        print("Initializing dLLM-Cache for LLaDA...")
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=100,
                    gen_interval_steps=7,
                    transfer_ratio=0.25,
                )
            )
        )
        register_cache_LLaDA(model, "model.transformer.blocks")
        print("✅ LLaDA cache initialized successfully")
        
        # Test generation
        test_prompt = "Please write a Python function that calculates the factorial of a number."
        inputs = tokenizer(test_prompt, return_tensors='pt').to("cuda")
        
        # Reset cache
        feature_cache = dLLMCache()
        feature_cache.reset_cache(inputs['input_ids'].shape[1])
        
        print("Testing generation with cache...")
        generated_ids = generate(
            model=model,
            input_ids=inputs['input_ids'],
            steps=256,
            gen_length=128,
            block_length=8,
            cfg_scale=0.0,
            remasking='low_confidence'
        )
        
        # Decode response
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"✅ LLaDA generation successful!")
        print(f"Generated text: {generated_text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ LLaDA cache test failed: {e}")
        return False

def test_cache_parameters():
    """Test different cache parameter configurations"""
    print("\n🧪 Testing cache parameter configurations...")
    
    try:
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dataclasses import asdict
        
        # Test different configurations
        configs = [
            {"prompt_interval_steps": 50, "gen_interval_steps": 5, "transfer_ratio": 0.1},
            {"prompt_interval_steps": 100, "gen_interval_steps": 7, "transfer_ratio": 0.25},
            {"prompt_interval_steps": 200, "gen_interval_steps": 10, "transfer_ratio": 0.5},
        ]
        
        for i, config in enumerate(configs):
            print(f"Testing config {i+1}: {config}")
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(**config)))
            cache = dLLMCache()
            cache.reset_cache(100)  # Test with 100 token prompt
            print(f"✅ Config {i+1} initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache parameter test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting dLLM-Cache integration tests...")
    
    # Test cache parameters first (lightweight)
    param_success = test_cache_parameters()
    
    # Test model-specific cache integration (requires GPU)
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
        
        # Test Dream cache (if model available)
        dream_success = test_dream_cache()
        
        # Test LLaDA cache (if model available)  
        llada_success = test_llada_cache()
        
        # Summary
        print("\n📊 Test Results:")
        print(f"  Cache Parameters: {'✅ PASS' if param_success else '❌ FAIL'}")
        print(f"  Dream Cache: {'✅ PASS' if dream_success else '❌ FAIL'}")
        print(f"  LLaDA Cache: {'✅ PASS' if llada_success else '❌ FAIL'}")
        
        if all([param_success, dream_success, llada_success]):
            print("\n🎉 All cache integration tests passed!")
        else:
            print("\n⚠️ Some tests failed. Check the logs above.")
    else:
        print("⚠️ No GPU available, skipping model tests")
        print(f"Cache Parameters: {'✅ PASS' if param_success else '❌ FAIL'}")
