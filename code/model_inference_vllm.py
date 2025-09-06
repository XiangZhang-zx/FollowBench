import os
import json
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from transformers import set_seed, AutoTokenizer
import numpy as np

from utils import convert_to_api_input, add_model_args



# Default (ChatML format)

SYSTEM_PREFIX="<|im_start|>system\n"
SYSTEM_SUFFIX="<|im_end|>\n"
USER_PREFIX="<|im_start|>user\n"
USER_SUFFIX="<|im_end|>\n"
ASSISTANT_PREFIX="<|im_start|>assistant\n"
ASSISTANT_SUFFIX="<|im_end|>\n"


def generate_prompts(data, tokenizer=None):
    # 强制使用简单格式，不使用chat template（适合base model比较）
    print(f"Using simple format without chat template for base model comparison!")

    # 对于base model，使用最简单的格式
    prompts = [f"{example['prompt_new']}" for example in data]
    return prompts


def generate_responses(model, sampling_params, data, args, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = generate_prompts(data, tokenizer=tokenizer)

    output_list = []

    responses = model.generate(prompts, sampling_params=sampling_params)

    for i, response in enumerate(responses):
        # Clean up the response text by removing any chat template artifacts
        response_text = response.outputs[0].text.strip()
        # Remove common chat template endings
        for ending in ["<|im_end|>", "<|eot_id|>", "</s>"]:
            if ending in response_text:
                response_text = response_text.split(ending)[0]
        output_list.append({'prompt': data[i]["prompt_new"], "choices": [{"message": {"content": response_text.strip()}}]})
    return output_list


def run_inference(args):
    devices = args.gpus.split(",")
    model = LLM(model=args.model_path, tensor_parallel_size=args.num_gpus, trust_remote_code=True, tokenizer=args.model_path)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    set_seed(42)
    for constraint_type in args.constraint_types:
        data = []
        with open(os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        # 限制样本数量
        if args.limit_samples > 0:
            data = data[:args.limit_samples]
            print(f"限制处理样本数量为: {len(data)}")

        # 限制样本数量用于测试
        if hasattr(args, 'limit_samples') and args.limit_samples > 0:
            data = data[:args.limit_samples]
            print(f"限制处理样本数量: {len(data)}")

        output_list = generate_responses(model, sampling_params, data, args, devices[0:args.num_gpus])

        os.makedirs(f"{args.api_output_path}/{args.model_path}", exist_ok=True)
        with open(os.path.join(args.api_output_path, f"{args.model_path}/{constraint_type}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for o in output_list:
                output_file.write(json.dumps(o) + "\n")



def main():
    parser = ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'situation', 'style', 'format', 'example', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_input_path", type=str, default="api_input")
    parser.add_argument("--api_output_path", type=str, default="api_output_vllm")
    parser.add_argument("--limit_samples", type=int, default=0, help="限制处理的样本数量，0表示处理全部")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    os.makedirs(args.api_input_path, exist_ok=True)
    for constraint_type in args.constraint_types:
        convert_to_api_input(
                            data_path=args.data_path, 
                            api_input_path=args.api_input_path, 
                            constraint_type=constraint_type
                            )

    run_inference(args)

if __name__ == '__main__':
    main()