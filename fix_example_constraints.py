#!/usr/bin/env python3
import json

def fix_example_constraints():
    """删除过长的example约束样本并保存修复后的文件"""
    
    # 读取原始数据
    with open('data/example_constraints.json', 'r') as f:
        data = json.load(f)
    
    print(f"原始样本数: {len(data)}")
    
    # 要删除的example_id（这些包含超长样本）
    problematic_examples = [4, 18, 24]
    
    # 统计删除前的信息
    before_stats = {}
    for item in data:
        example_id = item['example_id']
        if example_id in problematic_examples:
            if example_id not in before_stats:
                before_stats[example_id] = []
            before_stats[example_id].append({
                'level': item['level'],
                'length': len(item['instruction']),
                'tokens': len(item['instruction']) // 4
            })
    
    print("\n要删除的样本组:")
    for example_id in problematic_examples:
        if example_id in before_stats:
            print(f"Example {example_id}:")
            for info in before_stats[example_id]:
                print(f"  Level {info['level']}: {info['length']} 字符, ~{info['tokens']} tokens")
    
    # 过滤掉问题样本
    filtered_data = [item for item in data if item['example_id'] not in problematic_examples]
    
    print(f"\n删除后样本数: {len(filtered_data)}")
    print(f"删除了 {len(data) - len(filtered_data)} 个样本")
    
    # 检查剩余样本的长度分布
    remaining_lengths = [len(item['instruction']) for item in filtered_data]
    max_length = max(remaining_lengths)
    max_tokens = max_length // 4
    
    print(f"\n剩余样本统计:")
    print(f"最长样本: {max_length} 字符, ~{max_tokens} tokens")
    print(f"超过2000字符的样本: {sum(1 for x in remaining_lengths if x > 2000)}")
    print(f"超过5000字符的样本: {sum(1 for x in remaining_lengths if x > 5000)}")
    print(f"超过10000字符的样本: {sum(1 for x in remaining_lengths if x > 10000)}")
    
    # 备份原文件
    import shutil
    shutil.copy('data/example_constraints.json', 'data/example_constraints.json.backup')
    print(f"\n已备份原文件到: data/example_constraints.json.backup")
    
    # 保存修复后的数据
    with open('data/example_constraints.json', 'w') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"已保存修复后的文件: data/example_constraints.json")
    
    # 显示剩余的example_id分布
    remaining_examples = set(item['example_id'] for item in filtered_data)
    print(f"\n剩余的example_id: {sorted(remaining_examples)}")
    print(f"剩余example数量: {len(remaining_examples)}")

if __name__ == "__main__":
    fix_example_constraints()
