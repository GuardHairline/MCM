#!/usr/bin/env python3
"""
合并Twitter2015和Twitter2017的描述文件，生成mix数据集的描述文件
"""

import json
from pathlib import Path

def merge_description_files():
    """合并两个描述文件"""
    
    # 输入文件路径
    twitter2015_file = Path("reference/DEQA/DEQA/datasets/release/twitter2015/description_roberta.jsonl")
    twitter2017_file = Path("reference/DEQA/DEQA/datasets/release/twitter2017/description_roberta.jsonl")
    
    # 输出目录和文件
    output_dir = Path("reference/DEQA/DEQA/datasets/release/mix")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "description_roberta.jsonl"
    
    print(f"合并描述文件:")
    print(f"  输入1: {twitter2015_file}")
    print(f"  输入2: {twitter2017_file}")
    print(f"  输出: {output_file}")
    print()
    
    # 检查输入文件
    if not twitter2015_file.exists():
        print(f"❌ 文件不存在: {twitter2015_file}")
        return False
    
    if not twitter2017_file.exists():
        print(f"❌ 文件不存在: {twitter2017_file}")
        return False
    
    # 读取并合并
    descriptions = []
    
    # 读取twitter2015
    print("读取 twitter2015...")
    with open(twitter2015_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                descriptions.append(json.loads(line))
    print(f"  读取了 {len(descriptions)} 条记录")
    
    # 读取twitter2017
    print("读取 twitter2017...")
    count_2017 = 0
    with open(twitter2017_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                descriptions.append(json.loads(line))
                count_2017 += 1
    print(f"  读取了 {count_2017} 条记录")
    
    # 写入合并后的文件
    print(f"\n写入合并文件...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for desc in descriptions:
            f.write(json.dumps(desc, ensure_ascii=False) + '\n')
    
    print(f"✓ 成功合并！总共 {len(descriptions)} 条记录")
    print(f"✓ 输出文件: {output_file}")
    
    return True


if __name__ == "__main__":
    success = merge_description_files()
    if success:
        print("\n✅ 合并完成！")
    else:
        print("\n❌ 合并失败！")

