#!/usr/bin/env python3
"""
测试简化后的任务配置生成器

验证删除use_simplified参数后的功能。
"""

import json
from scripts.generate_task_config import TaskConfigGenerator


def test_dataset_files():
    """测试数据集文件获取"""
    generator = TaskConfigGenerator()
    
    print("=== 测试数据集文件获取 ===")
    
    test_cases = [
        ("masc", "200"),
        ("masc", "twitter2015"),
        ("mner", "200"),
        ("mner", "twitter2015"),
    ]
    
    for task_name, dataset in test_cases:
        print(f"\n任务: {task_name}, 数据集: {dataset}")
        files = generator.get_dataset_files(task_name, dataset)
        print(f"  训练文件: {files['train']}")
        print(f"  验证文件: {files['dev']}")
        print(f"  测试文件: {files['test']}")


def test_config_generation():
    """测试配置生成"""
    generator = TaskConfigGenerator()
    
    print("\n=== 测试配置生成 ===")
    
    # 测试200数据集
    print("\n1. 测试200数据集配置:")
    config = generator.generate_task_sequence_config(
        env="local",
        dataset="200",
        task_sequence=["masc", "mate", "mner", "mabsa"],
        mode_sequence=["text_only", "multimodal", "text_only", "multimodal"],
        strategy="ewc",
        use_label_embedding=True
    )
    
    print(f"  模式后缀: {config['mode_suffix']}")
    print(f"  任务数量: {len(config['tasks'])}")
    print(f"  模型文件: {config['global_params']['output_model_path']}")
    
    # 验证每个任务的模式
    for i, (task, mode) in enumerate(zip(config["tasks"], config["mode_sequence"])):
        print(f"  任务{i+1}: {task['task_name']} - {task['mode']}")
    
    # 测试twitter2015数据集
    print("\n2. 测试twitter2015数据集配置:")
    config = generator.generate_task_sequence_config(
        env="local",
        dataset="twitter2015",
        task_sequence=["masc", "mate", "mner", "mabsa"],
        mode_sequence=["multimodal", "multimodal", "multimodal", "multimodal"],
        strategy="ewc",
        use_label_embedding=False
    )
    
    print(f"  模式后缀: {config['mode_suffix']}")
    print(f"  任务数量: {len(config['tasks'])}")
    print(f"  模型文件: {config['global_params']['output_model_path']}")


def test_file_naming():
    """测试文件命名"""
    generator = TaskConfigGenerator()
    
    print("\n=== 测试文件命名 ===")
    
    test_cases = [
        ("local", "200", "ewc", ["text_only", "text_only", "text_only", "text_only"], True),
        ("local", "200", "ewc", ["multimodal", "multimodal", "multimodal", "multimodal"], True),
        ("local", "200", "ewc", ["text_only", "multimodal", "text_only", "multimodal"], True),
        ("server", "200", "ewc", ["text_only", "multimodal", "text_only", "multimodal"], False),
    ]
    
    for env, dataset, strategy, modes, use_label_emb in test_cases:
        file_names = generator.generate_file_names(env, dataset, strategy, modes, use_label_emb)
        print(f"\n{env}_{dataset}_{strategy}_{modes} (label_emb: {use_label_emb}):")
        print(f"  模式后缀: {file_names['mode_suffix']}")
        print(f"  模型文件: {file_names['model_name']}")
        print(f"  训练信息: {file_names['train_info_json']}")


def main():
    """运行所有测试"""
    print("开始测试简化后的任务配置生成器...")
    
    test_dataset_files()
    test_config_generation()
    test_file_naming()
    
    print("\n=== 所有测试完成 ===")
    print("\n说明:")
    print("- 200数据集: 使用简化文件 (train__.txt, dev__.txt, test__.txt)")
    print("- twitter2015/twitter2017/mix: 使用完整文件 (train.txt, dev.txt, test.txt)")
    print("- 删除了use_simplified参数，简化了配置")


if __name__ == "__main__":
    main() 