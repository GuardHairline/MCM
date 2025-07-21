#!/usr/bin/env python3
"""
测试任务配置生成器

验证各种配置组合的正确性。
"""

import json
import tempfile
import os
from scripts.generate_task_config import TaskConfigGenerator


def test_mode_suffix():
    """测试模式后缀生成"""
    generator = TaskConfigGenerator()
    
    test_cases = [
        (["text_only", "text_only", "text_only"], "t"),
        (["multimodal", "multimodal", "multimodal"], "m"),
        (["text_only", "multimodal", "text_only"], "t2m"),
        (["multimodal", "text_only", "multimodal"], "t2m"),
        ([], ""),
    ]
    
    print("=== 测试模式后缀生成 ===")
    for modes, expected in test_cases:
        result = generator.determine_mode_suffix(modes)
        status = "✓" if result == expected else "✗"
        print(f"{status} {modes} -> {result} (期望: {expected})")


def test_file_naming():
    """测试文件命名"""
    generator = TaskConfigGenerator()
    
    test_cases = [
        # (env, dataset, strategy, modes, use_label_embedding, expected_suffix)
        ("local", "twitter2015", "ewc", ["text_only", "text_only"], True, "t"),
        ("local", "twitter2015", "ewc", ["multimodal", "multimodal"], True, "m"),
        ("local", "twitter2015", "ewc", ["text_only", "multimodal"], True, "t2m"),
        ("server", "twitter2015", "ewc", ["text_only", "multimodal"], False, "t2m"),
    ]
    
    print("\n=== 测试文件命名 ===")
    for env, dataset, strategy, modes, use_label_emb, expected_suffix in test_cases:
        file_names = generator.generate_file_names(env, dataset, strategy, modes, use_label_emb)
        result_suffix = file_names["mode_suffix"]
        status = "✓" if result_suffix == expected_suffix else "✗"
        print(f"{status} {env}_{dataset}_{strategy}_{modes} -> {result_suffix} (期望: {expected_suffix})")
        print(f"    模型文件: {file_names['model_name']}")
        print(f"    训练信息: {file_names['train_info_json']}")


def test_task_config_generation():
    """测试任务配置生成"""
    generator = TaskConfigGenerator()
    
    test_cases = [
        {
            "name": "全text_only模式",
            "env": "local",
            "dataset": "twitter2015",
            "strategy": "ewc",
            "task_sequence": ["masc", "mate", "mner", "mabsa"],
            "mode_sequence": ["text_only", "text_only", "text_only", "text_only"],
            "use_label_embedding": True,
            "expected_suffix": "t"
        },
        {
            "name": "全multimodal模式",
            "env": "local",
            "dataset": "twitter2015",
            "strategy": "ewc",
            "task_sequence": ["masc", "mate", "mner", "mabsa"],
            "mode_sequence": ["multimodal", "multimodal", "multimodal", "multimodal"],
            "use_label_embedding": True,
            "expected_suffix": "m"
        },
        {
            "name": "混合模式",
            "env": "local",
            "dataset": "twitter2015",
            "strategy": "ewc",
            "task_sequence": ["masc", "mate", "mner", "mabsa"],
            "mode_sequence": ["text_only", "multimodal", "text_only", "multimodal"],
            "use_label_embedding": True,
            "expected_suffix": "t2m"
        },
        {
            "name": "服务器环境",
            "env": "server",
            "dataset": "twitter2015",
            "strategy": "ewc",
            "task_sequence": ["masc", "mate", "mner", "mabsa"],
            "mode_sequence": ["text_only", "multimodal", "text_only", "multimodal"],
            "use_label_embedding": False,
            "expected_suffix": "t2m"
        }
    ]
    
    print("\n=== 测试任务配置生成 ===")
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        
        try:
            config = generator.generate_task_sequence_config(
                env=test_case["env"],
                dataset=test_case["dataset"],
                task_sequence=test_case["task_sequence"],
                mode_sequence=test_case["mode_sequence"],
                strategy=test_case["strategy"],
                use_label_embedding=test_case["use_label_embedding"]
            )
            
            # 验证配置
            assert len(config["tasks"]) == len(test_case["task_sequence"]), "任务数量不匹配"
            assert config["mode_suffix"] == test_case["expected_suffix"], f"模式后缀不匹配: {config['mode_suffix']} != {test_case['expected_suffix']}"
            
            print(f"✓ 配置生成成功")
            print(f"  任务数量: {len(config['tasks'])}")
            print(f"  模式后缀: {config['mode_suffix']}")
            print(f"  模式序列: {config['mode_sequence']}")
            
            # 验证每个任务的模式
            for i, (task, expected_mode) in enumerate(zip(config["tasks"], test_case["mode_sequence"])):
                assert task["mode"] == expected_mode, f"任务{i+1}模式不匹配: {task['mode']} != {expected_mode}"
                print(f"  任务{i+1}: {task['task_name']} - {task['mode']}")
            
            # 验证文件路径
            model_path = config["global_params"]["output_model_path"]
            train_info_path = config["global_params"]["train_info_json"]
            print(f"  模型文件: {model_path}")
            print(f"  训练信息: {train_info_path}")
            
        except Exception as e:
            print(f"✗ 配置生成失败: {e}")


def test_config_file_output():
    """测试配置文件输出"""
    generator = TaskConfigGenerator()
    
    print("\n=== 测试配置文件输出 ===")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        # 生成配置
        config = generator.generate_task_sequence_config(
            env="local",
            dataset="twitter2015",
            task_sequence=["masc", "mate", "mner", "mabsa"],
            mode_sequence=["text_only", "multimodal", "text_only", "multimodal"],
            strategy="ewc",
            use_label_embedding=True
        )
        
        # 保存到文件
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 读取并验证
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["mode_suffix"] == "t2m", "模式后缀不正确"
        assert len(loaded_config["tasks"]) == 4, "任务数量不正确"
        assert len(loaded_config["mode_sequence"]) == 4, "模式序列长度不正确"
        
        print("✓ 配置文件输出测试通过")
        print(f"  文件: {temp_file}")
        print(f"  模式后缀: {loaded_config['mode_suffix']}")
        print(f"  任务数量: {len(loaded_config['tasks'])}")
        
    except Exception as e:
        print(f"✗ 配置文件输出测试失败: {e}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    """运行所有测试"""
    print("开始测试任务配置生成器...")
    
    test_mode_suffix()
    test_file_naming()
    test_task_config_generation()
    test_config_file_output()
    
    print("\n=== 所有测试完成 ===")


if __name__ == "__main__":
    main() 