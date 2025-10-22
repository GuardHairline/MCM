#!/usr/bin/env python3
"""
测试超参数搜索配置生成器

验证：
1. 配置文件生成是否正确
2. 路径设置是否正确
3. 可以导入必要的模块
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("超参数配置生成器测试")
print("=" * 70)
print(f"项目根目录: {project_root}")
print(f"当前工作目录: {Path.cwd()}")
print()

# 测试1: 导入模块
print("测试1: 导入必要模块...")
try:
    from scripts.generate_task_config import TaskConfigGenerator
    from scripts.generate_masc_hyperparameter_configs import HyperparameterSearchGenerator
    print("✓ 成功导入配置生成器模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 检查配置文件
print("\n测试2: 检查生成的配置文件...")
config_dir = project_root / "scripts" / "configs" / "hyperparam_search"

if not config_dir.exists():
    print(f"⚠️  配置目录不存在: {config_dir}")
    print("   请先运行: python scripts/generate_masc_hyperparameter_configs.py")
else:
    # 检查索引文件
    index_file = config_dir / "config_index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index = json.load(f)
        print(f"✓ 找到索引文件，包含 {index['total_configs']} 个配置")
        print(f"  策略: {', '.join(index['strategies'])}")
    else:
        print(f"❌ 索引文件不存在: {index_file}")
    
    # 随机检查一个配置文件
    config_files = list(config_dir.glob("server_*.json"))
    if config_files:
        test_config = config_files[0]
        print(f"\n  检查配置文件: {test_config.name}")
        with open(test_config, 'r') as f:
            config = json.load(f)
        
        # 验证必要字段
        required_fields = ["tasks", "global_params", "strategy", "env", "dataset", "hyperparameters"]
        missing_fields = [f for f in required_fields if f not in config]
        
        if missing_fields:
            print(f"  ❌ 缺少字段: {', '.join(missing_fields)}")
        else:
            print(f"  ✓ 配置文件结构正确")
            print(f"  - 任务数: {len(config['tasks'])}")
            print(f"  - 策略: {config['strategy']}")
            print(f"  - 超参数: lr={config['hyperparameters']['lr']}, "
                  f"step_size={config['hyperparameters']['step_size']}, "
                  f"gamma={config['hyperparameters']['gamma']}")
            
            # 检查任务配置
            for i, task in enumerate(config['tasks']):
                print(f"  - 任务{i+1}: {task['task_name']} ({task['mode']})")
    else:
        print(f"  ⚠️  未找到配置文件")

# 测试3: 检查脚本文件
print("\n测试3: 检查生成的脚本文件...")
scripts_to_check = [
    "run_all_experiments.sh",
    "start_experiments_detached.sh",
    "stop_all_experiments.sh",
    "README.md"
]

for script_name in scripts_to_check:
    script_path = config_dir / script_name
    if script_path.exists():
        size = script_path.stat().st_size
        print(f"  ✓ {script_name} ({size} bytes)")
        
        # 检查shell脚本的路径设置
        if script_name.endswith('.sh'):
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'PROJECT_ROOT' in content:
                    # 提取PROJECT_ROOT设置
                    for line in content.split('\n'):
                        if 'PROJECT_ROOT=' in line and not line.strip().startswith('#'):
                            print(f"    路径设置: {line.strip()}")
                            break
    else:
        print(f"  ❌ {script_name} 不存在")

# 测试4: 验证可以导入训练模块
print("\n测试4: 验证训练模块...")
try:
    from scripts.train_with_zero_shot import load_task_config
    print("✓ 成功导入 scripts.train_with_zero_shot")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("  请确保在项目根目录运行此脚本")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)

