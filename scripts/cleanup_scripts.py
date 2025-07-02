#!/usr/bin/env python3
# scripts/cleanup_scripts.py
"""
脚本清理工具
用于清理旧的脚本文件，保留新生成的标准化脚本
"""

import os
import shutil
from pathlib import Path


def get_old_script_patterns():
    """获取旧脚本的文件模式"""
    return [
        # 旧的命名模式
        "strain_AllTask_*",
        "train_AllTask_*",
        "strain_AllTask_2015_*",
        "strain_AllTask_AllDataset_*",
        "train_AllTask_2015_*",
        "train_AllTask_AllDataset_*",
        "train_AllTask_200_*",
        "train_AllTask_twitter2015_*",
        "twitter2015-*",
        "strain_AllTask_200_*",
        "train_masc_*",
        "all.sh",
        "all_ser.sh",
        "stmp.sh",
        "replay_test.sh",
        "location_question_append.sh",
        "train_TaskName_Dataset_Metrics_Fusion.sh"
    ]


def get_new_script_patterns():
    """获取新脚本的文件模式"""
    return [
        # 新的标准化命名模式
        "strain_AllTask_twitter2015_*_multi.sh",
        "strain_AllTask_twitter2017_*_multi.sh",
        "train_AllTask_200_*_multi.sh",
        "train_AllTask_twitter2015_*_multi.sh",
        "train_AllTask_twitter2017_*_multi.sh"
    ]


def backup_old_scripts():
    """备份旧脚本"""
    scripts_dir = Path("scripts")
    backup_dir = scripts_dir / "backup_old_scripts"
    
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    backup_dir.mkdir(exist_ok=True)
    
    old_patterns = get_old_script_patterns()
    backed_up_files = []
    
    print("开始备份旧脚本...")
    
    for pattern in old_patterns:
        for script_file in scripts_dir.glob(pattern):
            if script_file.is_file() and script_file.suffix == '.sh':
                # 备份文件
                backup_path = backup_dir / script_file.name
                shutil.copy2(script_file, backup_path)
                backed_up_files.append(script_file.name)
                print(f"  ✓ 已备份: {script_file.name}")
    
    print(f"总共备份了 {len(backed_up_files)} 个旧脚本到 {backup_dir}")
    return backup_dir, backed_up_files


def remove_old_scripts():
    """删除旧脚本"""
    scripts_dir = Path("scripts")
    old_patterns = get_old_script_patterns()
    removed_files = []
    
    print("开始删除旧脚本...")
    
    for pattern in old_patterns:
        for script_file in scripts_dir.glob(pattern):
            if script_file.is_file() and script_file.suffix == '.sh':
                # 删除文件
                script_file.unlink()
                removed_files.append(script_file.name)
                print(f"  ✓ 已删除: {script_file.name}")
    
    print(f"总共删除了 {len(removed_files)} 个旧脚本")
    return removed_files


def list_current_scripts():
    """列出当前脚本"""
    scripts_dir = Path("scripts")
    current_scripts = []
    
    print("当前脚本列表:")
    
    for script_file in scripts_dir.glob("*.sh"):
        if script_file.is_file():
            current_scripts.append(script_file.name)
            print(f"  - {script_file.name}")
    
    print(f"总共 {len(current_scripts)} 个脚本")
    return current_scripts


def create_migration_guide():
    """创建迁移指南"""
    guide_content = """# 脚本迁移指南

## 迁移概述
本项目已对训练脚本进行了标准化重构，旧脚本已被备份并删除。

## 旧脚本备份位置
所有旧脚本已备份到: `scripts/backup_old_scripts/`

## 新旧脚本对应关系

### 服务器版本
| 旧脚本 | 新脚本 |
|--------|--------|
| `strain_AllTask_2015_none_text.sh` | `strain_AllTask_twitter2015_none_multi.sh` |
| `strain_AllTask_2015_ewc_text.sh` | `strain_AllTask_twitter2015_ewc_multi.sh` |
| `strain_AllTask_2015_replay_text.sh` | `strain_AllTask_twitter2015_replay_multi.sh` |
| `strain_AllTask_2015_lwf_text.sh` | `strain_AllTask_twitter2015_lwf_multi.sh` |
| `strain_AllTask_2015_si_text.sh` | `strain_AllTask_twitter2015_si_multi.sh` |
| `strain_AllTask_2015_mas_text.sh` | `strain_AllTask_twitter2015_mas_multi.sh` |
| `strain_AllTask_2015_mymethod_text.sh` | `strain_AllTask_twitter2015_mymethod_multi.sh` |
| `strain_AllTask_AllDataset_moe_m.sh` | `strain_AllTask_twitter2015_moe_multi.sh` |

### 本地版本
| 旧脚本 | 新脚本 |
|--------|--------|
| `train_AllTask_200_none_text.sh` | `train_AllTask_200_none_multi.sh` |
| `train_AllTask_200_ewc_text.sh` | `train_AllTask_200_ewc_multi.sh` |
| `train_AllTask_200_replay_text.sh` | `train_AllTask_200_replay_multi.sh` |
| `train_AllTask_2015_none_text.sh` | `train_AllTask_twitter2015_none_multi.sh` |
| `train_AllTask_2015_ewc_text.sh` | `train_AllTask_twitter2015_ewc_multi.sh` |

## 新脚本特点

### 标准化命名
- 格式: `[环境]_[任务]_[数据集]_[策略]_[模式].sh`
- 环境: `strain_` (服务器) / `train_` (本地)
- 任务: `AllTask` (多任务)
- 数据集: `twitter2015`, `twitter2017`, `200`
- 策略: `none`, `ewc`, `replay`, `lwf`, `si`, `mas`, `mymethod`, `moe`
- 模式: `multi` (多模态)

### 统一配置
- 服务器版本: 模型统一命名为 `1.pt`，存储在 `/root/autodl-tmp/`
- 本地版本: 模型详细命名，存储在当前目录
- 支持标签嵌入功能
- 统一的参数配置和日志记录

## 使用方法

### 生成新脚本
```bash
# 生成单个脚本
python scripts/config_templates.py --env server --task_type AllTask --dataset twitter2015 --strategy ewc --mode multi

# 批量生成所有脚本
python scripts/generate_all_scripts.py
```

### 运行新脚本
```bash
# 服务器版本
./scripts/strain_AllTask_twitter2015_ewc_multi.sh

# 本地版本
./scripts/train_AllTask_200_none_multi.sh
```

## 恢复旧脚本
如果需要恢复旧脚本，可以从备份目录复制:
```bash
cp scripts/backup_old_scripts/旧脚本名.sh scripts/
chmod +x scripts/旧脚本名.sh
```

## 注意事项
1. 新脚本使用重构后的训练代码 (`modules/train_refactored.py`)
2. 旧脚本可能不兼容新的训练代码
3. 建议使用新脚本进行训练
4. 如需使用旧脚本，请确保训练代码兼容性
"""
    
    with open("scripts/MIGRATION_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("已生成迁移指南: scripts/MIGRATION_GUIDE.md")


def main():
    """主函数"""
    print("=== 脚本清理工具 ===")
    
    # 确认操作
    print("此操作将:")
    print("1. 备份所有旧脚本到 scripts/backup_old_scripts/")
    print("2. 删除旧脚本文件")
    print("3. 生成迁移指南")
    
    confirm = input("\n是否继续? (y/N): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        return
    
    # 备份旧脚本
    backup_dir, backed_up_files = backup_old_scripts()
    
    # 删除旧脚本
    removed_files = remove_old_scripts()
    
    # 创建迁移指南
    create_migration_guide()
    
    # 列出当前脚本
    current_scripts = list_current_scripts()
    
    print("\n=== 清理完成 ===")
    print(f"备份了 {len(backed_up_files)} 个旧脚本")
    print(f"删除了 {len(removed_files)} 个旧脚本")
    print(f"当前有 {len(current_scripts)} 个脚本")
    print("\n查看以下文件了解详情:")
    print("- scripts/MIGRATION_GUIDE.md - 迁移指南")
    print("- scripts/SCRIPT_INDEX.md - 脚本索引")
    print("- scripts/backup_old_scripts/ - 旧脚本备份")


if __name__ == "__main__":
    main() 