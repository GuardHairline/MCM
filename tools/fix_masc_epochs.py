#!/usr/bin/env python3
"""
修复所有配置文件中的MASC任务epochs设置
将第一个MASC任务的epochs从5增加到15
"""

import json
import os
from pathlib import Path

def fix_masc_epochs(config_file, min_epochs=15):
    """修复配置文件中的MASC epochs
    
    Args:
        config_file: 配置文件路径
        min_epochs: MASC任务的最小epochs数
    """
    print(f"\n处理文件: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    modified = False
    
    if 'tasks' in config:
        for i, task in enumerate(config['tasks']):
            if task.get('task_name') == 'masc':
                old_epochs = task.get('epochs', 5)
                if old_epochs < min_epochs:
                    task['epochs'] = min_epochs
                    print(f"  ✅ 任务 {i+1} (MASC): epochs {old_epochs} -> {min_epochs}")
                    modified = True
                else:
                    print(f"  ℹ️  任务 {i+1} (MASC): epochs {old_epochs} 已足够")
    
    if modified:
        # 备份原文件
        backup_file = config_file.replace('.json', '_backup.json')
        if not os.path.exists(backup_file):
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"  💾 已备份到: {backup_file}")
        
        # 保存修改后的文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  ✅ 已保存修改")
    else:
        print(f"  ℹ️  无需修改")
    
    return modified


def main():
    # 查找所有twitter2015的配置文件
    config_dir = Path("scripts/configs")
    config_files = list(config_dir.glob("*twitter2015*.json"))
    
    if not config_files:
        print("❌ 没有找到配置文件")
        return
    
    print(f"找到 {len(config_files)} 个配置文件")
    print("="*80)
    
    total_modified = 0
    for config_file in sorted(config_files):
        if '_backup' in str(config_file):
            continue
        if fix_masc_epochs(config_file, min_epochs=15):
            total_modified += 1
    
    print("\n" + "="*80)
    print(f"✅ 完成！共修改了 {total_modified} 个文件")
    print("\n📋 建议:")
    print("  1. 检查修改后的配置文件")
    print("  2. 重新运行训练脚本")
    print("  3. 监控MASC任务的NEG类recall")
    print("  4. 如果仍然失败，考虑将NEG权重从5.0增加到8.0")


if __name__ == "__main__":
    main()





