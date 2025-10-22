#!/usr/bin/env python3
"""
快速修复已生成的Shell脚本中的路径分隔符问题

将Windows风格路径 (使用反斜杠 \) 转换为Unix风格 (使用正斜杠 /)
"""

import re
from pathlib import Path


def fix_paths_in_script(script_path: Path):
    """修复Shell脚本中的Windows路径"""
    try:
        # 读取脚本内容
        with open(script_path, 'r', encoding='utf-8', newline='\n') as f:
            content = f.read()
        
        # 检测是否有Windows风格的路径
        windows_path_pattern = r'scripts\\configs\\hyperparam_search\\'
        
        if re.search(windows_path_pattern, content):
            # 替换所有反斜杠路径为正斜杠
            # scripts\configs\hyperparam_search\ -> scripts/configs/hyperparam_search/
            content = content.replace('scripts\\configs\\hyperparam_search\\', 
                                     'scripts/configs/hyperparam_search/')
            content = content.replace('scripts\\configs\\hyperparam_search', 
                                     'scripts/configs/hyperparam_search')
            
            # 写回文件
            with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            
            print(f"✓ 已修复路径: {script_path.name}")
            return True
        else:
            print(f"  无需修复: {script_path.name}")
            return False
            
    except Exception as e:
        print(f"❌ 修复失败: {script_path.name} - {e}")
        return False


def main():
    print("=" * 70)
    print("路径分隔符快速修复工具")
    print("=" * 70)
    print()
    
    # 修复目标目录
    target_dir = Path("scripts/configs/hyperparam_search")
    
    if not target_dir.exists():
        print(f"❌ 目录不存在: {target_dir}")
        return
    
    # 查找所有Shell脚本
    sh_files = list(target_dir.glob("*.sh"))
    
    if not sh_files:
        print("⚠️  未找到Shell脚本")
        return
    
    print(f"找到 {len(sh_files)} 个Shell脚本\n")
    
    fixed_count = 0
    for sh_file in sh_files:
        if fix_paths_in_script(sh_file):
            fixed_count += 1
    
    print()
    print("=" * 70)
    print(f"完成！修复了 {fixed_count} 个文件")
    print("=" * 70)
    
    if fixed_count > 0:
        print("\n现在可以重新运行实验了：")
        print("  bash scripts/configs/hyperparam_search/start_experiments_detached.sh")


if __name__ == "__main__":
    main()


