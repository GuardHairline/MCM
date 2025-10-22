#!/usr/bin/env python3
"""
修复Shell脚本的换行符问题

将Windows格式(CRLF \r\n)转换为Unix格式(LF \n)
"""

import os
import sys
from pathlib import Path


def fix_line_endings(file_path: Path):
    """修复单个文件的换行符"""
    try:
        # 读取文件内容（二进制模式）
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # 检查是否包含\r\n
        if b'\r\n' in content:
            # 替换\r\n为\n
            content = content.replace(b'\r\n', b'\n')
            
            # 写回文件
            with open(file_path, 'wb') as f:
                f.write(content)
            
            print(f"✓ 已修复: {file_path}")
            return True
        else:
            print(f"  跳过（已是Unix格式）: {file_path}")
            return False
    except Exception as e:
        print(f"❌ 修复失败: {file_path} - {e}")
        return False


def main():
    if len(sys.argv) > 1:
        # 指定了文件或目录
        target = Path(sys.argv[1])
    else:
        # 默认修复超参数搜索目录
        target = Path("scripts/configs/hyperparam_search")
    
    print("=" * 70)
    print("换行符修复工具")
    print("=" * 70)
    print(f"目标: {target}")
    print()
    
    if not target.exists():
        print(f"❌ 目标不存在: {target}")
        sys.exit(1)
    
    fixed_count = 0
    
    if target.is_file():
        # 单个文件
        if fix_line_endings(target):
            fixed_count += 1
    else:
        # 目录，修复所有.sh文件
        sh_files = list(target.glob("*.sh"))
        if not sh_files:
            print("⚠️  未找到.sh文件")
        else:
            print(f"找到 {len(sh_files)} 个Shell脚本\n")
            for sh_file in sh_files:
                if fix_line_endings(sh_file):
                    fixed_count += 1
                    
                # 设置可执行权限
                try:
                    sh_file.chmod(0o755)
                except:
                    pass
    
    print()
    print("=" * 70)
    print(f"完成！修复了 {fixed_count} 个文件")
    print("=" * 70)
    
    # 验证修复
    if fixed_count > 0:
        print("\n验证方法:")
        print("  file <script>.sh")
        print("  # 应该显示: ... ASCII text")
        print("  # 而不是: ... ASCII text, with CRLF line terminators")


if __name__ == "__main__":
    main()


