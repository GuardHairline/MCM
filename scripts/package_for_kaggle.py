#!/usr/bin/env python3
"""
打包项目为 mcm-code.zip

用途：
- 将项目打包成zip文件，用于上传到Kaggle
- 只包含必要的代码和配置文件
- 排除缓存、检查点、日志等不必要文件

使用方法：
    python scripts/package_for_kaggle.py
    或
    python scripts/package_for_kaggle.py --output my-custom-name.zip
"""

import zipfile
import os
from pathlib import Path
import argparse
import sys


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent


def should_exclude(path, exclude_patterns):
    """
    判断路径是否应该被排除
    
    Args:
        path: Path对象
        exclude_patterns: 排除模式列表
    
    Returns:
        bool: True表示应该排除
    """
    path_str = str(path)
    path_parts = path.parts
    
    for pattern in exclude_patterns:
        # 检查路径中是否包含排除模式
        if pattern in path_parts:
            return True
        # 检查文件名是否匹配排除模式
        if path.name == pattern:
            return True
        # 检查文件扩展名
        if pattern.startswith('*.') and path.name.endswith(pattern[1:]):
            return True
    
    return False


def create_package(output_path, verbose=False):
    """
    创建项目压缩包
    
    Args:
        output_path: 输出的zip文件路径
        verbose: 是否显示详细信息
    """
    project_root = get_project_root()
    
    # 要包含的顶级文件夹和文件
    include_items = [
        'continual',
        'datasets',
        'models',
        'modules',
        'scripts',
        'tools',
        'utils',
        'visualize',
        'requirements_kaggle.txt',
        'requirements.txt',
    ]
    
    # 要排除的模式
    exclude_patterns = [
        '__pycache__',
        '.pytest_cache',
        '.git',
        '.gitignore',
        '.vscode',
        '.idea',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.DS_Store',
        'Thumbs.db',
        # 排除检查点和输出
        'checkpoints',
        'outputs',
        'logs',
        'runs',
        'wandb',
        # 排除数据文件
        'data',
        'datasets/twitter2015',
        'datasets/twitter2017',
        'datasets/masad',
        '*.pt',
        '*.pth',
        '*.bin',
        '*.ckpt',
        # 排除文档（可选）
        'doc',
        'docs',
        '*.md',
        # 排除临时文件
        'temp',
        'tmp',
        '*.tmp',
        '*.swp',
        '*.swo',
        # 排除测试（可选，如果想包含测试可以注释掉）
        # 'tests',
    ]
    
    print(f"📦 开始打包项目...")
    print(f"项目根目录: {project_root}")
    print(f"输出文件: {output_path}")
    print()
    
    # 统计信息
    file_count = 0
    total_size = 0
    skipped_count = 0
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item_name in include_items:
            item_path = project_root / item_name
            
            if not item_path.exists():
                print(f"⚠️  跳过不存在的项: {item_name}")
                continue
            
            print(f"📁 处理: {item_name}")
            
            if item_path.is_file():
                # 单个文件
                if not should_exclude(item_path, exclude_patterns):
                    arcname = item_path.relative_to(project_root)
                    zipf.write(item_path, arcname)
                    file_count += 1
                    total_size += item_path.stat().st_size
                    if verbose:
                        print(f"  ✅ {arcname}")
                else:
                    skipped_count += 1
                    if verbose:
                        print(f"  ⏭️  跳过: {item_name}")
            
            elif item_path.is_dir():
                # 目录
                dir_file_count = 0
                for file_path in item_path.rglob('*'):
                    if file_path.is_file():
                        if not should_exclude(file_path, exclude_patterns):
                            arcname = file_path.relative_to(project_root)
                            zipf.write(file_path, arcname)
                            file_count += 1
                            dir_file_count += 1
                            total_size += file_path.stat().st_size
                            if verbose:
                                print(f"  ✅ {arcname}")
                        else:
                            skipped_count += 1
                            if verbose:
                                rel_path = file_path.relative_to(project_root)
                                print(f"  ⏭️  跳过: {rel_path}")
                
                print(f"  ✅ 添加了 {dir_file_count} 个文件")
    
    # 获取压缩包大小
    zip_size = os.path.getsize(output_path)
    
    print()
    print("="*60)
    print("✅ 打包完成!")
    print("="*60)
    print(f"总文件数: {file_count}")
    print(f"跳过文件数: {skipped_count}")
    print(f"原始大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"压缩后大小: {zip_size / 1024 / 1024:.2f} MB")
    print(f"压缩率: {(1 - zip_size / total_size) * 100:.1f}%")
    print(f"输出文件: {output_path}")
    print()
    print("📤 现在可以将此文件上传到Kaggle了!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='打包MCM项目为zip文件，用于Kaggle部署',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认名称 mcm-code.zip
  python scripts/package_for_kaggle.py
  
  # 指定自定义输出名称
  python scripts/package_for_kaggle.py --output my-project.zip
  
  # 显示详细信息
  python scripts/package_for_kaggle.py --verbose
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='mcm-code.zip',
        help='输出zip文件名 (默认: mcm-code.zip)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细的文件列表'
    )
    
    args = parser.parse_args()
    
    # 确保输出路径是绝对路径
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = get_project_root() / output_path
    
    # 如果文件已存在，询问是否覆盖
    if output_path.exists():
        response = input(f"⚠️  文件 {output_path} 已存在，是否覆盖? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ 取消操作")
            sys.exit(0)
        output_path.unlink()
    
    try:
        create_package(output_path, verbose=args.verbose)
    except Exception as e:
        print(f"❌ 打包失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

