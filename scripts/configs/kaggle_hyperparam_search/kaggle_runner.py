#!/usr/bin/env python3
"""
Kaggle Notebook运行脚本 - MATE、MNER、MABSA超参数搜索

此脚本设计在Kaggle Notebook中运行，针对Kaggle环境优化：
1. 独享P100 GPU，无需等待
2. 自动设置项目路径
3. 安装依赖
4. 串行运行实验（避免显存问题）
5. 定期保存结果到 /kaggle/working
6. 支持断点续跑（考虑9-12小时限制）

使用方法:
1. 在Kaggle Notebook中运行此脚本
2. 设置加速器为 GPU P100
3. 添加MCM项目数据集
4. 运行全部或指定范围的实验

参数:
    --start_exp: 起始实验ID（默认1）
    --end_exp: 结束实验ID（默认5）
    --config_dir: 配置文件目录（默认从数据集读取）
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import shutil

# Kaggle环境配置
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"
CODE_DATASET = "mcm-code"      # 代码数据集名称
DATA_DATASET = "mcm-data"      # 数据数据集名称（可选，如果使用分离方案）
PROJECT_NAME = "mcm-project"   # 完整项目数据集名称（向后兼容）

# 颜色输出
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def print_info(msg):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def print_separator():
    print(f"{Colors.BLUE}================================================================================{Colors.NC}")

def setup_environment():
    """设置Kaggle环境"""
    global KAGGLE_INPUT, KAGGLE_WORKING  # 必须在函数开头声明
    
    print_separator()
    print_info("设置Kaggle环境...")
    print_separator()
    
    # 1. 检查是否在Kaggle环境
    if not os.path.exists(KAGGLE_INPUT):
        print_warning("未检测到Kaggle环境，使用本地路径")
        KAGGLE_INPUT = "."
        KAGGLE_WORKING = "./output"
        os.makedirs(KAGGLE_WORKING, exist_ok=True)
    
    # 2. 查找项目路径（支持两种模式）
    code_path = None
    data_path = None
    use_split_mode = False
    
    # 模式1: 分离模式（代码和数据分开）
    possible_code_paths = [
        Path(KAGGLE_INPUT) / CODE_DATASET,
        Path(KAGGLE_INPUT) / CODE_DATASET / "MCM",
    ]
    
    for path in possible_code_paths:
        if path.exists():
            code_path = path
            use_split_mode = True
            print_info(f"✓ 检测到分离模式 - 代码路径: {path}")
            break
    
    # 如果使用分离模式，查找数据路径
    if use_split_mode:
        possible_data_paths = [
            Path(KAGGLE_INPUT) / DATA_DATASET,
            Path(KAGGLE_INPUT) / DATA_DATASET / "data",
        ]
        
        for path in possible_data_paths:
            if path.exists():
                data_path = path
                print_info(f"✓ 检测到分离模式 - 数据路径: {path}")
                break
        
        if data_path is None:
            print_warning(f"未找到数据数据集 '{DATA_DATASET}'，将只使用代码数据集")
    
    # 模式2: 完整模式（向后兼容）
    if not use_split_mode:
        possible_full_paths = [
            Path(KAGGLE_INPUT) / PROJECT_NAME,
            Path(KAGGLE_INPUT) / PROJECT_NAME / "MCM",
            Path(KAGGLE_INPUT) / "MCM",
        ]
        
        for path in possible_full_paths:
            if path.exists():
                code_path = path
                print_info(f"✓ 检测到完整模式 - 项目路径: {path}")
                break
    
    if code_path is None:
        print_error("未找到项目路径，尝试列出可用数据集...")
        if os.path.exists(KAGGLE_INPUT):
            print(f"可用数据集: {os.listdir(KAGGLE_INPUT)}")
        raise FileNotFoundError("无法找到MCM项目，请检查数据集配置")
    
    # 3. 复制项目到根目录
    work_project_path = Path("/MCM")
    
    if not work_project_path.exists():
        print_info(f"复制代码到项目目录: {work_project_path}")
        shutil.copytree(code_path, work_project_path, dirs_exist_ok=True)
    else:
        print_info(f"项目目录已存在: {work_project_path}")
    
    # 4. 如果是分离模式，链接或复制数据目录
    if use_split_mode and data_path:
        target_data_dir = work_project_path / "data"
        target_model_dir = work_project_path / "downloaded_model"
        
        # 处理data目录
        if (data_path / "data").exists():
            source_data = data_path / "data"
        else:
            source_data = data_path
        
        if not target_data_dir.exists():
            print_info(f"链接数据目录: {source_data} -> {target_data_dir}")
            try:
                # 尝试创建符号链接（更快）
                os.symlink(source_data, target_data_dir)
            except (OSError, NotImplementedError):
                # 如果不支持符号链接，则复制
                print_warning("不支持符号链接，复制数据目录（可能较慢）...")
                shutil.copytree(source_data, target_data_dir, dirs_exist_ok=True)
        
        # 处理downloaded_model目录  
        source_model = data_path / "downloaded_model"
        if source_model.exists() and not target_model_dir.exists():
            print_info(f"链接模型目录: {source_model} -> {target_model_dir}")
            try:
                os.symlink(source_model, target_model_dir)
            except (OSError, NotImplementedError):
                print_warning("不支持符号链接，复制模型目录...")
                shutil.copytree(source_model, target_model_dir, dirs_exist_ok=True)
    
    # 5. 切换到项目目录 /MCM
    os.chdir(work_project_path)
    print_info(f"当前工作目录: {os.getcwd()}")
    
    # 6. 添加到Python路径
    sys.path.insert(0, str(work_project_path))
    print_info(f"已添加到Python路径: {work_project_path}")
    
    # 7. 安装依赖（Kaggle优化）
    # 优先使用Kaggle优化的requirements文件
    kaggle_req = work_project_path / "requirements_kaggle.txt"
    regular_req = work_project_path / "requirements.txt"
    
    if kaggle_req.exists():
        print_info("检测到Kaggle优化的依赖文件，使用 requirements_kaggle.txt")
        requirements_file = kaggle_req
    elif regular_req.exists():
        print_info("使用标准依赖文件 requirements.txt")
        print_warning("⚠️  可能会有版本冲突警告（通常可以忽略）")
        requirements_file = regular_req
    else:
        print_warning("未找到依赖文件，安装最小依赖集")
        # 安装必需的包
        minimal_packages = ["pytorch_crf", "sentencepiece", "protobuf==3.20.3"]
        for pkg in minimal_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
            except:
                pass
        requirements_file = None
    
    if requirements_file:
        print_info(f"安装依赖: {requirements_file.name}")
        try:
            # 使用--no-deps避免自动解析冲突
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
                check=False  # 不因警告而失败
            )
            print_info("✓ 依赖安装完成（忽略版本冲突警告）")
        except subprocess.CalledProcessError as e:
            print_warning(f"依赖安装有警告: {e}")
            print_warning("继续运行，Kaggle预装包通常可用")
    
    # 8. 创建输出目录
    output_dir = Path(KAGGLE_WORKING) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"输出目录: {output_dir}")
    
    # 9. 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_info(f"✓ GPU可用: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print_warning("未检测到GPU，将使用CPU（速度会很慢）")
    except ImportError:
        print_warning("PyTorch未安装，无法检查GPU")
    
    print_separator()
    return work_project_path

def load_configs(config_dir: Path):
    """加载配置文件"""
    index_file = config_dir / "config_index.json"
    
    if not index_file.exists():
        raise FileNotFoundError(f"配置索引文件不存在: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    return index_data["configs"]

def update_config_paths(config_file: Path, kaggle_working: str):
    """更新配置文件中的路径为Kaggle路径"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 需要更新的路径字段（包括所有可能包含输出路径的字段）
    path_keys = [
        "checkpoint_path", "save_path", "output_dir",
        "train_info_json", "output_model_path", "pretrained_model_path",
        "ewc_dir", "label_embedding_path", "label_emb_path"
    ]
    
    # 更新所有路径字段
    def update_path(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in path_keys and isinstance(value, str):
                    # 将checkpoints路径替换为Kaggle工作目录
                    if "checkpoints" in value:
                        new_value = value.replace("checkpoints", f"{kaggle_working}/checkpoints")
                        obj[key] = new_value
                        print_info(f"  更新路径: {key}")
                        print_info(f"    从: {value}")
                        print_info(f"    到: {new_value}")
                elif isinstance(value, (dict, list)):
                    update_path(value)
        elif isinstance(obj, list):
            for item in obj:
                update_path(item)
    
    print_info("正在更新配置文件路径...")
    update_path(config)
    
    # 保存更新后的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config

def run_experiment(exp_id: int, config_file: Path, task: str, strategy: str, 
                   lr: float, step_size: int, gamma: float):
    """运行单个实验"""
    print_separator()
    print_info(f"运行实验 #{exp_id}")
    print_info(f"  任务: {task}")
    print_info(f"  策略: {strategy}")
    print_info(f"  超参数: lr={lr}, step_size={step_size}, gamma={gamma}")
    print_info(f"  配置文件: {config_file}")
    print_separator()
    
    # 更新配置文件路径
    config = update_config_paths(config_file, KAGGLE_WORKING)
    
    # 运行训练脚本
    start_time = time.time()
    
    try:
        # 使用subprocess运行训练脚本
        cmd = [
            sys.executable, "-m", "scripts.train_with_zero_shot",
            "--config", str(config_file),
            "--start_task", "0",
            "--end_task", "2"
        ]
        
        print_info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        elapsed_time = time.time() - start_time
        print_info(f"✓ 实验 #{exp_id} 完成 (耗时: {elapsed_time/60:.1f} 分钟)")
        
        # 验证输出文件
        output_dir = Path(KAGGLE_WORKING) / "checkpoints"
        if output_dir.exists():
            files = list(output_dir.glob("**/*"))
            files = [f for f in files if f.is_file()]
            print_info(f"  已保存 {len(files)} 个文件到 {output_dir}")
            
            # 显示最近修改的文件（可能是本次实验生成的）
            recent_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            if recent_files:
                print_info(f"  最近生成的文件:")
                for f in recent_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    rel_path = f.relative_to(output_dir)
                    print_info(f"    - {rel_path} ({size_mb:.2f} MB)")
        else:
            print_warning(f"  ⚠️  输出目录不存在: {output_dir}")
            print_warning(f"  文件可能被保存到了其他位置，请检查配置")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print_error(f"✗ 实验 #{exp_id} 失败 (耗时: {elapsed_time/60:.1f} 分钟)")
        print_error(f"错误信息: {e}")
        return False
    except Exception as e:
        print_error(f"✗ 实验 #{exp_id} 发生未知错误: {e}")
        return False

def save_progress(completed_experiments: list, output_file: Path):
    """保存进度"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "completed_experiments": completed_experiments,
            "total_completed": len(completed_experiments),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print_info(f"进度已保存: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaggle超参数搜索实验运行器")
    parser.add_argument("--start_exp", type=int, default=1, help="起始实验ID")
    parser.add_argument("--end_exp", type=int, default=5, help="结束实验ID")
    parser.add_argument("--config_dir", type=str, default=None, help="配置文件目录")
    
    args = parser.parse_args()
    
    # 设置环境
    project_path = setup_environment()
    
    # 确定配置文件目录
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = project_path / "scripts/configs/kaggle_hyperparam_search"
    
    if not config_dir.exists():
        print_error(f"配置目录不存在: {config_dir}")
        print_info("尝试查找配置文件...")
        # 尝试其他可能的路径
        alt_paths = [
            project_path / "configs/kaggle_hyperparam_search",
            Path(KAGGLE_INPUT) / PROJECT_NAME / "scripts/configs/kaggle_hyperparam_search",
        ]
        for alt_config_dir in alt_paths:
            if alt_config_dir.exists():
                config_dir = alt_config_dir
                print_info(f"找到配置目录: {config_dir}")
                break
        else:
            raise FileNotFoundError(f"无法找到配置目录")
    
    print_info(f"配置目录: {config_dir}")
    
    # 加载配置
    configs = load_configs(config_dir)
    print_info(f"加载了 {len(configs)} 个配置")
    
    # 准备运行实验
    completed_experiments = []
    failed_experiments = []
    
    progress_file = Path(KAGGLE_WORKING) / "experiment_progress.json"
    
    # 如果存在进度文件，加载之前的进度
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
            completed_experiments = progress_data.get("completed_experiments", [])
        print_info(f"从进度文件恢复，已完成 {len(completed_experiments)} 个实验")
    
    # 运行实验
    total_time_start = time.time()
    
    for i, config_info in enumerate(configs, 1):
        # 检查实验范围
        if i < args.start_exp or i > args.end_exp:
            continue
        
        # 检查是否已完成
        if i in completed_experiments:
            print_info(f"实验 #{i} 已完成，跳过")
            continue
        
        # 运行实验
        config_file = Path(config_info["file"])
        # 如果config_file是相对路径，转换为绝对路径
        if not config_file.is_absolute():
            config_file = project_path / config_file
        
        success = run_experiment(
            exp_id=i,
            config_file=config_file,
            task=config_info["task"],
            strategy=config_info["strategy"],
            lr=config_info["lr"],
            step_size=config_info["step_size"],
            gamma=config_info["gamma"]
        )
        
        if success:
            completed_experiments.append(i)
        else:
            failed_experiments.append(i)
        
        # 保存进度
        save_progress(completed_experiments, progress_file)
        
        # 显示总体进度
        elapsed = time.time() - total_time_start
        remaining = args.end_exp - i
        avg_time_per_exp = elapsed / len(completed_experiments) if completed_experiments else 0
        estimated_remaining = remaining * avg_time_per_exp if avg_time_per_exp > 0 else 0
        
        print_info(f"总体进度: {len(completed_experiments)}/{args.end_exp - args.start_exp + 1} 完成")
        print_info(f"已用时: {elapsed/3600:.1f} 小时")
        if estimated_remaining > 0:
            print_info(f"预计剩余: {estimated_remaining/3600:.1f} 小时")
    
    # 显示最终结果
    print_separator()
    print_info("实验完成！")
    print_info(f"成功: {len(completed_experiments)} 个")
    print_info(f"失败: {len(failed_experiments)} 个")
    print_info(f"总耗时: {(time.time() - total_time_start)/3600:.1f} 小时")
    
    if failed_experiments:
        print_warning(f"失败的实验ID: {failed_experiments}")
    
    print_info(f"结果保存在: {KAGGLE_WORKING}/checkpoints")
    print_separator()
    
    # 自动打包结果
    print_separator()
    print_info("正在检查并打包实验结果...")
    output_dir = Path(KAGGLE_WORKING) / "checkpoints"
    
    # 详细检查输出目录
    print_info(f"检查输出目录: {output_dir}")
    
    if not output_dir.exists():
        print_error(f"输出目录不存在: {output_dir}")
        print_info("尝试检查其他可能的位置...")
        
        # 检查项目目录下的checkpoints
        project_checkpoints = Path("/MCM/checkpoints")
        if project_checkpoints.exists():
            files = list(project_checkpoints.glob("**/*"))
            files = [f for f in files if f.is_file()]
            print_warning(f"发现文件被保存到了项目目录: {project_checkpoints}")
            print_warning(f"  共 {len(files)} 个文件")
            if files:
                print_info("  文件列表:")
                for f in files[:10]:  # 显示前10个
                    print_info(f"    - {f.relative_to(project_checkpoints)}")
                print_error("❌ 路径配置有问题！文件应该保存到 /kaggle/working/checkpoints")
                print_error("   但实际保存到了 /MCM/checkpoints")
    else:
        files = list(output_dir.glob("**/*"))
        files = [f for f in files if f.is_file()]
        print_info(f"✓ 输出目录存在，共 {len(files)} 个文件")
        
        if files:
            print_info("文件列表:")
            for f in files[:20]:  # 显示前20个
                size_mb = f.stat().st_size / (1024 * 1024)
                rel_path = f.relative_to(output_dir)
                print_info(f"  - {rel_path} ({size_mb:.2f} MB)")
            if len(files) > 20:
                print_info(f"  ... 还有 {len(files) - 20} 个文件")
    
    if output_dir.exists() and any(output_dir.iterdir()):
        try:
            print_info("开始打包...")
            archive_path = Path(KAGGLE_WORKING) / "experiment_results"
            shutil.make_archive(str(archive_path), 'zip', output_dir)
            
            archive_file = Path(f"{archive_path}.zip")
            if archive_file.exists():
                size_mb = archive_file.stat().st_size / (1024 * 1024)
                print_info(f"✓ 结果已打包: {archive_file}")
                print_info(f"  文件大小: {size_mb:.1f} MB")
                print_info(f"  请在右侧 'Output' 标签页下载 experiment_results.zip")
            else:
                print_error("打包失败：未生成zip文件")
        except Exception as e:
            print_error(f"打包失败: {e}")
    else:
        print_warning("输出目录为空，没有结果需要打包")
    
    print_separator()
    print_info("=" * 80)
    print_info("🎉 所有任务已完成！")
    print_info("=" * 80)
    print_info("")
    print_info("📦 结果已打包，请下载 experiment_results.zip")
    print_info("")
    print_warning("⚠️  为节省GPU配额，请立即执行以下操作：")
    print_warning("   1. 在右侧 'Output' 标签下载 experiment_results.zip")
    print_warning("   2. 点击右上角 'Stop Session' 按钮停止Notebook")
    print_warning("   3. 或者等待此脚本自动退出后手动停止")
    print_info("")
    print_separator()
    
    # 等待几秒让用户看到消息
    print_info("等待10秒后自动退出...")
    for i in range(10, 0, -1):
        print(f"  {i}...", end='\r')
        time.sleep(1)
    
    print_info("✓ 脚本执行完毕，请手动停止Session以释放GPU资源")
    print_separator()

if __name__ == "__main__":
    main()
