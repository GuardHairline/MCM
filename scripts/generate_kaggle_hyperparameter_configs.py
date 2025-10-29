#!/usr/bin/env python3
"""
生成MATE、MNER、MABSA超参数搜索配置文件 - Kaggle版本

针对Kaggle环境的特殊优化:
1. 输出目录: /kaggle/working (Kaggle可写目录)
2. 数据集路径: /kaggle/input/dataset-name/ (Kaggle数据集挂载点)
3. GPU策略: 独享P100，不需要等待，支持串行运行
4. 时间限制: 考虑Kaggle 9-12小时运行限制
5. 定期保存: 每个实验完成后立即保存结果
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_task_config import TaskConfigGenerator


class KaggleTaskHyperparameterSearchGenerator:
    """Kaggle环境下的多任务超参数搜索配置生成器"""
    
    def __init__(self):
        self.base_generator = TaskConfigGenerator()
        
        # 定义超参数搜索空间
        self.hyperparameter_grid = {
            "lr": [5e-5, 1e-5, 5e-6],
            "step_size": [5, 10, 15],
            "gamma": [0.3, 0.5, 0.7]
        }
        
        # 任务列表
        self.tasks = ["mate", "mner", "mabsa"]
        
        # 策略列表
        self.strategies = ["none"]
        
    def get_hyperparameter_combinations(self) -> List[Tuple[float, int, float]]:
        """生成合理的超参数组合"""
        combinations = []
        
        # 策略1: 固定gamma，变化lr和step_size
        for lr in self.hyperparameter_grid["lr"]:
            for step_size in self.hyperparameter_grid["step_size"]:
                combinations.append((lr, step_size, 0.5))
        
        # 策略2: 固定lr和step_size，变化gamma
        for gamma in self.hyperparameter_grid["gamma"]:
            if gamma != 0.5:
                combinations.append((1e-5, 10, gamma))
        
        # 策略3: 特殊组合
        special_combinations = [
            (5e-5, 5, 0.7),
            (5e-6, 15, 0.3),
            (1e-5, 10, 0.5),
        ]
        
        for combo in special_combinations:
            if combo not in combinations:
                combinations.append(combo)
        
        return combinations
    
    def generate_single_config(self, 
                              env: str,
                              dataset: str,
                              task_name: str,
                              strategy: str,
                              lr: float,
                              step_size: int,
                              gamma: float,
                              seq_suffix: str = "",
                              kaggle_output_path: str = "/kaggle/working") -> dict:
        """生成单个配置文件 - Kaggle优化版本
        
        注意：
        - 数据集路径使用项目内相对路径（如 data/twitter2015_images/）
        - 项目会被复制到 /kaggle/working/MCM，数据集会随之移动
        - 只需要修改checkpoint输出路径到 /kaggle/working/checkpoints
        """
        
        task_sequence = [task_name, task_name]
        mode_sequence = ["text_only", "multimodal"]
        
        # 使用基础生成器生成配置
        config = self.base_generator.generate_task_sequence_config(
            env=env,
            dataset=dataset,
            task_sequence=task_sequence,
            mode_sequence=mode_sequence,
            strategy=strategy,
            use_label_embedding=False,
            seq_suffix=seq_suffix,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            epochs=20,
            patience=999
        )
        
        # Kaggle特殊配置 - 只修改输出路径，数据路径保持相对路径
        config["kaggle_mode"] = True
        config["kaggle_output_path"] = kaggle_output_path
        
        # 添加超参数信息
        config["hyperparameters"] = {
            "lr": lr,
            "step_size": step_size,
            "gamma": gamma
        }
        
        return config
    
    def generate_all_configs(self, 
                            env: str = "server",
                            dataset: str = "twitter2015",
                            output_dir: str = "scripts/configs/kaggle_hyperparam_search",
                            kaggle_output_path: str = "/kaggle/working"):
        """生成所有配置文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hyperparams = self.get_hyperparameter_combinations()
        
        print(f"生成Kaggle配置文件到: {output_path}")
        print(f"任务列表: {self.tasks}")
        print(f"策略: {self.strategies}")
        print(f"超参数组合数量: {len(hyperparams)}")
        print(f"总配置文件数: {len(self.tasks) * len(self.strategies) * len(hyperparams)}")
        print(f"Kaggle输出路径: {kaggle_output_path}")
        print()
        
        configs_generated = []
        
        for task_name in self.tasks:
            print(f"\n{'='*60}")
            print(f"生成任务: {task_name.upper()}")
            print(f"{'='*60}\n")
            
            for strategy in self.strategies:
                for i, (lr, step_size, gamma) in enumerate(hyperparams):
                    lr_str = f"{lr:.0e}".replace("-", "").replace("+", "")
                    config_name = f"kaggle_{dataset}_{task_name}_{strategy}_lr{lr_str}_ss{step_size}_g{gamma:.1f}.json"
                    config_file = output_path / config_name
                    
                    seq_suffix = f"hp{i+1}"
                    
                    config = self.generate_single_config(
                        env=env,
                        dataset=dataset,
                        task_name=task_name,
                        strategy=strategy,
                        lr=lr,
                        step_size=step_size,
                        gamma=gamma,
                        seq_suffix=seq_suffix,
                        kaggle_output_path=kaggle_output_path
                    )
                    
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    
                    configs_generated.append({
                        "file": config_file.as_posix(),
                        "task": task_name,
                        "strategy": strategy,
                        "lr": lr,
                        "step_size": step_size,
                        "gamma": gamma
                    })
                    
                    print(f"✓ {config_name}")
        
        # 生成索引文件
        index_file = output_path / "config_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_configs": len(configs_generated),
                "tasks": self.tasks,
                "strategies": self.strategies,
                "hyperparameter_grid": self.hyperparameter_grid,
                "kaggle_optimized": True,
                "configs": configs_generated
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n索引文件已生成: {index_file}")
        print(f"\n总共生成 {len(configs_generated)} 个配置文件")
        
        return configs_generated


def main():
    parser = argparse.ArgumentParser(description="生成Kaggle环境的超参数搜索配置")
    parser.add_argument("--env", type=str, default="server",
                       choices=["local", "server"],
                       help="环境类型")
    parser.add_argument("--dataset", type=str, default="twitter2015",
                       choices=["twitter2015", "twitter2017", "mix"],
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, 
                       default="scripts/configs/kaggle_hyperparam_search",
                       help="输出目录")
    parser.add_argument("--kaggle_dataset_name", type=str,
                       default="mcm-project",
                       help="Kaggle数据集名称")
    parser.add_argument("--max_experiments_per_session", type=int,
                       default=5,
                       help="每个Kaggle会话最多运行的实验数（考虑时间限制）")
    
    args = parser.parse_args()
    
    # Kaggle输出路径
    kaggle_output_path = "/kaggle/working"
    
    # 生成配置
    generator = KaggleTaskHyperparameterSearchGenerator()
    configs = generator.generate_all_configs(
        env=args.env,
        dataset=args.dataset,
        output_dir=args.output_dir,
        kaggle_output_path=kaggle_output_path
    )
    
    # 生成Kaggle Notebook脚本
    notebook_path = Path(args.output_dir) / "kaggle_runner.py"
    _generate_kaggle_runner(notebook_path, configs, args.output_dir, args.max_experiments_per_session)
    
    # 生成部署说明
    readme_path = Path(args.output_dir) / "KAGGLE_DEPLOYMENT.md"
    _generate_deployment_guide(readme_path, args.kaggle_dataset_name, len(configs), args.max_experiments_per_session)
    
    # 生成项目准备脚本
    prep_script_path = Path(args.output_dir) / "prepare_for_kaggle.sh"
    _generate_preparation_script(prep_script_path)
    
    # 生成结果分析脚本（Kaggle版）
    analysis_script_path = Path(args.output_dir) / "analyze_kaggle_results.py"
    _generate_kaggle_analysis_script(analysis_script_path, configs)
    
    print(f"\n✓ Kaggle运行脚本已生成: {notebook_path}")
    print(f"✓ 部署说明已生成: {readme_path}")
    print(f"✓ 准备脚本已生成: {prep_script_path}")
    print(f"✓ 结果分析脚本已生成: {analysis_script_path}")
    print(f"\n请查看 {readme_path} 了解详细部署步骤")


def _generate_kaggle_runner(script_path: Path, configs: list, output_dir: str, max_experiments: int):
    """生成Kaggle Notebook运行脚本"""
    
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'''#!/usr/bin/env python3
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
    --end_exp: 结束实验ID（默认{max_experiments}）
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
    GREEN = '\\033[0;32m'
    YELLOW = '\\033[1;33m'
    RED = '\\033[0;31m'
    BLUE = '\\033[0;34m'
    CYAN = '\\033[0;36m'
    NC = '\\033[0m'

def print_info(msg):
    print(f"{{Colors.GREEN}}[INFO]{{Colors.NC}} {{msg}}")

def print_warning(msg):
    print(f"{{Colors.YELLOW}}[WARNING]{{Colors.NC}} {{msg}}")

def print_error(msg):
    print(f"{{Colors.RED}}[ERROR]{{Colors.NC}} {{msg}}")

def print_separator():
    print(f"{{Colors.BLUE}}{'='*80}{{Colors.NC}}")

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
            print_info(f"✓ 检测到分离模式 - 代码路径: {{path}}")
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
                print_info(f"✓ 检测到分离模式 - 数据路径: {{path}}")
                break
        
        if data_path is None:
            print_warning(f"未找到数据数据集 '{{DATA_DATASET}}'，将只使用代码数据集")
    
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
                print_info(f"✓ 检测到完整模式 - 项目路径: {{path}}")
                break
    
    if code_path is None:
        print_error("未找到项目路径，尝试列出可用数据集...")
        if os.path.exists(KAGGLE_INPUT):
            print(f"可用数据集: {{os.listdir(KAGGLE_INPUT)}}")
        raise FileNotFoundError("无法找到MCM项目，请检查数据集配置")
    
    # 3. 复制项目到根目录
    work_project_path = Path("/MCM")
    
    if not work_project_path.exists():
        print_info(f"复制代码到项目目录: {{work_project_path}}")
        shutil.copytree(code_path, work_project_path, dirs_exist_ok=True)
    else:
        print_info(f"项目目录已存在: {{work_project_path}}")
    
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
            print_info(f"链接数据目录: {{source_data}} -> {{target_data_dir}}")
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
            print_info(f"链接模型目录: {{source_model}} -> {{target_model_dir}}")
            try:
                os.symlink(source_model, target_model_dir)
            except (OSError, NotImplementedError):
                print_warning("不支持符号链接，复制模型目录...")
                shutil.copytree(source_model, target_model_dir, dirs_exist_ok=True)
    
    # 5. 切换到项目目录 /MCM
    os.chdir(work_project_path)
    print_info(f"当前工作目录: {{os.getcwd()}}")
    
    # 6. 添加到Python路径
    sys.path.insert(0, str(work_project_path))
    print_info(f"已添加到Python路径: {{work_project_path}}")
    
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
        print_info(f"安装依赖: {{requirements_file.name}}")
        try:
            # 使用--no-deps避免自动解析冲突
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
                check=False  # 不因警告而失败
            )
            print_info("✓ 依赖安装完成（忽略版本冲突警告）")
        except subprocess.CalledProcessError as e:
            print_warning(f"依赖安装有警告: {{e}}")
            print_warning("继续运行，Kaggle预装包通常可用")
    
    # 8. 创建输出目录
    output_dir = Path(KAGGLE_WORKING) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"输出目录: {{output_dir}}")
    
    # 9. 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_info(f"✓ GPU可用: {{gpu_name}} ({{gpu_memory:.1f}} GB)")
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
        raise FileNotFoundError(f"配置索引文件不存在: {{index_file}}")
    
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
                        new_value = value.replace("checkpoints", f"{{kaggle_working}}/checkpoints")
                        obj[key] = new_value
                        print_info(f"  更新路径: {{key}}")
                        print_info(f"    从: {{value}}")
                        print_info(f"    到: {{new_value}}")
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
    print_info(f"运行实验 #{{exp_id}}")
    print_info(f"  任务: {{task}}")
    print_info(f"  策略: {{strategy}}")
    print_info(f"  超参数: lr={{lr}}, step_size={{step_size}}, gamma={{gamma}}")
    print_info(f"  配置文件: {{config_file}}")
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
        
        print_info(f"执行命令: {{' '.join(cmd)}}")
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        elapsed_time = time.time() - start_time
        print_info(f"✓ 实验 #{{exp_id}} 完成 (耗时: {{elapsed_time/60:.1f}} 分钟)")
        
        # 验证输出文件
        output_dir = Path(KAGGLE_WORKING) / "checkpoints"
        if output_dir.exists():
            files = list(output_dir.glob("**/*"))
            files = [f for f in files if f.is_file()]
            print_info(f"  已保存 {{len(files)}} 个文件到 {{output_dir}}")
            
            # 显示最近修改的文件（可能是本次实验生成的）
            recent_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            if recent_files:
                print_info(f"  最近生成的文件:")
                for f in recent_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    rel_path = f.relative_to(output_dir)
                    print_info(f"    - {{rel_path}} ({{size_mb:.2f}} MB)")
        else:
            print_warning(f"  ⚠️  输出目录不存在: {{output_dir}}")
            print_warning(f"  文件可能被保存到了其他位置，请检查配置")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print_error(f"✗ 实验 #{{exp_id}} 失败 (耗时: {{elapsed_time/60:.1f}} 分钟)")
        print_error(f"错误信息: {{e}}")
        return False
    except Exception as e:
        print_error(f"✗ 实验 #{{exp_id}} 发生未知错误: {{e}}")
        return False

def save_progress(completed_experiments: list, output_file: Path):
    """保存进度"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({{
            "completed_experiments": completed_experiments,
            "total_completed": len(completed_experiments),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }}, f, indent=2)
    
    print_info(f"进度已保存: {{output_file}}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaggle超参数搜索实验运行器")
    parser.add_argument("--start_exp", type=int, default=1, help="起始实验ID")
    parser.add_argument("--end_exp", type=int, default={max_experiments}, help="结束实验ID")
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
        print_error(f"配置目录不存在: {{config_dir}}")
        print_info("尝试查找配置文件...")
        # 尝试其他可能的路径
        alt_paths = [
            project_path / "configs/kaggle_hyperparam_search",
            Path(KAGGLE_INPUT) / PROJECT_NAME / "scripts/configs/kaggle_hyperparam_search",
        ]
        for alt_config_dir in alt_paths:
            if alt_config_dir.exists():
                config_dir = alt_config_dir
                print_info(f"找到配置目录: {{config_dir}}")
                break
        else:
            raise FileNotFoundError(f"无法找到配置目录")
    
    print_info(f"配置目录: {{config_dir}}")
    
    # 加载配置
    configs = load_configs(config_dir)
    print_info(f"加载了 {{len(configs)}} 个配置")
    
    # 准备运行实验
    completed_experiments = []
    failed_experiments = []
    
    progress_file = Path(KAGGLE_WORKING) / "experiment_progress.json"
    
    # 如果存在进度文件，加载之前的进度
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
            completed_experiments = progress_data.get("completed_experiments", [])
        print_info(f"从进度文件恢复，已完成 {{len(completed_experiments)}} 个实验")
    
    # 运行实验
    total_time_start = time.time()
    
    for i, config_info in enumerate(configs, 1):
        # 检查实验范围
        if i < args.start_exp or i > args.end_exp:
            continue
        
        # 检查是否已完成
        if i in completed_experiments:
            print_info(f"实验 #{{i}} 已完成，跳过")
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
        
        print_info(f"总体进度: {{len(completed_experiments)}}/{{args.end_exp - args.start_exp + 1}} 完成")
        print_info(f"已用时: {{elapsed/3600:.1f}} 小时")
        if estimated_remaining > 0:
            print_info(f"预计剩余: {{estimated_remaining/3600:.1f}} 小时")
    
    # 显示最终结果
    print_separator()
    print_info("实验完成！")
    print_info(f"成功: {{len(completed_experiments)}} 个")
    print_info(f"失败: {{len(failed_experiments)}} 个")
    print_info(f"总耗时: {{(time.time() - total_time_start)/3600:.1f}} 小时")
    
    if failed_experiments:
        print_warning(f"失败的实验ID: {{failed_experiments}}")
    
    print_info(f"结果保存在: {{KAGGLE_WORKING}}/checkpoints")
    print_separator()
    
    # 自动打包结果
    print_separator()
    print_info("正在检查并打包实验结果...")
    output_dir = Path(KAGGLE_WORKING) / "checkpoints"
    
    # 详细检查输出目录
    print_info(f"检查输出目录: {{output_dir}}")
    
    if not output_dir.exists():
        print_error(f"输出目录不存在: {{output_dir}}")
        print_info("尝试检查其他可能的位置...")
        
        # 检查项目目录下的checkpoints
        project_checkpoints = Path("/MCM/checkpoints")
        if project_checkpoints.exists():
            files = list(project_checkpoints.glob("**/*"))
            files = [f for f in files if f.is_file()]
            print_warning(f"发现文件被保存到了项目目录: {{project_checkpoints}}")
            print_warning(f"  共 {{len(files)}} 个文件")
            if files:
                print_info("  文件列表:")
                for f in files[:10]:  # 显示前10个
                    print_info(f"    - {{f.relative_to(project_checkpoints)}}")
                print_error("❌ 路径配置有问题！文件应该保存到 /kaggle/working/checkpoints")
                print_error("   但实际保存到了 /MCM/checkpoints")
    else:
        files = list(output_dir.glob("**/*"))
        files = [f for f in files if f.is_file()]
        print_info(f"✓ 输出目录存在，共 {{len(files)}} 个文件")
        
        if files:
            print_info("文件列表:")
            for f in files[:20]:  # 显示前20个
                size_mb = f.stat().st_size / (1024 * 1024)
                rel_path = f.relative_to(output_dir)
                print_info(f"  - {{rel_path}} ({{size_mb:.2f}} MB)")
            if len(files) > 20:
                print_info(f"  ... 还有 {{len(files) - 20}} 个文件")
    
    if output_dir.exists() and any(output_dir.iterdir()):
        try:
            print_info("开始打包...")
            archive_path = Path(KAGGLE_WORKING) / "experiment_results"
            shutil.make_archive(str(archive_path), 'zip', output_dir)
            
            archive_file = Path(f"{{archive_path}}.zip")
            if archive_file.exists():
                size_mb = archive_file.stat().st_size / (1024 * 1024)
                print_info(f"✓ 结果已打包: {{archive_file}}")
                print_info(f"  文件大小: {{size_mb:.1f}} MB")
                print_info(f"  请在右侧 'Output' 标签页下载 experiment_results.zip")
            else:
                print_error("打包失败：未生成zip文件")
        except Exception as e:
            print_error(f"打包失败: {{e}}")
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
        print(f"  {{i}}...", end='\\r')
        time.sleep(1)
    
    print_info("✓ 脚本执行完毕，请手动停止Session以释放GPU资源")
    print_separator()

if __name__ == "__main__":
    main()
''')
    
    script_path.chmod(0o755)


def _generate_deployment_guide(readme_path: Path, dataset_name: str, total_configs: int, max_per_session: int):
    """生成Kaggle部署指南"""
    
    with open(readme_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'''# Kaggle部署指南 - MCM项目超参数搜索

本指南详细说明如何在Kaggle上运行MCM项目的超参数搜索实验。

## 📋 目录

1. [前期准备](#前期准备)
2. [项目打包上传](#项目打包上传)
3. [创建Kaggle Notebook](#创建kaggle-notebook)
4. [运行实验](#运行实验)
5. [结果下载](#结果下载)
6. [注意事项](#注意事项)
7. [故障排查](#故障排查)

---

## 🔧 前期准备

### 1. 检查项目结构

确保你的项目包含以下关键文件：
- `requirements.txt` - Python依赖
- `scripts/train_with_zero_shot.py` - 训练脚本
- `scripts/configs/kaggle_hyperparam_search/` - 配置文件
- 所有必要的代码文件（`models/`, `datasets/`, `continual/` 等）

### 2. 准备数据集

确保以下数据在项目中：
- `data/twitter2015_images/` - Twitter2015图片数据
- `data/MNER/` - MNER数据集
- `data/MNRE/` - MNRE数据集  
- `data/MASC/` - MASC数据集
- `data/MABSA/` - MABSA数据集（如果有）
- `downloaded_model/` - 预训练模型（DeBERTa, ViT等）

### 3. 清理不必要文件

为了减小上传大小，删除以下内容：
```bash
bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
```

这会自动：
- 删除 `__pycache__/` 和 `.pyc` 文件
- 删除已有的 `checkpoints/` （结果会在Kaggle上重新生成）
- 删除 `.git/` 目录（如果需要）
- 压缩项目为 `MCM_kaggle.zip`

---

## 📦 项目打包上传

### 方法1：使用准备脚本（推荐）

```bash
# 运行准备脚本
cd scripts/configs/kaggle_hyperparam_search
bash prepare_for_kaggle.sh

# 脚本会生成 MCM_kaggle.zip
```

### 方法2：手动打包

```bash
# 在项目根目录
cd /path/to/MCM

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

# 打包（排除不必要文件）
zip -r MCM_kaggle.zip . \\
    -x "*.git*" \\
    -x "*__pycache__*" \\
    -x "*.pyc" \\
    -x "*checkpoints/*" \\
    -x "*.zip"
```

### 上传到Kaggle数据集

1. 访问 [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
2. 点击 **"New Dataset"**
3. 上传 `MCM_kaggle.zip`
4. 设置数据集名称：`{dataset_name}` （或你喜欢的名称）
5. 选择 **Private** （私有数据集）
6. 点击 **"Create"**

⚠️ **注意**：Kaggle数据集上传后会自动解压，所以你的项目文件会在 `/kaggle/input/{dataset_name}/MCM/` 或 `/kaggle/input/{dataset_name}/` 下。

---

## 📓 创建Kaggle Notebook

### 1. 创建新Notebook

1. 访问 [https://www.kaggle.com/code](https://www.kaggle.com/code)
2. 点击 **"New Notebook"**
3. 选择 **Python**
4. 设置Notebook标题：`MCM Hyperparameter Search`

### 2. 配置Notebook设置

点击右侧设置面板：

**加速器 (Accelerator)**：
- 选择 **GPU P100** （推荐）
- 或 **GPU T4** （如果P100不可用）
- ⚠️ 不要选择 TPU

**持久化 (Persistence)**：
- 如果可用，开启 **"Enable GPU"** 和 **"Internet"**

**数据集 (Data)**：
- 点击 **"Add Data"**
- 搜索并添加你上传的数据集：`{dataset_name}`
- 数据集会挂载到 `/kaggle/input/{dataset_name}/`

### 3. Notebook代码

在第一个Cell中粘贴以下代码：

```python
# Cell 1: 环境设置和项目复制
import os
import sys
import shutil
from pathlib import Path

# 检查项目路径
print("检查数据集路径...")
print("可用数据集:", os.listdir("/kaggle/input"))

# 找到项目路径
dataset_name = "{dataset_name}"
possible_paths = [
    f"/kaggle/input/{{dataset_name}}/MCM",
    f"/kaggle/input/{{dataset_name}}",
]

project_source = None
for path in possible_paths:
    if os.path.exists(path):
        project_source = Path(path)
        print(f"✓ 找到项目: {{path}}")
        break

if project_source is None:
    raise FileNotFoundError("未找到MCM项目！")

# 复制到工作目录
work_dir = Path("/kaggle/working/MCM")
if not work_dir.exists():
    print("复制项目到工作目录...")
    shutil.copytree(project_source, work_dir)
    print("✓ 复制完成")

# 切换工作目录
os.chdir(work_dir)
sys.path.insert(0, str(work_dir))
print(f"当前工作目录: {{os.getcwd()}}")
```

```python
# Cell 2: 安装依赖
!pip install -q transformers datasets torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn tqdm

print("✓ 依赖安装完成")
```

```python
# Cell 3: 检查GPU
import torch

if torch.cuda.is_available():
    print(f"✓ GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"  显存: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")
else:
    print("⚠️ 未检测到GPU")
```

```python
# Cell 4: 运行实验
# 使用kaggle_runner.py脚本

# 从数据集中复制运行脚本
runner_script = work_dir / "scripts/configs/kaggle_hyperparam_search/kaggle_runner.py"

if not runner_script.exists():
    print(f"错误: 运行脚本不存在: {{runner_script}}")
else:
    # 运行前5个实验（根据时间调整）
    !python {{str(runner_script)}} --start_exp 1 --end_exp 5
```

### 4. 调整实验范围

根据Kaggle时间限制调整实验数量：

| GPU类型 | 可用时间 | 建议实验数 |
|---------|----------|-----------|
| P100    | 9小时    | 3-5个实验  |
| T4      | 9小时    | 2-3个实验  |

**估算**：每个实验约1.5-2小时（取决于任务和数据集大小）

---

## 🚀 运行实验

### 方式1：运行全部Cell（推荐）

点击 **"Run All"** 按钮

### 方式2：逐个Cell运行

依次点击每个Cell的运行按钮

### 监控进度

- 观察输出日志
- 检查 `/kaggle/working/checkpoints/` 目录
- 查看 `experiment_progress.json` 了解进度

### 分批运行策略

由于Kaggle有9-12小时时间限制，建议分批运行：

**第1批**（实验1-5）：
```python
!python kaggle_runner.py --start_exp 1 --end_exp 5
```

**第2批**（实验6-10）：
```python
!python kaggle_runner.py --start_exp 6 --end_exp 10
```

每批运行完成后：
1. 下载 `/kaggle/working/checkpoints/` 到本地
2. 创建新的Notebook继续下一批

---

## 💾 结果下载

### 下载检查点文件

在Notebook的最后一个Cell中：

```python
# 打包结果
import shutil

output_dir = Path("/kaggle/working/checkpoints")
if output_dir.exists():
    shutil.make_archive("/kaggle/working/results", 'zip', output_dir)
    print("✓ 结果已打包: /kaggle/working/results.zip")
    print(f"  大小: {{(Path('/kaggle/working/results.zip').stat().st_size / 1e6):.1f}} MB")
```

然后点击右侧 **Output** 标签页，下载 `results.zip`

### 下载单个文件

也可以在Notebook中直接查看和下载单个文件：

```python
# 列出所有结果文件
!ls -lh /kaggle/working/checkpoints/
```

---

## ⚠️ 注意事项

### Kaggle限制

1. **运行时间**: 9-12小时后会自动终止
   - 解决：分批运行，每批3-5个实验
   
2. **磁盘空间**: ~20GB
   - 解决：定期删除中间结果，只保留最终模型

3. **GPU显存**: P100约16GB
   - 解决：如果OOM，减小batch_size

4. **网络限制**: 某些外部资源可能无法访问
   - 解决：将预训练模型包含在数据集中

### 路径问题

- Kaggle数据集是**只读**的（`/kaggle/input/`）
- 所有输出必须写到 `/kaggle/working/`
- 项目代码层级不能超过5层（已通过复制到工作目录解决）

### 模型保存

配置文件已自动将checkpoint路径设置为：
```
/kaggle/working/checkpoints/
```

不需要手动修改。

---

## 🔍 故障排查

### 问题1: ModuleNotFoundError: No module named 'scripts'

**原因**: 工作目录不正确

**解决**:
```python
import os, sys
os.chdir("/kaggle/working/MCM")
sys.path.insert(0, "/kaggle/working/MCM")
```

### 问题2: FileNotFoundError: 数据集路径不存在

**原因**: 数据集未正确挂载

**解决**:
```python
# 检查数据集
!ls -la /kaggle/input/
!ls -la /kaggle/input/{dataset_name}/
```

### 问题3: CUDA out of memory

**原因**: GPU显存不足

**解决**:
1. 修改配置文件中的 `batch_size`
2. 或在训练脚本中添加：
```python
torch.cuda.empty_cache()
```

### 问题4: 运行时间超过限制

**原因**: Kaggle 9小时限制

**解决**:
- 减少每批实验数量
- 使用 `--start_exp` 和 `--end_exp` 参数分批运行

### 问题5: 无法保存结果

**原因**: 写入只读目录

**解决**:
确保所有输出路径都在 `/kaggle/working/` 下

---

## 📊 结果分析

下载结果后，在本地运行分析脚本：

```bash
# 解压结果
unzip results.zip -d ./kaggle_results

# 运行分析
python scripts/configs/kaggle_hyperparam_search/analyze_kaggle_results.py \\
    --results_dir ./kaggle_results
```

---

## 📞 获取帮助

遇到问题？

1. 检查Kaggle Notebook的输出日志
2. 查看 `/kaggle/working/experiment_progress.json`
3. 检查数据集是否正确上传
4. 确认GPU是否可用

---

## 实验配置总结

- **总实验数**: {total_configs}
- **任务**: MATE, MNER, MABSA
- **每个任务**: text_only → multimodal
- **超参数**: lr, step_size, gamma
- **每批建议数**: {max_per_session}
- **预计总时间**: 约 {total_configs * 1.5 / max_per_session:.0f} 个Kaggle会话

---

Good luck! 🚀
''')


def _generate_preparation_script(script_path: Path):
    """生成项目准备脚本"""
    
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('''#!/bin/bash
#===============================================================================
# Kaggle项目准备脚本
#
# 功能:
#   1. 清理不必要的文件（缓存、checkpoints等）
#   2. 压缩项目为 MCM_kaggle.zip
#   3. 准备上传到Kaggle数据集
#
# 使用方法:
#   cd MCM  # 进入项目根目录
#   bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
#===============================================================================

set -e

# 颜色定义
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
BLUE='\\033[0;34m'
NC='\\033[0m'

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Kaggle项目准备脚本${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""

# 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    echo -e "${YELLOW}当前目录: $(pwd)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"
echo ""

# 1. 清理Python缓存
echo -e "${BLUE}[1/5] 清理Python缓存...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Python缓存已清理${NC}"
echo ""

# 2. 清理checkpoints（可选，节省空间）
echo -e "${BLUE}[2/5] 清理checkpoints目录...${NC}"
read -p "是否删除checkpoints目录？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf checkpoints/*
    echo -e "${GREEN}✓ checkpoints已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留checkpoints${NC}"
fi
echo ""

# 3. 清理日志文件（可选）
echo -e "${BLUE}[3/5] 清理日志文件...${NC}"
read -p "是否删除log目录？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf log/*
    echo -e "${GREEN}✓ 日志已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留日志${NC}"
fi
echo ""

# 4. 清理.git（可选，大幅减小体积）
echo -e "${BLUE}[4/5] 清理.git目录...${NC}"
read -p "是否删除.git目录？(这会删除Git历史！) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf .git
    echo -e "${GREEN}✓ .git已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留.git${NC}"
fi
echo ""

# 5. 压缩项目
echo -e "${BLUE}[5/5] 压缩项目...${NC}"

OUTPUT_ZIP="MCM_kaggle.zip"

# 删除旧的压缩包
if [ -f "$OUTPUT_ZIP" ]; then
    rm "$OUTPUT_ZIP"
fi

# 压缩（排除不必要的文件）
zip -r "$OUTPUT_ZIP" . \\
    -x "*.git*" \\
    -x "*__pycache__*" \\
    -x "*.pyc" \\
    -x "*checkpoints/*" \\
    -x "*log/*" \\
    -x "*.zip" \\
    -x "*test_outputs/*" \\
    -x "*.ipynb_checkpoints*" \\
    -q

FILE_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)

echo -e "${GREEN}✓ 项目已压缩: $OUTPUT_ZIP (大小: $FILE_SIZE)${NC}"
echo ""

# 显示后续步骤
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}准备完成！${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}下一步:${NC}"
echo -e "  1. 访问 https://www.kaggle.com/datasets"
echo -e "  2. 点击 'New Dataset'"
echo -e "  3. 上传 $OUTPUT_ZIP"
echo -e "  4. 设置数据集名称（例如: mcm-project）"
echo -e "  5. 选择 Private（私有）"
echo -e "  6. 点击 Create"
echo ""
echo -e "${YELLOW}注意:${NC}"
echo -e "  - Kaggle会自动解压zip文件"
echo -e "  - 你的项目会在 /kaggle/input/<数据集名称>/ 下"
echo -e "  - 上传可能需要一些时间，取决于文件大小"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
''')
    
    script_path.chmod(0o755)


def _generate_kaggle_analysis_script(script_path: Path, configs: list):
    """生成Kaggle结果分析脚本"""
    
    with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('''#!/usr/bin/env python3
"""
Kaggle超参数搜索结果分析脚本

从Kaggle下载的结果目录中提取和分析实验结果
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
import argparse


def calculate_metrics(acc_matrix):
    """计算持续学习指标"""
    n = len(acc_matrix)
    if n == 0:
        return {}
    
    # AA (Average Accuracy)
    aa = np.mean([acc_matrix[n-1][j] for j in range(n)])
    
    # AIA (Average Incremental Accuracy)
    aia_values = []
    for i in range(n):
        avg_acc = np.mean([acc_matrix[i][j] for j in range(i+1)])
        aia_values.append(avg_acc)
    aia = np.mean(aia_values)
    
    # FM (Forgetting Measure)
    fm_values = []
    for j in range(n-1):
        max_acc = max([acc_matrix[i][j] for i in range(j, n)])
        final_acc = acc_matrix[n-1][j]
        fm_values.append(max_acc - final_acc)
    fm = np.mean(fm_values) if fm_values else 0.0
    
    # BWT (Backward Transfer)
    bwt_values = []
    for j in range(n-1):
        final_acc = acc_matrix[n-1][j]
        acc_after_j = acc_matrix[j][j]
        bwt_values.append(final_acc - acc_after_j)
    bwt = np.mean(bwt_values) if bwt_values else 0.0
    
    # FWT (Forward Transfer)
    fwt_values = []
    for j in range(1, n):
        acc_before_j = acc_matrix[j-1][j] if j > 0 else 0.0
        fwt_values.append(acc_before_j)
    fwt = np.mean(fwt_values) if fwt_values else 0.0
    
    return {
        "AA": aa,
        "AIA": aia,
        "FM": fm,
        "BWT": bwt,
        "FWT": fwt
    }


def find_train_info_files(results_dir: Path):
    """查找所有train_info文件"""
    return list(results_dir.glob("**/train_info_*.json"))


def extract_hyperparams_from_filename(filename: str):
    """从文件名提取超参数信息"""
    # 文件名格式: train_info_twitter2015_none_t2m_hpX.json
    # 我们需要从配置文件或其他地方获取超参数
    return None


def analyze_results(results_dir: Path, output_dir: Path):
    """分析Kaggle结果"""
    
    print(f"分析目录: {results_dir}")
    print()
    
    # 查找所有train_info文件
    train_info_files = find_train_info_files(results_dir)
    
    if not train_info_files:
        print("❌ 未找到任何train_info文件")
        return
    
    print(f"找到 {len(train_info_files)} 个结果文件")
    print()
    
    all_results = []
    
    for train_info_path in train_info_files:
        print(f"处理: {train_info_path.name}", end=" ... ")
        
        try:
            with open(train_info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            acc_matrix = np.array(data.get("accuracy_matrix", []))
            
            if len(acc_matrix) == 0:
                print("❌ 无准确率矩阵")
                continue
            
            # 计算指标
            metrics = calculate_metrics(acc_matrix)
            
            # 从文件名提取信息
            # 格式: train_info_<dataset>_<strategy>_<mode>_<seq>.json
            parts = train_info_path.stem.replace("train_info_", "").split("_")
            
            result = {
                "file": train_info_path.name,
                "dataset": parts[0] if len(parts) > 0 else "unknown",
                "strategy": parts[1] if len(parts) > 1 else "unknown",
                "mode": parts[2] if len(parts) > 2 else "unknown",
                "seq": parts[3] if len(parts) > 3 else "unknown",
                **metrics,
                "acc_matrix": acc_matrix.tolist()
            }
            
            # 添加任务准确率
            n = len(acc_matrix)
            for i in range(n):
                result[f"Task{i+1}_AfterT{i+1}"] = acc_matrix[i][i]
            for j in range(n):
                result[f"Task{j+1}_Final"] = acc_matrix[n-1][j]
            
            all_results.append(result)
            print("✓")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    if not all_results:
        print("\\n没有成功提取的结果")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = output_dir / "kaggle_results_summary.csv"
    df.to_csv(results_csv, index=False, encoding='utf-8')
    print(f"\\n✓ 结果已保存: {results_csv}")
    
    # 显示统计
    print("\\n" + "="*80)
    print("结果统计")
    print("="*80)
    print(f"\\n平均 AA: {df['AA'].mean():.4f} ± {df['AA'].std():.4f}")
    print(f"平均 AIA: {df['AIA'].mean():.4f} ± {df['AIA'].std():.4f}")
    print(f"平均 FM: {df['FM'].mean():.4f} ± {df['FM'].std():.4f}")
    print(f"平均 BWT: {df['BWT'].mean():.4f} ± {df['BWT'].std():.4f}")
    print(f"平均 FWT: {df['FWT'].mean():.4f} ± {df['FWT'].std():.4f}")
    
    # 显示最佳结果
    print("\\n" + "="*80)
    print("最佳结果 (按AA排序)")
    print("="*80)
    
    df_sorted = df.sort_values("AA", ascending=False)
    print("\\nTop 5:")
    for idx, row in df_sorted.head(5).iterrows():
        print(f"  {row['file']}: AA={row['AA']:.4f}, FM={row['FM']:.4f}")
    
    print("\\n分析完成！")


def main():
    parser = argparse.ArgumentParser(description="分析Kaggle实验结果")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Kaggle结果目录（解压后的checkpoints目录）")
    parser.add_argument("--output_dir", type=str, default="./kaggle_analysis",
                       help="分析结果输出目录")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return
    
    analyze_results(results_dir, output_dir)


if __name__ == "__main__":
    main()
''')
    
    script_path.chmod(0o755)


if __name__ == "__main__":
    main()

