# CRF修复测试 - Kaggle完整部署指南

## 📌 测试目标

验证三个关键修复在序列标注任务上的效果：

1. **CRF层** - 强制BIO约束
2. **valid_len修复** - 边界准确性
3. **Span Loss** - 边界强化

**测试任务**：MATE、MNER、MABSA（各1个实验，共3个）

**预期改进**：Chunk F1从30%提升到60-75%

---

## 🔧 步骤1：本地准备

### 1.1 生成测试配置

```bash
cd /path/to/MCM

# 生成本地测试配置
python scripts/generate_crf_test_configs.py

# 生成Kaggle测试配置
python scripts/generate_crf_test_configs.py --kaggle
```

生成的文件：

- ✅ `scripts/configs/crf_test/crf_test_twitter2015_*.json` - 配置文件
- ✅ `scripts/configs/crf_test/test_index.json` - 索引
- ✅ `scripts/configs/crf_test/run_crf_tests.sh` - 本地运行脚本

### 1.2 打包项目

**选择模式**：

**模式A：完整模式**（首次推荐）

```bash
# 使用现有的打包脚本
bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
# 生成 MCM_kaggle.zip
```

**模式B：分离模式**（推荐，代码更新快）

```bash
# 1. 打包数据（一次性）
bash scripts/configs/kaggle_hyperparam_search/prepare_data_only.sh
# 生成 MCM_data.zip

# 2. 打包代码
bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh
# 生成 MCM_code.zip
```

---

## 📦 步骤2：上传到Kaggle

### 完整模式

1. 上传 `MCM_kaggle.zip` 到Kaggle Datasets
2. 命名为 `mcm-project`

### 分离模式（推荐）

1. 上传 `MCM_data.zip` → 命名为 `mcm-data`
2. 上传 `MCM_code.zip` → 命名为 `mcm-code`

详细步骤参考：`scripts/configs/kaggle_hyperparam_search/KAGGLE_SETUP_GUIDE.md`

---

## 📓 步骤3：创建Kaggle Notebook

### 3.1 新建Notebook

1. 访问 [https://www.kaggle.com/code](https://www.kaggle.com/code)
2. 点击 **"New Notebook"**
3. 标题：`CRF Fix Test - MATE MNER MABSA`

### 3.2 配置Notebook

**Accelerator**: GPU P100 或 T4**Internet**: 开启**Data**:

- 完整模式：添加 `mcm-project`
- 分离模式：添加 `mcm-code` 和 `mcm-data`

### 3.3 Notebook代码

#### Cell 1: 环境检查（自动检测模式）

```python
import os
import sys
import shutil
from pathlib import Path

print("="*80)
print("CRF修复测试 - 环境检查")
print("="*80)

# 检查Kaggle环境
print("\n📦 可用数据集:")
for dataset in os.listdir("/kaggle/input"):
    print(f"  - {dataset}")

# 自动检测模式
use_split_mode = False
code_path = None
data_path = None

# 检测分离模式
if os.path.exists("/kaggle/input/mcm-code"):
    use_split_mode = True
    code_path = Path("/kaggle/input/mcm-code")
    print("\n✓ 检测到分离模式")
    print(f"  代码路径: {code_path}")
  
    if os.path.exists("/kaggle/input/mcm-data"):
        data_path = Path("/kaggle/input/mcm-data")
        print(f"  数据路径: {data_path}")
    else:
        print("  ⚠️ 未找到 mcm-data，请在Data面板添加")

# 检测完整模式
else:
    possible_paths = [
        Path("/kaggle/input/mcm-project/MCM"),
        Path("/kaggle/input/mcm-project"),
    ]
  
    for path in possible_paths:
        if path.exists() and (path / "scripts").exists():
            code_path = path
            print(f"\n✓ 检测到完整模式")
            print(f"  项目路径: {path}")
            break

if code_path is None:
    raise FileNotFoundError("❌ 未找到项目！请检查数据集配置")

# 列出项目内容
print("\n📁 项目内容（前10项）:")
items = sorted(list(code_path.iterdir()))[:10]
for item in items:
    print(f"  - {item.name}{'/' if item.is_dir() else ''}")

print("\n✅ 环境检查完成")
```

#### Cell 2: 复制项目到工作目录

```python
# 复制项目到可写目录
work_project_path = Path("/MCM")

print("="*80)
print("复制项目到工作目录")
print("="*80)

if not work_project_path.exists():
    print(f"\n📋 复制代码...")
    print(f"  源: {code_path}")
    print(f"  目标: {work_project_path}")
    shutil.copytree(code_path, work_project_path, dirs_exist_ok=True)
    print("✓ 代码复制完成")
else:
    print("⚠️ 工作目录已存在，跳过复制")

# 如果是分离模式，链接数据目录
if use_split_mode and data_path:
    print("\n📋 链接数据目录（分离模式）...")
  
    target_data = work_project_path / "data"
    target_model = work_project_path / "downloaded_model"
  
    # 链接data
    if not target_data.exists():
        source_data = data_path / "data" if (data_path / "data").exists() else data_path
        print(f"  data: {source_data} → {target_data}")
        try:
            os.symlink(source_data, target_data)
            print("  ✓ data链接成功（符号链接）")
        except:
            print("  ⚠️ 符号链接失败，改用复制...")
            shutil.copytree(source_data, target_data, dirs_exist_ok=True)
            print("  ✓ data复制完成")
    else:
        print(f"  ✓ data目录已存在")
  
    # 链接模型
    source_model = data_path / "downloaded_model"
    if source_model.exists() and not target_model.exists():
        print(f"  downloaded_model: {source_model} → {target_model}")
        try:
            os.symlink(source_model, target_model)
            print("  ✓ downloaded_model链接成功（符号链接）")
        except:
            shutil.copytree(source_model, target_model, dirs_exist_ok=True)
            print("  ✓ downloaded_model复制完成")
    else:
        print(f"  ✓ downloaded_model目录已存在")

# 切换工作目录
os.chdir(work_project_path)
sys.path.insert(0, str(work_project_path))

print(f"\n📂 当前工作目录: {os.getcwd()}")
print(f"🐍 Python路径: {sys.path[0]}")

# 验证数据集
data_dir = work_project_path / "data"
print(f"\n📁 数据目录: {data_dir}")
print(f"   存在: {data_dir.exists()}")

if data_dir.exists():
    print("\n📦 可用数据集:")
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            file_count = len(list(item.iterdir()))
            print(f"  - {item.name}/ ({file_count} files)")

print("\n✅ 项目准备完成")
```

#### Cell 3: 安装依赖

```python
import subprocess

print("="*80)
print("安装依赖")
print("="*80)

# 检查依赖文件
kaggle_req = work_project_path / "requirements_kaggle.txt"
regular_req = work_project_path / "requirements.txt"

print("\n📦 检查依赖文件...")

if kaggle_req.exists():
    print("✓ 找到 requirements_kaggle.txt（Kaggle优化版）")
    print("\n安装Kaggle特定依赖...")
    !pip install -q -r {str(kaggle_req)}
    print("✓ 依赖安装完成")
  
elif regular_req.exists():
    print("✓ 找到 requirements.txt（标准版）")
    print("\n⚠️ 可能有版本冲突警告（可以忽略）")
    !pip install -q -r {str(regular_req)}
    print("\n✓ 依赖安装完成（版本冲突警告可忽略）")
  
else:
    print("⚠️ 未找到依赖文件，安装最小依赖集...")
    !pip install -q pytorch_crf sentencepiece protobuf==3.20.3
    print("✓ 最小依赖安装完成")

# 验证关键包
print("\n🔍 验证关键依赖...")
try:
    import torch
    print(f"  ✓ torch: {torch.__version__}")
except:
    print("  ✗ torch导入失败")

try:
    import transformers
    print(f"  ✓ transformers: {transformers.__version__}")
except:
    print("  ✗ transformers导入失败")

try:
    from torchcrf import CRF
    print(f"  ✓ torchcrf: 可用")
except:
    print("  ⚠️ torchcrf不可用（将使用内置SimpleCRF）")

print("\n💡 说明:")
print("  • Kaggle已预装大部分包")
print("  • 版本冲突警告通常可以忽略")
print("  • torchcrf不可用时会自动使用SimpleCRF")

print("\n✅ 依赖检查完成")
```

#### Cell 4: 检查GPU和测试修复

```python
import torch

print("="*80)
print("GPU信息 & 修复验证")
print("="*80)

# GPU信息
print("\n🖥️ GPU状态:")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
  
    print(f"  ✓ GPU: {gpu_name}")
    print(f"  ✓ 总显存: {gpu_memory:.1f} GB")
    print(f"  ✓ CUDA: {torch.version.cuda}")
    print(f"  ✓ PyTorch: {torch.__version__}")
else:
    print("  ❌ 未检测到GPU")
    print("  请在Settings → Accelerator中选择GPU")

# 验证CRF修复
print("\n🔧 验证修复是否已集成:")

# 1. 检查CRF是否可用
try:
    from models.task_heads.mate_head import MATEHead
    head = MATEHead(768, 3, use_crf=True)
    if hasattr(head, 'crf') and head.crf is not None:
        print("  ✓ CRF层: 已集成")
        print(f"    类型: {type(head.crf).__name__}")
    else:
        print("  ✗ CRF层: 未找到")
except Exception as e:
    print(f"  ✗ CRF层检查失败: {e}")

# 2. 检查Span Loss
try:
    from utils.span_loss import SpanLoss
    span_loss = SpanLoss('mate')
    print("  ✓ Span Loss: 已集成")
except Exception as e:
    print(f"  ✗ Span Loss检查失败: {e}")

# 3. 检查训练循环
try:
    with open(work_project_path / "modules/training_loop_fixed.py", "r") as f:
        content = f.read()
        if "span_loss" in content.lower():
            print("  ✓ 训练循环: 已集成Span Loss")
        else:
            print("  ⚠️ 训练循环: 未找到Span Loss")
except:
    print("  ⚠️ 无法检查训练循环")

print("\n✅ 系统准备完成")
```

#### Cell 5: 配置测试参数

```python
print("="*80)
print("CRF修复测试配置")
print("="*80)

# 测试配置
TEST_TASKS = ["mate", "mner", "mabsa"]
DATASET = "twitter2015"

print("\n📋 测试任务:")
for i, task in enumerate(TEST_TASKS, 1):
    print(f"  {i}. {task.upper()}")

print(f"\n📊 数据集: {DATASET}")
print(f"📦 总测试数: {len(TEST_TASKS)}")

# 时间估算
time_per_test = 0.5  # 小时（CRF测试较快，因为只是验证，不是完整训练）
total_time = len(TEST_TASKS) * time_per_test

print(f"\n⏱️ 预计时间: {total_time:.1f} 小时")
print(f"   (每个测试约 {time_per_test} 小时)")

print("\n🎯 测试目标:")
print("  验证修复效果:")
print("  1. CRF层 → 强制BIO约束")
print("  2. valid_len修复 → 边界准确")
print("  3. Span Loss → 边界强化")

print("\n📈 预期改进:")
print("  Chunk F1: 30% → 60-75% (+30-45%)")

print("\n✅ 配置完成")
```

#### Cell 6: 运行CRF测试

```python
import json
import subprocess
import time
from datetime import datetime

print("="*80)
print("开始运行CRF修复测试")
print("="*80)
print()

# 确保输出目录存在
output_dir = Path("/kaggle/working/checkpoints")
output_dir.mkdir(parents=True, exist_ok=True)

# 加载测试索引
index_file = work_project_path / "scripts/configs/crf_test/test_index.json"
if not index_file.exists():
    print(f"❌ 索引文件不存在: {index_file}")
    print("请确保配置文件已正确生成")
else:
    with open(index_file) as f:
        test_index = json.load(f)
  
    total_tests = test_index['total_configs']
    print(f"📊 加载测试索引: {total_tests} 个测试")
    print()
  
    # 运行每个测试
    results = []
    overall_start = time.time()
  
    for i, config_info in enumerate(test_index['configs'], 1):
        task = config_info['task']
        config_file = work_project_path / "scripts/configs/crf_test" / config_info['file']
      
        print("="*80)
        print(f"测试 [{i}/{total_tests}]: {task.upper()}")
        print("="*80)
        print(f"配置: {config_info['file']}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
      
        # 更新配置文件路径到Kaggle输出目录
        print("📝 更新配置文件路径...")
        with open(config_file, 'r') as f:
            config = json.load(f)
      
        # 更新输出路径
        original_output = config["global_params"]["output_model_path"]
        original_train_info = config["global_params"]["train_info_json"]
      
        config["global_params"]["output_model_path"] = f"/kaggle/working/checkpoints/{Path(original_output).name}"
        config["global_params"]["train_info_json"] = f"/kaggle/working/checkpoints/{Path(original_train_info).name}"
      
        # 更新其他路径
        if "ewc_dir" in config["global_params"]:
            config["global_params"]["ewc_dir"] = "/kaggle/working/checkpoints/ewc"
        if "gem_mem_dir" in config["global_params"]:
            config["global_params"]["gem_mem_dir"] = "/kaggle/working/checkpoints/gem"
      
        # 保存更新后的配置
        temp_config = Path("/kaggle/working") / f"temp_config_{task}.json"
        with open(temp_config, 'w') as f:
            json.dump(config, f, indent=2)
      
        print(f"  ✓ 输出路径更新到: /kaggle/working/checkpoints/")
        print()
      
        # 运行测试
        test_start = time.time()
        cmd = [
            "python", "-m", "scripts.train_with_zero_shot",
            "--config", str(temp_config)
        ]
      
        # 设置环境变量，确保Python能找到模块
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(work_project_path)
      
        try:
            print(f"🚀 运行命令: {' '.join(cmd)}")
            print(f"   工作目录: {work_project_path}")
            print(f"   PYTHONPATH: {env['PYTHONPATH']}")
            print()
            # 在项目根目录运行，并设置PYTHONPATH
            result = subprocess.run(cmd, check=True, capture_output=False, 
                                   cwd=str(work_project_path), env=env)
            success = True
            print()
            print(f"✅ {task.upper()} 测试完成")
        except subprocess.CalledProcessError as e:
            success = False
            print()
            print(f"❌ {task.upper()} 测试失败: {e}")
      
        test_time = time.time() - test_start
      
        # 验证输出文件
        output_files = list(output_dir.glob("**/*"))
        output_files = [f for f in output_files if f.is_file()]
      
        print(f"⏱️ 耗时: {test_time/60:.1f} 分钟")
        print(f"📁 输出文件数: {len(output_files)}")
      
        # 记录结果
        results.append({
            "task": task,
            "config": config_info['file'],
            "success": success,
            "time_minutes": round(test_time/60, 1),
            "output_files": len(output_files)
        })
      
        # 保存进度
        progress_file = Path("/kaggle/working/crf_test_progress.json")
        with open(progress_file, 'w') as f:
            json.dump({
                "completed": i,
                "total": total_tests,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
      
        print()
  
    # 总结
    total_time = time.time() - overall_start
    success_count = sum(1 for r in results if r['success'])
  
    print("="*80)
    print("🎉 所有测试完成!")
    print("="*80)
    print(f"\n📊 测试统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  成功: {success_count}")
    print(f"  失败: {total_tests - success_count}")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
  
    print(f"\n📋 详细结果:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"  {status} {r['task'].upper()}: {r['time_minutes']:.1f} 分钟")
  
    print(f"\n💾 结果保存在: /kaggle/working/checkpoints/")
    print(f"📝 进度文件: /kaggle/working/crf_test_progress.json")
```

#### Cell 7: 检查结果和性能对比

```python
import json
from pathlib import Path

print("="*80)
print("结果检查 & 性能对比")
print("="*80)

output_dir = Path("/kaggle/working/checkpoints")

if output_dir.exists():
    # 统计文件
    all_files = list(output_dir.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    train_info_files = [f for f in files if "train_info" in f.name and f.suffix == ".json"]
    model_files = [f for f in files if f.suffix == ".pt"]
  
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  总文件数: {len(files)}")
    print(f"  train_info: {len(train_info_files)}")
    print(f"  模型文件: {len(model_files)}")
  
    # 分析性能
    print(f"\n📊 性能分析:")
    print("-" * 80)
  
    for train_info in sorted(train_info_files):
        try:
            with open(train_info, 'r') as f:
                data = json.load(f)
          
            # 提取任务名
            task_name = "unknown"
            if "sessions" in data and len(data["sessions"]) > 0:
                task_name = data["sessions"][0].get("task_name", "unknown")
          
            print(f"\n🔍 {task_name.upper()}")
          
            # 提取指标
            if "sessions" in data and len(data["sessions"]) > 0:
                session = data["sessions"][0]
                details = session.get("details", {})
              
                # 最终指标
                final_dev = details.get("final_dev_metrics", {})
                final_test = details.get("final_test_metrics", {})
              
                if final_test:
                    print(f"  📈 Test Set:")
                  
                    # 显示主指标
                    if "chunk_f1" in final_test:
                        chunk_f1 = final_test["chunk_f1"]
                        print(f"    Chunk F1: {chunk_f1:.2f}% ⭐ (主指标1)")
                  
                    if "token_micro_f1_no_o" in final_test:
                        token_f1 = final_test["token_micro_f1_no_o"]
                        print(f"    Token Micro F1 (无O): {token_f1:.2f}% (主指标2)")
                  
                    if "token_acc" in final_test:
                        token_acc = final_test["token_acc"]
                        print(f"    Token Accuracy: {token_acc:.2f}% (参考)")
                  
                    # 边界检测指标
                    if "chunk_precision" in final_test:
                        print(f"    Chunk Precision: {final_test['chunk_precision']:.2f}%")
                    if "chunk_recall" in final_test:
                        print(f"    Chunk Recall: {final_test['chunk_recall']:.2f}%")
              
                # CRF使用信息
                if "args" in session:
                    args = session["args"]
                    use_crf = args.get("use_crf", False)
                    use_span_loss = args.get("use_span_loss", False)
                    print(f"\n  🔧 修复启用状态:")
                    print(f"    CRF: {'✅ 已启用' if use_crf else '❌ 未启用'}")
                    print(f"    Span Loss: {'✅ 已启用' if use_span_loss else '❌ 未启用'}")
      
        except Exception as e:
            print(f"  ⚠️ 读取失败: {e}")
  
    # 计算总大小
    total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"\n💾 总大小: {total_size:.1f} MB")
  
    print("\n" + "="*80)
    print("💡 对比说明:")
    print("="*80)
    print("  修复前（预期）:")
    print("    - Token Accuracy: ~90%")
    print("    - Chunk F1: ~30%")
    print("    - 问题: token准确但边界识别失败")
    print()
    print("  修复后（目标）:")
    print("    - Token Accuracy: ~90%")
    print("    - Chunk F1: 60-75% (+30-45%)")
    print("    - CRF强制BIO约束，span loss强化边界")
    print("="*80)
  
else:
    print("❌ 输出目录不存在")

print("\n✅ 结果检查完成")
```

#### Cell 8: 打包结果

```python
import shutil
from pathlib import Path

print("="*80)
print("打包实验结果")
print("="*80)

output_dir = Path("/kaggle/working/checkpoints")
output_zip = Path("/kaggle/working/crf_test_results.zip")

if output_dir.exists():
    print("\n📦 正在打包...")
    print(f"  源目录: {output_dir}")
    print(f"  目标文件: {output_zip}")
  
    # 创建压缩包
    shutil.make_archive(
        str(output_zip.with_suffix('')),
        'zip',
        output_dir
    )
  
    if output_zip.exists():
        zip_size = output_zip.stat().st_size / (1024 * 1024)
        print(f"\n✅ 结果已打包!")
        print(f"  文件: {output_zip.name}")
        print(f"  大小: {zip_size:.1f} MB")
        print(f"\n📥 下载方式:")
        print(f"  1. 点击右侧 'Output' 标签")
        print(f"  2. 找到 {output_zip.name}")
        print(f"  3. 点击下载按钮")
    else:
        print("❌ 打包失败")
else:
    print("❌ 输出目录不存在，无内容可打包")

print("\n" + "="*80)
print("⚠️ 为节省GPU配额，请完成以下操作:")
print("="*80)
print("  1. 下载 crf_test_results.zip")
print("  2. 点击右上角 'Stop Session' 停止Notebook")
print("  3. 或等待脚本自动退出后手动停止")
print("="*80)

print("\n✅ 打包完成")
```

---

## 🚀 步骤4：运行测试

### 4.1 执行方式

点击 **"Run All"** 或逐个运行Cell

### 4.2 监控进度

- 观察Cell 6的输出
- 查看 `/kaggle/working/crf_test_progress.json`

### 4.3 时间估算

- 每个测试：~30分钟
- 总计：~1.5小时（3个测试）

---

## 💾 步骤5：分析结果

### 5.1 下载结果

1. 确保Cell 8已执行
2. 在 **Output** 标签下载 `crf_test_results.zip`

### 5.2 本地分析

```bash
# 解压
unzip crf_test_results.zip -d ./crf_test_results

# 查看结果
cat crf_test_results/train_info_*.json | jq '.sessions[0].details.final_test_metrics'
```

### 5.3 对比验证

检查是否达到预期改进：

- Chunk F1提升 +30-45%
- 边界Precision/Recall提升
- CRF和Span Loss是否启用

---

## 📊 预期结果

### 成功标志

Cell 7应该显示类似：

```
📊 性能分析:
--------------------------------------------------------------------------------

🔍 MATE
  📈 Test Set:
    Chunk F1: 65.34% ⭐ (主指标1)          ← 提升！
    Token Micro F1 (无O): 88.23% (主指标2)
    Token Accuracy: 90.12% (参考)
    Chunk Precision: 67.45%
    Chunk Recall: 63.28%

  🔧 修复启用状态:
    CRF: ✅ 已启用
    Span Loss: ✅ 已启用

🔍 MNER
  📈 Test Set:
    Chunk F1: 72.56% ⭐ (主指标1)          ← 提升！
    ...

🔍 MABSA
  📈 Test Set:
    Chunk F1: 68.91% ⭐ (主指标1)          ← 提升！
    ...
```

---

## ⚠️ 常见问题

详细问题解决参考：`scripts/configs/kaggle_hyperparam_search/KAGGLE_SETUP_GUIDE.md`

### 快速解决

1. **找不到项目** → 检查数据集是否已添加
2. **CUDA out of memory** → 减小batch_size
3. **ModuleNotFoundError** → 确认Cell 2已执行
4. **输出文件为空** → 检查路径是否更新到 `/kaggle/working/checkpoints`

---

## 📋 检查清单

### 运行前

- [ ] 生成了测试配置
- [ ] 项目已打包并上传
- [ ] Notebook已配置GPU
- [ ] 数据集已添加到Notebook

### 运行中

- [ ] Cell 1: 环境检查通过
- [ ] Cell 2: 项目复制成功
- [ ] Cell 3: 依赖安装完成
- [ ] Cell 4: GPU和修复验证通过
- [ ] Cell 6: 测试运行完成
- [ ] Cell 7: 性能对比显示改进

### 运行后

- [ ] Chunk F1提升30-45%
- [ ] 结果已打包下载
- [ ] Session已停止（节省配额）

---

## 📚 相关文档

- 📖 详细Kaggle指南: `scripts/configs/kaggle_hyperparam_search/KAGGLE_SETUP_GUIDE.md`
- 🔧 修复详情: `doc/FIXES_GUIDE.md`
- 📝 快速参考: `doc/FIXES_SUMMARY.md`

---

Good luck with testing! 🚀
