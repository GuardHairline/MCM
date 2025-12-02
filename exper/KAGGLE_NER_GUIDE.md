# Kaggle NER训练指南 - 超参数搜索 & 可视化

## 📌 概述

本指南提供在 Kaggle 上运行 `simple_ner_training.py` 进行 **NER超参数搜索** 的完整流程。

### 核心功能

- ✅ 自动运行多组超参数实验
- ✅ DeBERTa-v3-base + BiLSTM + CRF 架构
- ✅ Twitter2015 MNER 数据集
- ✅ 同时计算 Token-level 和 Span-level F1
- ✅ 自动保存最佳模型
- ✅ 结果自动打包下载
- ✅ 自动生成训练曲线、F1散点图、前2000条DEV的gold/pred spans（tests/ 目录）
- ✅ 复用 `visualize/feature_clustering_enhanced.py` 生成 t-SNE 特征聚类图（真实/预测标签对比），与模型输出同目录

---

## 🚀 快速开始（5分钟设置）

### Step 1: 本地准备数据

```bash
cd /path/to/MCM

# 确保数据结构正确
# data/
#   ├── MNER/
#   │   └── twitter2015/
#   │       ├── train.txt
#   │       ├── dev.txt
#   │       └── test.txt
#   └── img/  # 或 twitter2015_images/
#       ├── xxx.jpg
#       └── ...
```

### Step 2: 打包项目

```bash
# 完整打包（首次使用）
zip -r MCM_ner.zip data/ downloaded_model/ tests/ datasets/ models/ modules/ continual/

# 或只打包必需文件（更快）
zip -r MCM_ner_minimal.zip \
  data/MNER/ \
  data/img/ \
  downloaded_model/deberta-v3-base/ \
  tests/simple_ner_training.py \
  datasets/mner_dataset.py \
  -x "*.pyc" -x "__pycache__/*"
```

### Step 3: 上传到 Kaggle

1. 访问 [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
2. **New Dataset** → 上传 `MCM_ner.zip`
3. Title: `MCM NER Training`
4. Slug: `mcm-ner-training` ⚠️ 重要
5. **Create**

### Step 4: 创建 Notebook

1. [https://www.kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Settings:
   - **GPU P100** 或 **T4**
   - 添加数据集: `mcm-ner-training`
3. 复制下面的代码到 Notebook

---

## 📓 Kaggle Notebook 代码

### Cell 1: 环境检查和设置

```python
import os
import sys
import shutil
from pathlib import Path

print("="*80)
print("环境检查")
print("="*80)

# 检查Kaggle环境
print("\n可用数据集:")
for dataset in os.listdir("/kaggle/input"):
    print(f"  - {dataset}")

# 自动检测模式
use_split_mode = False
code_path = None
data_path = None

# 检测分离模式
if os.path.exists("/kaggle/input/mcm-ner-code"):
    use_split_mode = True
    code_path = Path("/kaggle/input/mcm-ner-code")
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
        if path.exists():
            code_path = path
            print(f"\n✓ 检测到完整模式")
            print(f"  项目路径: {path}")
            break

if code_path is None:
    raise FileNotFoundError("未找到项目！请检查数据集配置")

# 列出项目内容
print("\n项目内容:")
all_items = sorted([item.name for item in code_path.iterdir()])
print(f"  共 {len(all_items)} 项")
for item in all_items[:15]:  # 显示前15项
    print(f"  - {item}")
if len(all_items) > 15:
    print(f"  ... 还有 {len(all_items) - 15} 项")

# 复制项目到可写目录
work_project_path = Path("/MCM")

# 检查是否需要重新复制
need_copy = False
if not work_project_path.exists():
    need_copy = True
    reason = "目录不存在"
else:
    # 检查关键文件
    critical_files = [
        work_project_path / "tests/simple_ner_training.py",
        work_project_path / "datasets/mner_dataset.py",
    ]
    missing_files = [f for f in critical_files if not f.exists()]
  
    if missing_files:
        need_copy = True
        reason = f"缺少关键文件: {[f.name for f in missing_files]}"
        print(f"\n⚠️ 检测到 {work_project_path} 已存在但不完整")
        print(f"   原因: {reason}")
        print("   将删除旧目录并重新复制...")
        shutil.rmtree(work_project_path)
    else:
        print(f"\n✓ {work_project_path} 已存在且完整，跳过复制")

if need_copy:
    print(f"\n复制代码到工作目录 (原因: {reason})...")
    print(f"  源: {code_path}")
    print(f"  目标: {work_project_path}")
    shutil.copytree(code_path, work_project_path, dirs_exist_ok=True)
    print("✓ 复制完成")
  
    # 验证复制结果
    test_file = work_project_path / "tests/simple_ner_training.py"
    if test_file.exists():
        print(f"✓ 验证成功: tests/simple_ner_training.py 存在")
    else:
        print(f"❌ 警告: tests/simple_ner_training.py 仍不存在！")
        print(f"   请检查源路径: {code_path / 'tests'}")

# 如果是分离模式，链接数据目录
if use_split_mode and data_path:
    target_data = work_project_path / "data"
    target_model = work_project_path / "downloaded_model"
  
    # 链接data
    if not target_data.exists():
        source_data = data_path / "data" if (data_path / "data").exists() else data_path
        print(f"\n链接数据目录: {source_data} -> {target_data}")
        try:
            os.symlink(source_data, target_data)
            print("✓ data链接成功")
        except:
            print("  符号链接失败，改用复制...")
            shutil.copytree(source_data, target_data, dirs_exist_ok=True)
            print("✓ data复制完成")
  
    # 链接模型
    source_model = data_path / "downloaded_model"
    if source_model.exists() and not target_model.exists():
        print(f"\n链接模型目录: {source_model} -> {target_model}")
        try:
            os.symlink(source_model, target_model)
            print("✓ downloaded_model链接成功")
        except:
            shutil.copytree(source_model, target_model, dirs_exist_ok=True)
            print("✓ downloaded_model复制完成")

# 切换工作目录
os.chdir(work_project_path)
sys.path.insert(0, str(work_project_path))

print(f"\n当前工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path[0]}")

# 验证数据集
data_dir = work_project_path / "data"
print(f"\n数据目录: {data_dir}")
print(f"数据集存在: {data_dir.exists()}")

if data_dir.exists():
    print("\n可用数据集:")
    for item in data_dir.iterdir():
        if item.is_dir():
            print(f"  - {item.name}/")


print(f"\n当前工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path[0]}")

# 检查关键文件
print("\n关键文件检查:")
key_files = [
    "tests/simple_ner_training.py",
    "datasets/mner_dataset.py",
    "data/MNER/twitter2015/train.txt",
]
all_exist = True
for f in key_files:
    file_path = Path(f)
    exists = file_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {f}")
    if not exists:
        all_exist = False
        # 检查父目录是否存在
        parent_dir = file_path.parent
        if parent_dir.exists():
            print(f"      目录 {parent_dir}/ 存在，但文件不存在")
        else:
            print(f"      目录 {parent_dir}/ 不存在")

if not all_exist:
    print("\n⚠️ 部分文件缺失，请检查:")
    print("  1. 上传的zip是否包含所有必需文件？")
    print("  2. 是否使用了正确的打包命令？")
    print("  3. 建议重新打包并上传")
    print("\n正确的打包命令:")
    print("  zip -r MCM_ner.zip data/ downloaded_model/ tests/ datasets/ models/ modules/ continual/")
else:
    print("\n✅ 所有关键文件检查通过！")
```

### Cell 2: 安装依赖

```python
print("="*80)
print("安装依赖")
print("="*80)

# Kaggle预装了大部分包，只需安装特定的
!pip install -q torchcrf

print("\n✓ 依赖安装完成")

# 验证
import torch
from torchcrf import CRF

print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ torchcrf: 可用")
```

### Cell 3: GPU检查

```python
import torch

print("="*80)
print("GPU信息")
print("="*80)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
  
    print(f"\n✓ GPU: {gpu_name}")
    print(f"  显存: {gpu_memory:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
else:
    print("\n❌ 未检测到GPU")
```

### Cell 4: 定义超参数搜索空间

```python
import json
from pathlib import Path

print("="*80)
print("超参数搜索配置")
print("="*80)

# 定义多组超参数
HYPERPARAMETER_CONFIGS = [
    {
        "id": 1,
        "name": "baseline",
        "learning_rate": 1e-5,
        "lstm_lr": 1e-4,
        "crf_lr": 1e-3,
        "batch_size": 16,
        "num_epochs": 20,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "dropout": 0.3,
    },
    {
        "id": 2,
        "name": "higher_lr",
        "learning_rate": 2e-5,
        "lstm_lr": 2e-4,
        "crf_lr": 2e-3,
        "batch_size": 16,
        "num_epochs": 20,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "dropout": 0.3,
    },
    {
        "id": 3,
        "name": "larger_lstm",
        "learning_rate": 1e-5,
        "lstm_lr": 1e-4,
        "crf_lr": 1e-3,
        "batch_size": 16,
        "num_epochs": 20,
        "lstm_hidden": 512,  # 更大
        "lstm_layers": 3,    # 更深
        "dropout": 0.3,
    },
    {
        "id": 4,
        "name": "higher_dropout",
        "learning_rate": 1e-5,
        "lstm_lr": 1e-4,
        "crf_lr": 1e-3,
        "batch_size": 16,
        "num_epochs": 20,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "dropout": 0.5,  # 更高dropout
    },
    {
        "id": 5,
        "name": "smaller_batch",
        "learning_rate": 1e-5,
        "lstm_lr": 1e-4,
        "crf_lr": 1e-3,
        "batch_size": 8,   # 更小batch
        "num_epochs": 20,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "dropout": 0.3,
    },
]

# 选择要运行的实验（根据时间限制调整）
START_EXP = 1
END_EXP = 3  # 建议3-5个实验

selected_configs = [c for c in HYPERPARAMETER_CONFIGS 
                    if START_EXP <= c['id'] <= END_EXP]

print(f"\n将运行 {len(selected_configs)} 个实验:")
for cfg in selected_configs:
    print(f"\n实验 #{cfg['id']}: {cfg['name']}")
    print(f"  LR: {cfg['learning_rate']}, LSTM_LR: {cfg['lstm_lr']}")
    print(f"  LSTM: {cfg['lstm_hidden']}x{cfg['lstm_layers']}")
    print(f"  Batch: {cfg['batch_size']}, Epochs: {cfg['num_epochs']}")

# 预估时间
est_time_per_exp = 1.5  # hours on P100
total_time = len(selected_configs) * est_time_per_exp
print(f"\n预计总耗时: {total_time:.1f} 小时")
print(f"⏰ Kaggle限制: 9-12 小时")

if total_time > 9:
    print("\n⚠️ 警告: 可能超时，建议减少实验数量")
```

### Cell 5: 创建实验运行脚本

```python
# 创建修改后的训练脚本，支持命令行参数
runner_code = '''
import os
import sys
import json
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, "/MCM")

# 导入训练脚本的函数
from exper.simple_ner_training import (
    SimpleNERModel, train_epoch, evaluate, 
    MNERDataset, compute_f1_metrics, extract_entities, compute_span_f1
)

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path("/MCM")

def run_experiment(config, exp_id):
    """运行单个实验"""
    print("=" * 80)
    print(f"实验 #{exp_id}: {config['name']}")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print("=" * 80)
  
    start_time = time.time()
  
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    # 数据加载
    print("\\n📂 加载数据...")
    train_dataset = MNERDataset(
        text_file=str(PROJECT_ROOT / 'data/MNER/twitter2015/train.txt'),
        image_dir=str(PROJECT_ROOT / 'data/img'),
        tokenizer_name='microsoft/deberta-v3-base',
        max_seq_length=128
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
  
    dev_dataset = MNERDataset(
        text_file=str(PROJECT_ROOT / 'data/MNER/twitter2015/dev.txt'),
        image_dir=str(PROJECT_ROOT / 'data/img'),
        tokenizer_name='microsoft/deberta-v3-base',
        max_seq_length=128
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
  
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(dev_dataset)} 样本")
  
    # 模型构建
    print("\\n🏗️ 构建模型...")
    model = SimpleNERModel(
        text_encoder_name='microsoft/deberta-v3-base',
        num_labels=9,
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout'],
        use_crf=True
    )
    model = model.to(device)
  
    # 优化器
    optimizer_grouped_parameters = [
        {'params': model.text_encoder.parameters(), 
         'lr': config['learning_rate']},
        {'params': model.bilstm.parameters(), 
         'lr': config['lstm_lr']},
        {'params': model.classifier.parameters(), 
         'lr': config['lstm_lr']},
        {'params': model.crf.parameters(), 
         'lr': config['crf_lr']},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
  
    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
  
    # 训练循环
    print("\\n🚀 开始训练...")
    best_dev_f1 = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'dev_loss': [],
        'dev_span_f1': [],
        'dev_token_f1': []
    }
  
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\\nEpoch {epoch}/{config['num_epochs']}")
      
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        history['train_loss'].append(train_loss)
      
        # 验证
        dev_loss, dev_metrics = evaluate(model, dev_loader, device, "Dev")
        history['dev_loss'].append(dev_loss)
        history['dev_span_f1'].append(dev_metrics['span_f1'])
        history['dev_token_f1'].append(dev_metrics['token_micro_f1'])
      
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Dev Loss: {dev_loss:.4f}")
        print(f"  Span F1: {dev_metrics['span_f1']:.2%}")
        print(f"  Token F1: {dev_metrics['token_micro_f1']:.2%}")
      
        # 保存最佳模型
        if dev_metrics['span_f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['span_f1']
            best_epoch = epoch
          
            save_path = f'/kaggle/working/best_model_exp{exp_id}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dev_f1': best_dev_f1,
                'config': config
            }, save_path)
            print(f"  ✓ 保存最佳模型 (F1={best_dev_f1:.2%})")
  
    elapsed = (time.time() - start_time) / 3600
  
    # 保存实验结果
    results = {
        'exp_id': exp_id,
        'config': config,
        'best_epoch': best_epoch,
        'best_dev_span_f1': best_dev_f1,
        'history': history,
        'elapsed_hours': elapsed
    }
  
    result_path = f'/kaggle/working/results_exp{exp_id}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\\n🎨 生成可视化与样例...")
    output_dir = Path("/kaggle/working")
    
    # 1. 导出预测样例 (jsonl)
    debug_limit = 2000
    records = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in dev_loader:
                if len(records) >= debug_limit: break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                _, logits = model(input_ids, attention_mask, labels)
                if model.use_crf:
                    preds = model.decode(input_ids, attention_mask)
                else:
                    preds = torch.argmax(logits, dim=-1)
                
                for i in range(input_ids.size(0)):
                    if len(records) >= debug_limit: break
                    
                    # 过滤padding
                    valid_mask = labels[i] != -100
                    gold_seq = labels[i][valid_mask].cpu().tolist()
                    pred_seq = preds[i][valid_mask].cpu().tolist()
                    
                    # 解码span
                    gold_spans = list(decode_mner(gold_seq))
                    pred_spans = list(decode_mner(pred_seq))
                    
                    records.append({
                        "exp_id": exp_id,
                        "gold_seq": gold_seq,
                        "pred_seq": pred_seq,
                        "gold_spans": gold_spans,
                        "pred_spans": pred_spans
                    })
        
        # 保存样例
        jsonl_path = output_dir / f"exp{exp_id}_samples.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        print(f"  ✓ 样例已导出: {jsonl_path}")
        
    except Exception as e:
        print(f"  ⚠️ 样例导出失败: {e}")

# 2. 生成 t-SNE (实体Token级聚类 - 严格筛除O标签)
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        all_entity_feats = []
        all_entity_labs = []
        
        # 限制Token数量，避免计算太慢
        max_tokens = 3000
        collected_tokens = 0
        
        # 标签映射: (B/I 统一为一个类别)
        # 1(B-PER), 2(I-PER) -> 0 (PER)
        # 3(B-ORG), 4(I-ORG) -> 1 (ORG)
        # 5(B-LOC), 6(I-LOC) -> 2 (LOC)
        # 7(B-MISC), 8(I-MISC)-> 3 (MISC)
        def map_label(l):
            return (l - 1) // 2
            
        label_names = {0: 'PER', 1: 'ORG', 2: 'LOC', 3: 'MISC'}
        
        with torch.no_grad():
            for batch in dev_loader:
                if collected_tokens >= max_tokens: break
                
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # 获取Token特征: (batch, seq, hidden)
                out = model.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                
                # 展平所有batch
                out_flat = out.view(-1, out.size(-1)) # (N, hidden)
                labels_flat = labels.view(-1)         # (N)
                
                # 筛选条件: 不是Padding (-100) 且 不是O (0)
                mask_entity = (labels_flat != -100) & (labels_flat != 0)
                
                if mask_entity.sum() > 0:
                    entity_feats = out_flat[mask_entity]
                    entity_labs = labels_flat[mask_entity]
                    
                    all_entity_feats.append(entity_feats.cpu().numpy())
                    all_entity_labs.append(entity_labs.cpu().numpy())
                    
                    collected_tokens += entity_feats.size(0)
                
        if len(all_entity_feats) > 0:
            feats = np.concatenate(all_entity_feats, axis=0)
            raw_labs = np.concatenate(all_entity_labs, axis=0)
            
            # 如果token过多，随机采样以加快t-SNE
            if feats.shape[0] > max_tokens:
                indices = np.random.choice(feats.shape[0], max_tokens, replace=False)
                feats = feats[indices]
                raw_labs = raw_labs[indices]
            
            # 映射标签到实体大类
            labs = np.array([map_label(l) for l in raw_labs])
            
            print(f"  t-SNE: 正在处理 {feats.shape[0]} 个实体Token...")
            tsne = TSNE(n_components=2, init="pca", learning_rate='auto', random_state=42)
            emb = tsne.fit_transform(feats)
            
            plt.figure(figsize=(10, 8))
            # 绘制散点图
            scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labs, cmap="tab10", s=20, alpha=0.7)
            
            # 添加图例
            handles, _ = scatter.legend_elements()
            # 确保图例标签对应正确
            legend_labels = [label_names.get(i, str(i)) for i in range(len(handles))]
            plt.legend(handles, legend_labels, title="Entity Type")
            
            plt.title(f"Exp {exp_id} Entity Token Clustering (No 'O')")
            plt.savefig(output_dir / f"exp{exp_id}_tsne_entity.png")
            plt.close()
            print(f"  ✓ t-SNE已保存: exp{exp_id}_tsne_entity.png")
        else:
            print("  ⚠️ 无实体Token用于 t-SNE (可能模型预测全为O或样本中无实体)")
        
    except Exception as e:
        print(f"  ⚠️ t-SNE生成失败: {e}")

    print(f"\\n✓ 实验 #{exp_id} 完成")
    print(f"  最佳Span F1: {best_dev_f1:.2%} (Epoch {best_epoch})")
    print(f"  耗时: {elapsed:.2f} 小时")
  
    return results

# 主函数
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--exp_id', type=int, required=True)
    args = parser.parse_args()
  
    with open(args.config_file, 'r') as f:
        config = json.load(f)
  
    run_experiment(config, args.exp_id)
'''

# 保存脚本
with open('/kaggle/working/run_ner_experiment.py', 'w') as f:
    f.write(runner_code)

print("✓ 实验运行脚本已创建: /kaggle/working/run_ner_experiment.py")
```

### Cell 6: 运行所有实验

```python
import subprocess
import time
import json

print("="*80)
print(f"开始运行 {len(selected_configs)} 个实验")
print("="*80)

all_results = []
total_start = time.time()

for cfg in selected_configs:
    print(f"\n{'='*80}")
    print(f"实验 #{cfg['id']}/{END_EXP}: {cfg['name']}")
    print(f"{'='*80}\n")
  
    # 保存配置
    config_file = f'/kaggle/working/config_exp{cfg["id"]}.json'
    with open(config_file, 'w') as f:
        json.dump(cfg, f, indent=2)
  
    # 运行实验
    exp_start = time.time()
  
    cmd = [
        'python', '/kaggle/working/run_ner_experiment.py',
        '--config_file', config_file,
        '--exp_id', str(cfg['id'])
    ]
  
    try:
        subprocess.run(cmd, check=True)
      
        # 读取结果
        result_file = f'/kaggle/working/results_exp{cfg["id"]}.json'
        with open(result_file, 'r') as f:
            result = json.load(f)
        all_results.append(result)
      
        exp_elapsed = (time.time() - exp_start) / 3600
        print(f"\n✓ 实验 #{cfg['id']} 完成 ({exp_elapsed:.2f}小时)")
      
    except Exception as e:
        print(f"\n❌ 实验 #{cfg['id']} 失败: {e}")
        continue
  
    # 清理GPU缓存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

total_elapsed = (time.time() - total_start) / 3600

print("\n" + "="*80)
print("所有实验完成！")
print("="*80)
print(f"总耗时: {total_elapsed:.2f} 小时")
print(f"完成实验: {len(all_results)}/{len(selected_configs)}")

# 保存汇总结果
summary = {
    'total_experiments': len(selected_configs),
    'completed': len(all_results),
    'total_hours': total_elapsed,
    'results': all_results
}

with open('/kaggle/working/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ 汇总结果已保存: /kaggle/working/summary.json")
```

### Cell 7: 结果分析

```python
import json
import pandas as pd
from pathlib import Path

print("="*80)
print("实验结果分析")
print("="*80)

# 读取汇总结果
with open('/kaggle/working/summary.json', 'r') as f:
    summary = json.load(f)

# 创建结果表格
results_data = []
for res in summary['results']:
     results_data.append({
        'ID': res['exp_id'],
        'Name': res['config']['name'],
        'LR': res['config']['learning_rate'],
        'LSTM_LR': res['config']['lstm_lr'],
        'CRF_LR': res['config']['crf_lr'],
        'LSTM_Hidden': res['config']['lstm_hidden'],
        'LSTM_Layers': res['config']['lstm_layers'],
        'Batch_Size': res['config']['batch_size'],
        'Dropout': res['config']['dropout'],
        'Best_Epoch': res['best_epoch'],
        'Span_F1': res['best_dev_span_f1'],  # 改为Span_F1
        'Time_Hours': res['elapsed_hours']
    })

df = pd.DataFrame(results_data)
df = df.sort_values('Span_F1', ascending=False)  # 改为Span_F1

print("\n📊 实验结果排名（按Span F1）:")
print(df.to_string(index=False))

# 最佳结果
best_exp = df.iloc[0]
print("\n" + "="*80)
print("🏆 最佳配置:")
print("="*80)
print(f"  实验ID: {int(best_exp['ID'])}")
print(f"  名称: {best_exp['Name']}")
print(f"  Span F1: {best_exp['Span_F1']:.2%}")
print(f"  最佳Epoch: {int(best_exp['Best_Epoch'])}")
print(f"\n超参数:")
print(f"  Learning Rate: {best_exp['LR']}")
print(f"  LSTM LR: {best_exp['LSTM_LR']}")
print(f"  CRF LR: {best_exp['CRF_LR']}")
print(f"  LSTM Hidden: {int(best_exp['LSTM_Hidden'])}")
print(f"  LSTM Layers: {int(best_exp['LSTM_Layers'])}")
print(f"  Batch Size: {int(best_exp['Batch_Size'])}")
print(f"  Dropout: {best_exp['Dropout']}")

# 保存结果表格
df.to_csv('/kaggle/working/results_table.csv', index=False)
print("\n✓ 结果表格已保存: /kaggle/working/results_table.csv")
```

### Cell 8: 可视化学习曲线

```python
import matplotlib.pyplot as plt
import json

print("="*80)
print("学习曲线可视化")
print("="*80)

# 读取所有实验结果
with open('/kaggle/working/summary.json', 'r') as f:
    summary = json.load(f)

# 创建子图
n_exp = len(summary['results'])
fig, axes = plt.subplots(n_exp, 2, figsize=(15, 5*n_exp))

if n_exp == 1:
    axes = axes.reshape(1, -1)

for i, res in enumerate(summary['results']):
    exp_id = res['exp_id']
    name = res['config']['name']
    history = res['history']
  
    epochs = range(1, len(history['train_loss']) + 1)
  
    # Loss曲线
    ax1 = axes[i, 0]
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['dev_loss'], label='Dev Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Exp #{exp_id}: {name} - Loss')
    ax1.legend()
    ax1.grid(True)
  
    # F1曲线
    ax2 = axes[i, 1]
    ax2.plot(epochs, history['dev_span_f1'], label='Span F1', marker='o')
    ax2.plot(epochs, history['dev_token_f1'], label='Token F1', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'Exp #{exp_id}: {name} - F1')
    ax2.legend()
    ax2.grid(True)

plt.tight_layout()
plt.savefig('/kaggle/working/learning_curves.png', dpi=150, bbox_inches='tight')
print("✓ 学习曲线已保存: /kaggle/working/learning_curves.png")

plt.show()
```

### Cell 9: 打包所有结果

```python
import shutil
from pathlib import Path

print("="*80)
print("打包实验结果")
print("="*80)

# 收集所有文件
output_files = list(Path('/kaggle/working').glob('*.json'))
output_files += list(Path('/kaggle/working').glob('*.pt'))
output_files += list(Path('/kaggle/working').glob('*.csv'))
output_files += list(Path('/kaggle/working').glob('*.png'))

print(f"\n找到 {len(output_files)} 个文件:")
for f in output_files:
    size = f.stat().st_size / (1024 * 1024)
    print(f"  - {f.name} ({size:.2f} MB)")

# 打包
print("\n正在打包...")
shutil.make_archive(
    '/kaggle/working/ner_experiments',
    'zip',
    '/kaggle/working'
)

zip_path = Path('/kaggle/working/ner_experiments.zip')
zip_size = zip_path.stat().st_size / (1024 * 1024)

print("\n" + "="*80)
print("✅ 打包完成！")
print("="*80)
print(f"📦 文件: ner_experiments.zip")
print(f"📏 大小: {zip_size:.2f} MB")
print(f"\n请在右侧 'Output' 标签页下载此文件")
print("\n⚠️ 下载完成后，请点击右上角 'Stop Session' 节省GPU配额")
```

---

## 📊 本地结果分析

下载 `ner_experiments.zip` 后：

```python
# 解压
import zipfile
import json
import pandas as pd

with zipfile.ZipFile('ner_experiments.zip', 'r') as zip_ref:
    zip_ref.extractall('ner_results/')

# 读取汇总
with open('ner_results/summary.json', 'r') as f:
    summary = json.load(f)

# 分析
results = []
for res in summary['results']:
    results.append({
        'Exp': res['exp_id'],
        'Name': res['config']['name'],
        'Span_F1': res['best_dev_span_f1'],
        'Epoch': res['best_epoch'],
        'Hours': res['elapsed_hours']
    })

df = pd.DataFrame(results)
print(df.sort_values('Span_F1', ascending=False))
```

---

## ⚙️ 超参数建议

### 学习率调整

```python
# 保守策略（推荐）
learning_rate: 1e-5
lstm_lr: 1e-4
crf_lr: 1e-3

# 激进策略
learning_rate: 2e-5
lstm_lr: 2e-4
crf_lr: 2e-3

# 微调策略
learning_rate: 5e-6
lstm_lr: 5e-5
crf_lr: 5e-4
```

### LSTM大小

```python
# 小模型（快速）
lstm_hidden: 128
lstm_layers: 1

# 中等模型（推荐）
lstm_hidden: 256
lstm_layers: 2

# 大模型（可能过拟合）
lstm_hidden: 512
lstm_layers: 3
```

### Batch Size

```python
# P100 (16GB) 推荐
batch_size: 16

# T4 (8GB) 推荐
batch_size: 8

# 显存不足时
batch_size: 4
```

### Dropout

```python
# 轻微正则化
dropout: 0.1

# 中等正则化（推荐）
dropout: 0.3

# 强正则化
dropout: 0.5
```

---

## ⏱️ 时间估算

### 单个实验耗时（Twitter2015，20 epochs）

| GPU  | Batch Size | 耗时/Epoch | 总耗时              |
| ---- | ---------- | ---------- | ------------------- |
| P100 | 16         | 3-4分钟    | **1-1.5小时** |
| P100 | 8          | 4-5分钟    | 1.5-2小时           |
| T4   | 16         | 4-5分钟    | 1.5-2小时           |
| T4   | 8          | 5-6分钟    | 2-2.5小时           |

### 多实验规划

| 实验数 | P100耗时  | T4耗时       | 建议            |
| ------ | --------- | ------------ | --------------- |
| 3个    | 3-4.5小时 | 4.5-7.5小时  | ✅ 推荐         |
| 5个    | 5-7.5小时 | 7.5-12.5小时 | ⚠️ 可能超时   |
| 8个    | 8-12小时  | 12-20小时    | ❌ 必超时，分批 |

**建议**：每个 Notebook 运行 3-5 个实验

---

## ⚠️ 常见问题

### 1. CUDA out of memory

```python
# 减小batch_size
batch_size: 8  # 或 4

# 或减小LSTM
lstm_hidden: 128
lstm_layers: 1
```

### 2. 找不到数据集

```bash
# 检查路径
ls /kaggle/input/
ls /kaggle/input/mcm-ner-training/

# 调整路径
'data_dir': Path('/kaggle/input/mcm-ner-training/data/MNER/twitter2015'),
'image_dir': Path('/kaggle/input/mcm-ner-training/data/img'),
```

### 3. torchcrf 导入失败

```python
!pip install torchcrf
```

### 4. 模型未保存

```python
# 确保目录存在
!mkdir -p /kaggle/working

# 检查保存路径
ls /kaggle/working/*.pt
```

---

## 📈 预期结果

### Twitter2015 MNER基准

- Token Micro F1: 60-70%
- **Span F1: 55-65%** ⭐ (主要指标)
- 最佳配置通常在 Epoch 15-20 达到

### 最佳配置经验值

```python
{
    "learning_rate": 1e-5,
    "lstm_lr": 1e-4,
    "crf_lr": 1e-3,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "batch_size": 16,
    "dropout": 0.3,
    "num_epochs": 20
}
```

---

## 🎯 快速检查清单

### 上传前

- [ ] 数据完整 (`data/MNER/twitter2015/`, `data/img/`)
- [ ] 预训练模型 (`downloaded_model/deberta-v3-base/`)
- [ ] 项目文件 (`tests/`, `datasets/`, `models/`)

### Kaggle配置

- [ ] GPU P100 或 T4
- [ ] 数据集已添加
- [ ] Internet开启（如需下载包）

### 运行中

- [ ] Cell 1: 项目路径正确
- [ ] Cell 2: 依赖安装成功
- [ ] Cell 3: GPU可用
- [ ] Cell 4: 超参数已配置
- [ ] Cell 6: 实验运行中

### 完成后

- [ ] Cell 7: 结果分析
- [ ] Cell 8: 可视化
- [ ] Cell 9: 下载 zip
- [ ] 停止 Session

---

## 🚀 进阶玩法

### 1. Grid Search（网格搜索）

```python
import itertools

# 定义搜索空间
lr_space = [5e-6, 1e-5, 2e-5]
lstm_hidden_space = [128, 256, 512]
dropout_space = [0.1, 0.3, 0.5]

# 生成所有组合
configs = []
for lr, hidden, dropout in itertools.product(
    lr_space, lstm_hidden_space, dropout_space
):
    configs.append({
        'learning_rate': lr,
        'lstm_hidden': hidden,
        'dropout': dropout,
        # ... 其他固定参数
    })

print(f"总共 {len(configs)} 个组合")
```

### 2. Random Search（随机搜索）

```python
import random

def sample_config():
    return {
        'learning_rate': random.choice([5e-6, 1e-5, 2e-5, 5e-5]),
        'lstm_lr': random.choice([5e-5, 1e-4, 2e-4]),
        'lstm_hidden': random.choice([128, 256, 384, 512]),
        'dropout': random.uniform(0.1, 0.5),
        # ...
    }

configs = [sample_config() for _ in range(10)]
```

### 3. 早停（Early Stopping）

```python
# 在训练循环中添加
patience = 3
no_improve_epochs = 0

for epoch in range(1, num_epochs + 1):
    # ... 训练和验证 ...
  
    if dev_f1 > best_f1:
        best_f1 = dev_f1
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
  
    if no_improve_epochs >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

Good luck with your NER experiments! 🎯

需要帮助？检查日志输出或参考主项目的 `tests/simple_ner_training.py`
