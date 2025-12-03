# Kaggle 快速回归部署指南（分离模式，200样本/epoch=2）

严格遵循 Kaggle 流程：检查分离数据集 → 复制到可写目录 → 安装依赖 → 逐个运行现成配置 → 汇总并打包结果（全部放在 `/kaggle/working` 可下载）。

## 1) 环境检查与复制（Cell 1）
```python
import os, shutil
from pathlib import Path
print('='*80); print('环境检查'); print('='*80)
print('\n可用数据集:')
for d in os.listdir('/kaggle/input'):
    print(' -', d)

code_src = Path('/kaggle/input/mcm-code')
data_src = Path('/kaggle/input/mcm-data')
assert code_src.exists(), '缺少 mcm-code 数据集'
assert data_src.exists(), '缺少 mcm-data 数据集'
assert (code_src/'scripts/configs/kaggle_quick_test').exists(), '缺少 quick_test 配置目录'

work = Path('/kaggle/working/MCM')
if work.exists():
    shutil.rmtree(work)
shutil.copytree(code_src, work)
shutil.copytree(data_src/'data', work/'data')
if (data_src/'downloaded_model').exists():
    shutil.copytree(data_src/'downloaded_model', work/'downloaded_model')
os.chdir(work)
print('✓ 已复制到:', work)
print('项目内容示例:', sorted([p.name for p in work.iterdir()])[:12])
```

## 2) 安装依赖（Cell 2）
```bash
pip install -q -r requirements_kaggle.txt
```

## 3) GPU/设备检查（Cell 3，可选）
```python
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

## 4) 运行全部配置（Cell 4）
> 配置已随 mcm-code 提供，不要再生成；不要用 run_all.sh。确保输出在 `/kaggle/working`。
```python
import subprocess, sys
from pathlib import Path
root = Path('/kaggle/working/MCM')
configs = sorted(root.glob('scripts/configs/kaggle_quick_test/*/*.json'))
print('待运行配置数:', len(configs))
for cfg in configs:
    print('\n=== Running', cfg)
    subprocess.check_call([sys.executable, '-m', 'scripts.train_with_zero_shot', '--config', str(cfg)], cwd=root)
```

## 5) 汇总结果并打包（Cell 5）
- 结果路径前统一加 `/kaggle/working`：检查 `checkpoints/kaggle_quick` 与 `checkpoints/quick_test`。
- 汇总所有 `train_info_*.json` 指标为表格，存 `quick_test_summary.csv`，同时打印日志。
```python
import json, zipfile, shutil
from pathlib import Path
import pandas as pd
root = Path('/kaggle/working/MCM')
out_root = Path('/kaggle/working/quick_test_results')
out_root.mkdir(parents=True, exist_ok=True)

# 收集目录
collected = []
for sd in ['checkpoints/kaggle_quick', 'checkpoints/quick_test']:
    src = root/sd
    if src.exists():
        dest = out_root/src.name
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print('✓ 拷贝', src, '->', dest)
        collected.append(dest)

# 汇总 train_info
rows = []
for d in collected:
    for info in d.rglob('train_info_*.json'):
        with open(info, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 从 sessions 中取最后一项（当前任务）指标
        if data.get('sessions'):
            last = data['sessions'][-1]
            rows.append({
                'file': info.as_posix(),
                'session': last.get('session_name'),
                'task': last.get('task_name'),
                'acc': last.get('acc'),
                'chunk_f1': last.get('chunk_f1'),
                'token_micro_f1_no_o': last.get('token_micro_f1_no_o'),
            })
summary = pd.DataFrame(rows)
summary_path = out_root/'quick_test_summary.csv'
summary.to_csv(summary_path, index=False)
print('\n=== 汇总结果 ===')
print(summary)
print('保存至:', summary_path)

# 打包
zip_path = Path('/kaggle/working/quick_test_results.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in out_root.rglob('*'):
        zf.write(p, p.relative_to(out_root))
print('✓ 打包完成:', zip_path)
print('请在 Kaggle 文件面板下载 quick_test_results.zip')
```

## 6) 说明
- 任务序列：masc(text)→mate(text)→mner(text)→mabsa(text)→masc(mm)→mate(mm)→mner(mm)→mabsa(mm)。
- 配置包含共享头方案与独立头对照 `none_8heads`。
- 若显存不足，可在运行 Cell 4 前过滤 `configs` 只跑部分。 
