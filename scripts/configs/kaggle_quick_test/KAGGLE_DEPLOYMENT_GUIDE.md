# Kaggle 快速回归部署指南（分离模式，200样本/epoch=2）

严格遵循 Kaggle 限制：先检查分离数据集，再复制到可写目录，安装依赖，逐个运行现成配置（不可重新生成，不用 run_all.sh），输出在 `/kaggle/working` 并打包下载。

## 1. 环境检查（Notebook Cell 1）
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

## 2. 安装依赖（Cell 2）
```bash
pip install -q -r requirements_kaggle.txt
```

## 3. GPU/设备检查（Cell 3，可选）
```python
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

## 4. 运行所有配置（Cell 4）
> 配置已随 mcm-code 提供，无需再运行生成脚本；**不要用 run_all.sh**。确保输出写在 `/kaggle/working`。
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

## 5. 收集与打包（Cell 5）
```python
import zipfile, shutil
from pathlib import Path
root = Path('/kaggle/working/MCM')
out_root = Path('/kaggle/working/quick_test_results')
out_root.mkdir(parents=True, exist_ok=True)
for sd in ['checkpoints/kaggle_quick', 'checkpoints/quick_test']:
    src = root/sd
    if src.exists():
        dest = out_root/src.name
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print('✓ 拷贝', src, '->', dest)
zip_path = Path('/kaggle/working/quick_test_results.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in out_root.rglob('*'):
        zf.write(p, p.relative_to(out_root))
print('✓ 打包完成:', zip_path)
print('请在 Kaggle 文件面板下载 quick_test_results.zip')
```

## 6. 说明
- 任务序列固定：masc(text)→mate(text)→mner(text)→mabsa(text)→masc(mm)→mate(mm)→mner(mm)→mabsa(mm)。
- 配置包含共享头方案（text/mm 同任务共享 head_key，共4头）与 `none_8heads` 独立头对照。
- 若显存不足，可在 Cell 4 过滤 `configs` 仅跑部分方法。 
