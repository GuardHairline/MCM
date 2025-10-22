import optuna
import subprocess
import json
import os

# 配置
CONFIG_PATH = 'scripts/configs/local_200_none_label_emb.json'  # 你的config路径
LOG_PATH = 'scripts/tune_trial_log.json'  # 每次trial的日志
EPOCHS = 2  # 每次trial训练轮数，可调小加速

# 目标函数

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    # 生成临时config，覆盖lr和weight_decay
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    for task in config['tasks']:
        if task.get('use_label_embedding', False):
            task['lr'] = lr
            task['weight_decay'] = weight_decay
            task['epochs'] = EPOCHS
    tmp_config_path = f'scripts/configs/tmp_trial_{trial.number}.json'
    with open(tmp_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 调用训练脚本（只训练第一个任务）
    cmd = [
        'python', '-m', 'scripts.train_with_zero_shot',
        '--config', tmp_config_path,
        '--start_task', '0',
        '--end_task', '1'
    ]
    print(f"[Optuna] Running trial {trial.number}: lr={lr}, weight_decay={weight_decay}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(result.stdout)
        # 读取训练结果
        train_info_path = config['global_params']['train_info_json']
        with open(train_info_path, 'r', encoding='utf-8') as f:
            train_info = json.load(f)
        # 取第一个任务的dev loss/acc
        dev_metrics = train_info['sessions'][0]['details']['final_dev_metrics']
        val_loss = dev_metrics.get('loss', None)
        val_acc = dev_metrics.get('acc', None)
        # 记录日志
        with open(LOG_PATH, 'a', encoding='utf-8') as logf:
            logf.write(json.dumps({
                'trial': trial.number,
                'lr': lr,
                'weight_decay': weight_decay,
                'val_loss': val_loss,
                'val_acc': val_acc
            }) + '\n')
        # 优先最小化loss，否则最大化acc
        if val_loss is not None:
            return val_loss
        elif val_acc is not None:
            return -val_acc
        else:
            return float('inf')
    except Exception as e:
        print(f"[Optuna] Trial {trial.number} failed: {e}")
        return float('inf')
    finally:
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print('Best params:', study.best_params) 