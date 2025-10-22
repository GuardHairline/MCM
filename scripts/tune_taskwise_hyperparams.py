import optuna
import json
import subprocess
import os

# CONFIG_PATH = 'scripts/configs/server_twitter2015_none_label_emb.json'
CONFIG_PATH = 'scripts/configs/local_200_none_label_emb.json'
EPOCHS = 2  # 每次trial训练轮数，可调小加速
TASKS = ['masc', 'mate', 'mner', 'mabsa']


def objective(trial, task_name):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.1, 0.9)

    # 读取原始config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 只保留当前任务的text_only+label_emb
    filtered_tasks = []
    for t in config['tasks']:
        if t['task_name'] == task_name and t['mode'] == 'text_only' and t.get('use_label_embedding', False):
            t['lr'] = lr
            t['weight_decay'] = weight_decay
            t['gamma'] = gamma
            t['epochs'] = EPOCHS
            filtered_tasks.append(t)
    if not filtered_tasks:
        print(f"[WARN] No text_only+label_emb task found for {task_name}")
        return float('inf')
    config['tasks'] = filtered_tasks
    config['total_tasks'] = 1

    tmp_config_path = f'scripts/configs/tmp_{task_name}_trial_{trial.number}.json'
    with open(tmp_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    cmd = [
        'python', '-m', 'scripts.train_with_zero_shot',
        '--config', tmp_config_path,
        '--start_task', '0',
        '--end_task', '1'
    ]
    print(f"[Optuna] Running {task_name} trial {trial.number}: lr={lr}, weight_decay={weight_decay}, gamma={gamma}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(result.stdout)
        train_info_path = config['global_params']['train_info_json']
        with open(train_info_path, 'r', encoding='utf-8') as f:
            train_info = json.load(f)
        dev_metrics = train_info['sessions'][0]['details']['final_dev_metrics']
        val_loss = dev_metrics.get('loss', None)
        val_acc = dev_metrics.get('acc', None)
        if val_loss is not None:
            return val_loss
        elif val_acc is not None:
            return -val_acc
        else:
            return float('inf')
    except Exception as e:
        print(f"[Optuna] Trial failed: {e}")
        return float('inf')
    finally:
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)

if __name__ == '__main__':
    for task in TASKS:
        print(f'==== Tuning {task} (text_only + label_emb) ====')
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, task), n_trials=20)
        print(f'Best params for {task}:', study.best_params) 