import itertools, copy, json, os, datetime, argparse
from modules.parser import create_train_parser, validate_args
from modules.train_refactored import train
from utils.logging import setup_logger

def run_single(args, max_epoch=5):
    # 将 epoch 改为5
    args.epochs = max_epoch
    # 禁用保存模型文件，缩短实验时间
    args.save_model = False
    
    # 创建一个简单的logger
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    result = train(args, logger=logger)
    # 假设 result 包含 dev_loss/acc
    return result.get('details', {}).get('final_dev_metrics', {}).get('acc', 0.0)

def save_progress(results, best_params, best_score, outfile):
    data = {'results': results, 'best': {'params': best_params, 'score': best_score}}
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def search_hyperparams(base_args, search_space, outfile):
    best_score = -1
    best_params = None
    results = []
    combos = list(itertools.product(*search_space.values()))
    for i, (lr, wd, gamma, emb_dim, sim_w) in enumerate(combos):
        args = copy.deepcopy(base_args)
        args.lr = lr
        args.weight_decay = wd
        args.gamma = gamma
        args.label_emb_dim = emb_dim
        args.similarity_weight = sim_w
        acc = run_single(args, max_epoch=5)
        results.append({'params': [lr, wd, gamma, emb_dim, sim_w], 'acc': acc})
        if acc > best_score:
            best_score = acc
            best_params = [lr, wd, gamma, emb_dim, sim_w]
        save_progress(results, best_params, best_score, outfile)  # 实时保存
    return best_params, best_score

def fine_tune(base_args, best_params, tune_space, outfile):
    lr_base, wd_base, gamma_base, emb_base, sim_base = best_params
    fine_space = {
        'lr': [lr_base * x for x in tune_space['lr']],
        'wd': [wd_base * x for x in tune_space['wd']],
        'gamma': [gamma_base * x for x in tune_space['gamma']],
        'emb_dim': [emb_base],  # 固定或微调
        'sim_weight': [sim_base * x for x in tune_space['sim_weight']]
    }
    return search_hyperparams(base_args, fine_space, outfile)

def main():
    # 预定义超参范围
    coarse_space = {
        'lr': [1e-3, 1e-4, 1e-5, 3e-5, 5e-5],
        'wd': [0.0, 0.01, 0.001, 0.0001],
        'gamma': [0.5, 0.7, 0.9, 1.0],
        'emb_dim': [128, 256],
        'sim_weight': [0.0, 0.05, 0.1]
    }
    tune_factors = {
        'lr': [0.5, 1.0, 2.0],
        'wd': [0.5, 1.0, 2.0],
        'gamma': [0.8, 1.0, 1.2],
        'sim_weight': [0.5, 1.0, 1.5]
    }
    tasks = ['mabsa', 'mner', 'mate', 'masc']
    modes = ['text_only', 'multimodal']
    
    # 创建参数解析器
    parser = create_train_parser()
    
    for task in tasks:
        for mode in modes:
            # 根据任务设置正确的数据文件路径
            if task == 'mabsa':
                data_dir = './data/MASC'
            elif task == 'mner':
                data_dir = './data/MNER'
            elif task == 'mate':
                data_dir = './data/MASC'
            elif task == 'masc':
                data_dir = './data/MASC'
            else:
                data_dir = './data/MASC'
            
            # 根据任务设置num_labels
            if task in ['mate', 'mner', 'mabsa']:
                num_labels = 7  # 序列标注任务的标签数
            elif task == 'masc':
                num_labels = 3  # 句子分类任务的标签数
            else:
                num_labels = 3  # 默认值
            
            # 生成基本参数
            cmd_args = [
                '--task_name', task,
                '--session_name', f'{task}_search',
                '--mode', mode,
                '--use_label_embedding',
                '--use_hierarchical_head',
                '--num_labels', str(num_labels),  # 添加num_labels参数
                '--data_dir', './data',
                '--dataset_name', 'twitter2015',  # 添加必需的dataset_name
                '--train_info_json', f'./train_info_{task}_search.json',  # 添加必需的train_info_json
                '--output_model_path', f'./checkpoints/{task}_search_model.pth',  # 添加必需的output_model_path
                '--train_text_file', f'{data_dir}/twitter2015/train.txt',  # 根据任务设置正确的数据文件路径
                '--dev_text_file', f'{data_dir}/twitter2015/dev.txt',      # 根据任务设置正确的数据文件路径
                '--test_text_file', f'{data_dir}/twitter2015/test.txt',    # 根据任务设置正确的数据文件路径
                '--image_dir', './data/img',                                  # 根据任务设置正确的图像目录
                '--epochs', '5',
            ]
            args = parser.parse_args(cmd_args)
            validate_args(args)
            
            outfile = f'hyperparam_search_{task}_{mode}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            # 第一步粗搜索
            best_params, best_score = search_hyperparams(args, coarse_space, outfile)
            # 第二步微调
            fine_best, fine_score = fine_tune(args, best_params, tune_factors, outfile.replace('.json','_fine.json'))

if __name__ == '__main__':
    main()
