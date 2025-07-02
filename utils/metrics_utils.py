# utils/metrics_utils.py

import json
from continual.metrics import ContinualMetrics, compute_metrics_example

def compute_modality_metrics(train_info_path: str) -> dict:
    """
    读取 train_info.json，计算三组持续学习指标：
      1. 全任务 (all)
      2. 纯文本模态 (_1)
      3. 多模态 (_2)

    参数:
        train_info_path: str, train_info.json 的文件路径

    返回:
        dict 包含 keys: 'all', 'text', 'multi'，value 为各自的 {AA, AIA, FM, BWT}
    """
    # 1) 读取 JSON
    with open(train_info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    full_acc = info["acc_matrix"]
    sessions = info["sessions"]

    # 2) 放索引集合
    text_idxs  = [i for i, s in enumerate(sessions) if s["session_name"].endswith("_1")]
    multi_idxs = [i for i, s in enumerate(sessions) if s["session_name"].endswith("_2")]

    # 3) 通用：构建子集 ContinualMetrics
    def build_cm(sub_idxs):
        cm = ContinualMetrics()
        cm.acc_matrix = []
        for t, full_row_idx in enumerate(sub_idxs):
            full_row = full_acc[full_row_idx]
            # 取出该子集训练至第 t+1 轮时，对前 t+1 个子任务的 acc
            sub_row = [ full_row[j] for j in sub_idxs[:t+1] ]
            cm.acc_matrix.append(sub_row)
        return cm

    # 4) 计算三组指标
    results = {}
    # 全任务
    cm_all = ContinualMetrics()
    cm_all.acc_matrix = full_acc
    k_all = len(full_acc)
    results['all'] = compute_metrics_example(cm_all, k_all)

    # 纯文本模态
    cm_text = build_cm(text_idxs)
    k_text = len(text_idxs)
    results['text'] = compute_metrics_example(cm_text, k_text)

    # 多模态
    cm_multi = build_cm(multi_idxs)
    k_multi = len(multi_idxs)
    results['multi'] = compute_metrics_example(cm_multi, k_multi)

    return results


if __name__ == "__main__":
    # 例：直接运行查看结果
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_info", type=str, required=True, default="checkpoints/final3/train_info_twitter2015_none_m.json",
                        help="路径到 train_info.json")
    args = parser.parse_args()

    metrics = compute_modality_metrics(args.train_info)
    print("=== All Tasks Metrics ===")
    print(metrics['all'])
    print("=== Text-Only Tasks Metrics ===")
    print(metrics['text'])
    print("=== Multi-Modal Tasks Metrics ===")
    print(metrics['multi'])
