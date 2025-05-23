# continual/metrics.py
import json
import numpy as np
from pathlib import Path


class ContinualMetrics:
    """
    用来存储并计算多任务的持续学习指标，包括:
      - 平均准确率: Average Accuracy (AA)
      - 平均增量准确率: Average Incremental Accuracy (AIA)
      - 遗忘度量: Forgetting Measure (FM)
      - 后向转移: Backward Transfer (BWT)
      - 瞬时度量: Intransigence Measure (IM)
      - 前向转移: Forward Transfer (FWT)
    这里示例只演示 AA, AIA, FM, BWT 的实现，其它可自行补全。

    注：acc_matrix 的结构要求在完成第 k 个任务后，
         第 k 行包含 k 个元素：[a11, a12, ..., a_{kk}]
         即 a_{ij} 表示第 i 次任务训练后在任务 j 上的准确率。
    """

    def __init__(self):
        # acc_matrix[k][j] = a_{k+1, j+1}
        # 其中 k 表示第 k+1 个任务训练结束, j 表示第 j+1 个任务测试集准确率
        self.acc_matrix = []

    def update_acc_matrix(self, task_idx, performance_list):
        """
        当学完第 task_idx (0-based) 个任务后，
        performance_list 表示对所有任务（从1到 task_idx+1以及旧任务）的测试准确率。
        如果 acc_matrix 还没有那么多行，则自动填充。
        """
        while len(self.acc_matrix) <= task_idx:
            self.acc_matrix.append([])

        # 更新或新建这一行
        self.acc_matrix[task_idx] = performance_list

    def get_average_accuracy(self, k):
        """
        AA_k = 1/k * sum_{j=1 to k} a_{k,j}
        k: 任务数量(1-based)
        在 acc_matrix 里索引为 k-1
        """
        if k - 1 < 0 or k - 1 >= len(self.acc_matrix):
            return None
        row_k = self.acc_matrix[k - 1]
        # 只取前 k 个任务的准确率
        if len(row_k) < k:
            return None
        aa_k = np.mean(row_k[:k])
        return aa_k

    def get_average_incremental_accuracy(self, k):
        """
        AIA_k = (1/k) * sum_{i=1}^{k} AA_i
        """
        if k > len(self.acc_matrix):
            return None
        aa_list = []
        for i in range(1, k + 1):
            aa_i = self.get_average_accuracy(i)
            if aa_i is None:
                return None
            aa_list.append(aa_i)
        return float(np.mean(aa_list))

    def get_forgetting_measure(self, k):
        """
        FM_k = 1/(k-1) * sum_{j=1}^{k-1} [ max_{i in [j, k-1]} (a_{i,j} - a_{k,j}) ]
        其中 a_{i,j} 表示学完第 i 个任务后，在第 j 个任务上的准确率。
        注意：由于 acc_matrix 的每一行只包含前 i 个任务的准确率，
              对于第 j 个任务，只有行 i 满足 i >= j-1（即 i+1 >= j）才有该值。
        """
        if k < 2 or k > len(self.acc_matrix):
            return None
        fm_list = []
        for j in range(1, k):  # j=1..(k-1)，对应 acc_matrix 列索引 j-1
            a_kj = self.acc_matrix[k - 1][j - 1]  # 当前模型在第 j 个任务上的准确率（来自第 k 行）
            # 只遍历那些包含第 j 个测试结果的行：即 i 从 j-1 到 k-2（对应任务 1~k-1 中有该列的行）
            max_ij = max(self.acc_matrix[i][j - 1] for i in range(j - 1, k - 1))
            fm_list.append(max_ij - a_kj)
        fm_k = np.mean(fm_list)
        return fm_k

    def get_bwt(self, k):
        """
        BWT_k = 1/(k-1) * sum_{j=1}^{k-1} (a_{k,j} - a_{j,j})
        """
        if k < 2 or k > len(self.acc_matrix):
            return None
        bwt_list = []
        for j in range(1, k):
            a_kj = self.acc_matrix[k - 1][j - 1]  # 学完第 k 个任务后, 第 j 个任务准确率
            a_jj = self.acc_matrix[j - 1][j - 1]  # 学完第 j 个任务后, 第 j 个任务准确率
            bwt_list.append(a_kj - a_jj)
        return float(np.mean(bwt_list))

    # 如果要实现 IM, FWT, 同理写对应函数

    def save_to_json(self, file_path):
        """
        将当前的 acc_matrix 存到 JSON，以备后续继续或分析。
        """
        data = {
            "acc_matrix": self.acc_matrix
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, file_path):
        """
        读取 acc_matrix。
        """
        p = Path(file_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.acc_matrix = data["acc_matrix"]


def compute_metrics_example(cm: ContinualMetrics, task_count):
    """
    简单演示如何使用 ContinualMetrics 计算各项指标。
    task_count: 目前一共训练了多少个任务
    """
    AA_k = cm.get_average_accuracy(task_count)
    AIA_k = cm.get_average_incremental_accuracy(task_count)
    FM_k = cm.get_forgetting_measure(task_count)
    BWT_k = cm.get_bwt(task_count)
    return {
        "AA": AA_k,
        "AIA": AIA_k,
        "FM": FM_k,
        "BWT": BWT_k
    }
