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

    注：acc_matrix 的结构要求是一个完整的正方形矩阵，
         其中 acc_matrix[i][j] 表示第 i+1 个任务训练后在任务 j+1 上的准确率。
         对于 j > i 的情况，表示0样本转移准确率。
    """

    def __init__(self):
        # acc_matrix[i][j] = a_{i+1, j+1}
        # 其中 i 表示第 i+1 个任务训练结束, j 表示第 j+1 个任务测试集准确率
        # 对于 j > i 的情况，表示0样本转移准确率
        self.acc_matrix = []
        
        # ✨ 新增：针对序列任务的两种指标矩阵
        self.chunk_f1_matrix = []  # 序列任务指标1：Chunk-level F1
        self.token_micro_f1_no_o_matrix = []  # 序列任务指标2：Token-level Micro F1 (no O)

    def update_acc_matrix(self, task_idx, performance_list, zero_shot_metrics=None, 
                         chunk_f1_list=None, token_micro_f1_no_o_list=None,
                         zero_shot_chunk_f1_metrics=None, zero_shot_token_micro_f1_no_o_metrics=None):
        """
        当学完第 task_idx (0-based) 个任务后，
        performance_list 表示对所有已学习任务的测试准确率。
        zero_shot_metrics 表示对后续任务的0样本准确率。
        
        ✨ 新增参数：
        chunk_f1_list: 序列任务的chunk-level F1列表
        token_micro_f1_no_o_list: 序列任务的token-level micro F1 (no O)列表
        zero_shot_chunk_f1_metrics: 0样本的chunk F1指标
        zero_shot_token_micro_f1_no_o_metrics: 0样本的token micro F1 (no O)指标
        
        如果 acc_matrix 还没有那么多行，则自动填充。
        """
        while len(self.acc_matrix) <= task_idx:
            self.acc_matrix.append([])
        while len(self.chunk_f1_matrix) <= task_idx:
            self.chunk_f1_matrix.append([])
        while len(self.token_micro_f1_no_o_matrix) <= task_idx:
            self.token_micro_f1_no_o_matrix.append([])

        # 构建完整的行：已学习任务的准确率 + 0样本转移准确率
        full_row = performance_list.copy()
        chunk_f1_row = chunk_f1_list.copy() if chunk_f1_list else []
        token_micro_f1_no_o_row = token_micro_f1_no_o_list.copy() if token_micro_f1_no_o_list else []
        
        # 如果有0样本指标，添加到行中
        if zero_shot_metrics:
            for future_session_name, metrics in zero_shot_metrics.items():
                if metrics and 'acc' in metrics:
                    full_row.append(metrics['acc'])
                else:
                    full_row.append(None)  # 表示没有0样本数据
        
        # 处理chunk_f1的0样本指标
        if zero_shot_chunk_f1_metrics:
            for future_session_name, metrics in zero_shot_chunk_f1_metrics.items():
                if metrics and 'chunk_f1' in metrics:
                    chunk_f1_row.append(metrics['chunk_f1'])
                else:
                    chunk_f1_row.append(None)
        
        # 处理token_micro_f1_no_o的0样本指标
        if zero_shot_token_micro_f1_no_o_metrics:
            for future_session_name, metrics in zero_shot_token_micro_f1_no_o_metrics.items():
                if metrics and 'token_micro_f1_no_o' in metrics:
                    token_micro_f1_no_o_row.append(metrics['token_micro_f1_no_o'])
                else:
                    token_micro_f1_no_o_row.append(None)
        
        # 更新或新建这一行
        self.acc_matrix[task_idx] = full_row
        self.chunk_f1_matrix[task_idx] = chunk_f1_row
        self.token_micro_f1_no_o_matrix[task_idx] = token_micro_f1_no_o_row

    def get_average_accuracy(self, k, matrix_type='acc'):
        """
        AA_k = 1/k * sum_{j=1 to k} a_{k,j}
        k: 任务数量(1-based)
        在 acc_matrix 里索引为 k-1
        
        ✨ 新增参数：
        matrix_type: 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
        """
        # 选择要使用的矩阵
        if matrix_type == 'chunk_f1':
            matrix = self.chunk_f1_matrix
        elif matrix_type == 'token_micro_f1_no_o':
            matrix = self.token_micro_f1_no_o_matrix
        else:
            matrix = self.acc_matrix
        
        if k - 1 < 0 or k - 1 >= len(matrix):
            return None
        row_k = matrix[k - 1]
        # 只取前 k 个任务的准确率（对角线及以下）
        # 边界检查：row_k可能比k短（有zero-shot数据）
        valid_k = min(k, len(row_k))
        # 过滤掉None值
        valid_accs = [acc for acc in row_k[:valid_k] if acc is not None]
        if not valid_accs:
            return None
        aa_k = np.mean(valid_accs)
        return aa_k

    def get_average_incremental_accuracy(self, k, matrix_type='acc'):
        """
        AIA_k = (1/k) * sum_{i=1}^{k} AA_i
        
        ✨ 新增参数：
        matrix_type: 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
        """
        # 选择要使用的矩阵
        if matrix_type == 'chunk_f1':
            matrix = self.chunk_f1_matrix
        elif matrix_type == 'token_micro_f1_no_o':
            matrix = self.token_micro_f1_no_o_matrix
        else:
            matrix = self.acc_matrix
            
        if k > len(matrix):
            return None
        aa_list = []
        for i in range(1, k + 1):
            aa_i = self.get_average_accuracy(i, matrix_type)
            if aa_i is None:
                return None
            aa_list.append(aa_i)
        return float(np.mean(aa_list))

    def get_forgetting_measure(self, k, matrix_type='acc'):
        """
        FM_k = 1/(k-1) * sum_{j=1}^{k-1} [ max_{i in [j, k-1]} (a_{i,j} - a_{k,j}) ]
        其中 a_{i,j} 表示学完第 i 个任务后，在第 j 个任务上的准确率。
        
        ✨ 新增参数：
        matrix_type: 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
        """
        # 选择要使用的矩阵
        if matrix_type == 'chunk_f1':
            matrix = self.chunk_f1_matrix
        elif matrix_type == 'token_micro_f1_no_o':
            matrix = self.token_micro_f1_no_o_matrix
        else:
            matrix = self.acc_matrix
            
        if k < 2 or k > len(matrix):
            return None
        fm_list = []
        for j in range(1, k):  # j=1..(k-1)，对应 matrix 列索引 j-1
            # 边界检查（重要！）
            if k - 1 >= len(matrix) or j - 1 >= len(matrix[k - 1]):
                continue
            a_kj = matrix[k - 1][j - 1]  # 当前模型在第 j 个任务上的准确率
            if a_kj is None:
                continue
            # 只遍历那些包含第 j 个测试结果的行：即 i 从 j-1 到 k-2
            max_ij = None
            for i in range(j - 1, k - 1):
                if i < len(matrix) and j - 1 < len(matrix[i]):
                    acc = matrix[i][j - 1]
                    if acc is not None:
                        if max_ij is None or acc > max_ij:
                            max_ij = acc
            if max_ij is not None:
                fm_list.append(max_ij - a_kj)
        if not fm_list:
            return None
        fm_k = np.mean(fm_list)
        return fm_k

    def get_bwt(self, k, matrix_type='acc'):
        """
        BWT_k = 1/(k-1) * sum_{j=1}^{k-1} (a_{k,j} - a_{j,j})
        
        ✨ 新增参数：
        matrix_type: 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
        """
        # 选择要使用的矩阵
        if matrix_type == 'chunk_f1':
            matrix = self.chunk_f1_matrix
        elif matrix_type == 'token_micro_f1_no_o':
            matrix = self.token_micro_f1_no_o_matrix
        else:
            matrix = self.acc_matrix
            
        if k < 2 or k > len(matrix):
            return None
        bwt_list = []
        for j in range(1, k):
            # 边界检查
            if k - 1 >= len(matrix) or j - 1 >= len(matrix[k - 1]):
                continue
            if j - 1 >= len(matrix) or j - 1 >= len(matrix[j - 1]):
                continue
            
            a_kj = matrix[k - 1][j - 1]  # 学完第 k 个任务后, 第 j 个任务准确率
            a_jj = matrix[j - 1][j - 1]  # 学完第 j 个任务后, 第 j 个任务准确率
            if a_kj is not None and a_jj is not None:
                bwt_list.append(a_kj - a_jj)
        if not bwt_list:
            return None
        return float(np.mean(bwt_list))

    def get_fwt(self, k, matrix_type='acc'):
        """
        FWT_k = 1/(k-1) * sum_{j=2}^{k} (a_{j-1,j} - a_{0,j})
        其中 a_{0,j} 表示在任务 j 上的随机性能（通常设为 1/num_classes）
        
        ✨ 新增参数：
        matrix_type: 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
        """
        # 选择要使用的矩阵
        if matrix_type == 'chunk_f1':
            matrix = self.chunk_f1_matrix
        elif matrix_type == 'token_micro_f1_no_o':
            matrix = self.token_micro_f1_no_o_matrix
        else:
            matrix = self.acc_matrix
            
        if k < 2 or k > len(matrix):
            return None
        fwt_list = []
        for j in range(2, k + 1):  # j=2..k，对应任务索引 j-1
            # 边界检查
            if j - 2 >= len(matrix):
                continue
            if j - 1 >= len(matrix[j - 2]):
                continue
            
            a_j1_j = matrix[j - 2][j - 1]  # 学完第 j-1 个任务后, 第 j 个任务准确率
            if a_j1_j is not None:
                # a_0_j 是随机性能，对于分类任务通常是 1/num_classes
                # 这里我们假设平均随机性能为 0.5，实际应用中可以根据具体任务调整
                a_0_j = 0.5  # 可以根据任务的具体类别数调整
                fwt_list.append(a_j1_j - a_0_j)
        if not fwt_list:
            return None
        return float(np.mean(fwt_list))

    def get_zero_shot_accuracy(self, k, matrix_type='acc'):
        """
        ZS_ACC_k = a_{k-1,k} 
        即学完第 k-1 个任务后，在第 k 个任务上的0样本准确率
        
        ✨ 新增参数：
        matrix_type: 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
        """
        # 选择要使用的矩阵
        if matrix_type == 'chunk_f1':
            matrix = self.chunk_f1_matrix
        elif matrix_type == 'token_micro_f1_no_o':
            matrix = self.token_micro_f1_no_o_matrix
        else:
            matrix = self.acc_matrix
            
        if k < 2 or k > len(matrix):
            return None
        # 边界检查
        if k - 2 < 0 or k - 2 >= len(matrix):
            return None
        if k - 1 >= len(matrix[k - 2]):
            return None
        
        acc = matrix[k - 2][k - 1]
        if acc is not None:
            return float(acc)
        return None


    def save_to_json(self, file_path):
        """
        将当前的 acc_matrix 存到 JSON，以备后续继续或分析。
        ✨ 新增：同时保存三个矩阵
        """
        data = {
            "acc_matrix": self.acc_matrix,
            "chunk_f1_matrix": self.chunk_f1_matrix,
            "token_micro_f1_no_o_matrix": self.token_micro_f1_no_o_matrix
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, file_path):
        """
        读取 acc_matrix。
        ✨ 新增：同时读取三个矩阵（向后兼容）
        """
        p = Path(file_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.acc_matrix = data.get("acc_matrix", [])
                # 向后兼容：如果旧的JSON没有这两个字段，初始化为空列表
                self.chunk_f1_matrix = data.get("chunk_f1_matrix", [])
                self.token_micro_f1_no_o_matrix = data.get("token_micro_f1_no_o_matrix", [])


def compute_metrics_example(cm: ContinualMetrics, task_count, matrix_type='acc'):
    """
    简单演示如何使用 ContinualMetrics 计算各项指标。
    task_count: 目前一共训练了多少个任务
    
    ✨ 新增参数：
    matrix_type: 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
    """
    AA_k = cm.get_average_accuracy(task_count, matrix_type)
    AIA_k = cm.get_average_incremental_accuracy(task_count, matrix_type)
    FM_k = cm.get_forgetting_measure(task_count, matrix_type)
    BWT_k = cm.get_bwt(task_count, matrix_type)
    FWT_k = cm.get_fwt(task_count, matrix_type)
    ZS_ACC_k = cm.get_zero_shot_accuracy(task_count, matrix_type)
    return {
        "AA": AA_k,
        "AIA": AIA_k,
        "FM": FM_k,
        "BWT": BWT_k,
        "FWT": FWT_k,
        "ZS_ACC": ZS_ACC_k
    }


def compute_multimodal_transfer_metrics(cm: ContinualMetrics, task_count, task_names=None, matrix_type='acc'):
    """
    专门针对多模态持续学习的转移指标计算。
    
    Args:
        cm: ContinualMetrics 实例
        task_count: 任务数量
        task_names: 任务名称列表，用于分析不同任务类型的转移效果
        matrix_type: ✨ 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
    
    Returns:
        dict: 包含各种转移指标的字典
    """
    # 基础持续学习指标
    basic_metrics = compute_metrics_example(cm, task_count, matrix_type)
    
    # 多模态特定的转移指标
    multimodal_metrics = {}
    
    # 选择要使用的矩阵
    if matrix_type == 'chunk_f1':
        matrix = cm.chunk_f1_matrix
    elif matrix_type == 'token_micro_f1_no_o':
        matrix = cm.token_micro_f1_no_o_matrix
    else:
        matrix = cm.acc_matrix
    
    if task_names and len(task_names) >= 2:
        # 分析任务间的转移效果
        text_tasks = ['masc', 'mate', 'mabsa']  # 文本相关任务
        ner_tasks = ['mner', 'mnre']  # 实体识别相关任务
        
        # 计算文本任务间的转移
        text_transfer = []
        for i, task in enumerate(task_names):
            if task in text_tasks and i < len(matrix):
                for j, other_task in enumerate(task_names):
                    if other_task in text_tasks and j < len(matrix[i]):
                        if i != j:  # 不同任务间
                            acc = matrix[i][j]
                            if acc is not None:
                                text_transfer.append(acc)
        
        if text_transfer:
            multimodal_metrics['text_task_transfer'] = float(np.mean(text_transfer))
        
        # 计算NER任务间的转移
        ner_transfer = []
        for i, task in enumerate(task_names):
            if task in ner_tasks and i < len(matrix):
                for j, other_task in enumerate(task_names):
                    if other_task in ner_tasks and j < len(matrix[i]):
                        if i != j:  # 不同任务间
                            acc = matrix[i][j]
                            if acc is not None:
                                ner_transfer.append(acc)
        
        if ner_transfer:
            multimodal_metrics['ner_task_transfer'] = float(np.mean(ner_transfer))
        
        # 计算跨任务类型的转移（文本任务到NER任务，或反之）
        cross_type_transfer = []
        for i, task in enumerate(task_names):
            if i < len(matrix):
                for j, other_task in enumerate(task_names):
                    if j < len(matrix[i]):
                        if ((task in text_tasks and other_task in ner_tasks) or 
                            (task in ner_tasks and other_task in text_tasks)):
                            acc = matrix[i][j]
                            if acc is not None:
                                cross_type_transfer.append(acc)
        
        if cross_type_transfer:
            multimodal_metrics['cross_type_transfer'] = float(np.mean(cross_type_transfer))
    
    # 合并所有指标
    all_metrics = {**basic_metrics, **multimodal_metrics}
    
    return all_metrics


def analyze_task_similarity_transfer(cm: ContinualMetrics, task_names, matrix_type='acc'):
    """
    分析任务相似性与转移效果的关系。
    
    Args:
        cm: ContinualMetrics 实例
        task_names: 任务名称列表
        matrix_type: ✨ 'acc', 'chunk_f1', 'token_micro_f1_no_o' 三选一
    
    Returns:
        dict: 任务相似性分析结果
    """
    # 选择要使用的矩阵
    if matrix_type == 'chunk_f1':
        matrix = cm.chunk_f1_matrix
    elif matrix_type == 'token_micro_f1_no_o':
        matrix = cm.token_micro_f1_no_o_matrix
    else:
        matrix = cm.acc_matrix
        
    if len(task_names) < 2 or len(matrix) < 1:
        return {}
    
    # 定义任务相似性矩阵（基于任务类型）
    task_types = {
        'masc': 'sentiment',
        'mate': 'extraction', 
        'mabsa': 'sentiment',
        'mner': 'entity',
        'mnre': 'relation'
    }
    
    # 计算同类型任务间的转移
    same_type_transfer = []
    different_type_transfer = []
    
    for i, task_i in enumerate(task_names):
        if i >= len(matrix):
            continue
        for j, task_j in enumerate(task_names):
            if j >= len(matrix[i]):
                continue
            if i != j:
                transfer_score = matrix[i][j]
                if transfer_score is not None:
                    type_i = task_types.get(task_i, 'unknown')
                    type_j = task_types.get(task_j, 'unknown')
                    
                    if type_i == type_j:
                        same_type_transfer.append(transfer_score)
                    else:
                        different_type_transfer.append(transfer_score)
    
    analysis = {}
    if same_type_transfer:
        analysis['same_type_transfer'] = float(np.mean(same_type_transfer))
    if different_type_transfer:
        analysis['different_type_transfer'] = float(np.mean(different_type_transfer))
    
    return analysis
