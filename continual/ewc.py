# continual/ewc_new.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import logging
logger = logging.getLogger("ewc")

class MultiTaskEWC:
    """
    多任务版本EWC：每学完一个任务后，估计 Fisher 并保存到磁盘；
    下次学新任务时，加载所有旧任务 Fisher 与旧参数进行 penalty。
    不再需要旧数据集。
    """
    def __init__(self, model, current_task_name, session_name, num_labels, ewc_lambda, ewc_dir="ewc_params"):
        """
        :param model: 当前要学习的模型
        :param current_task_name: 当前任务的名称（字符串）
        :param session_name: 当前任务的会话名称（字符串）
        :param num_labels: 当前任务的标签数量
        :param ewc_lambda: EWC损失项系数
        :param ewc_dir: 存放 fisher/param 的目录
        """
        self.task_count = None
        self.model = model
        self.task_name = current_task_name
        self.session_name = session_name
        self.num_labels = num_labels
        self.ewc_lambda = ewc_lambda
        self.ewc_dir = ewc_dir

        # 用来累加所有旧任务的fisher和optpar
        # fisher_all, optpar_all => {param_name: aggregated_value}
        # 这里仅在 penalty() 时构建
        self.fisher_all = {}
        self.optpar_all = {}

    def load_all_previous_tasks(self, train_info):
        """
        加载所有历史任务的 Fisher 和 optpar 信息，
        只会加载之前保存的任务，如果是第一次任务，这里不会加载任何内容。
        """
        try:
            # 如果没有历史任务信息文件（train_info），则跳过
            if "sessions" not in train_info:
                logger.warning("No previous sessions in train_info, skipping loading of Fisher and optpar.")
                return

            self.fisher_all = {}
            self.optpar_all = {}
            logger.info("Loading Fisher and optpar for previous tasks...")
            self.task_count = len(train_info["sessions"])
            for session in train_info["sessions"]:
                fisher_file_path = session["fisher_file"]
                if os.path.exists(fisher_file_path):
                    logger.info(f"Loading Fisher and optpar from: {fisher_file_path}")
                    checkpoint = torch.load(fisher_file_path)
                    fisher_dict = checkpoint["fisher"]
                    optpar_dict = checkpoint["optpar"]

                    for n, p in fisher_dict.items():
                        # 跳过所有和 task-specific classifier 相关的参数
                        if "head.classifier" in n:
                            continue
                        p_tensor = torch.tensor(p) if not isinstance(p, torch.Tensor) else p
                        if n not in self.fisher_all:
                            self.fisher_all[n] = p_tensor
                        else:
                            self.fisher_all[n] += p_tensor

                    for n, p in optpar_dict.items():
                        # 跳过所有和 task-specific classifier 相关的参数
                        if "head.classifier" in n:
                            continue
                        p_tensor = torch.tensor(p) if not isinstance(p, torch.Tensor) else p
                        if n not in self.optpar_all:
                            self.optpar_all[n] = p_tensor
                        else:
                            self.optpar_all[n] += p_tensor
                    T = len(train_info["sessions"])
                    for n in self.fisher_all:
                        self.fisher_all[n] = self.fisher_all[n] / T
                        self.optpar_all[n] = self.optpar_all[n] / T
                    logger.info("Normalized fisher/optpar by %d tasks", T)
                else:
                    logger.warning(f"Fisher file for session {session['session_name']} not found at {fisher_file_path}")

            logger.info("Finished loading Fisher and optpar from previous tasks.")
        except Exception as e:
            logger.error(f"Error loading Fisher and optpar: {e}")
            raise e

    def penalty(self, model):
        """
        计算 EWC penalty = sum_{n} fisher_all[n] * (p - optpar_all[n])^2
        如果 self.fisher_all/optpar_all 为空，则返回 0。
        """
        if not self.fisher_all or not self.optpar_all:
            logger.warning("Fisher 或 optpar 为空，返回 penalty = 0.")
            return torch.tensor(0., device=next(model.parameters()).device)

        # 动态 λ： λ_t = λ_0 / (1 + alpha * (task_count-1))
        lambda_dyn = self.ewc_lambda / (1 + 0.5 * (self.task_count - 1))

        device = next(model.parameters()).device
        loss_ewc = torch.tensor(0., device=device)

        current_task_num_labels = self.num_labels  # 当前任务的标签数量
        max_num_labels = 23  # 假设所有任务中最大的标签数量为 23

        for n, p in model.named_parameters():
            if n in self.fisher_all:
                fisher_val = self.fisher_all[n].to(device)
                optpar_val = self.optpar_all[n].to(device)
                # 如果参数名称中包含 'classifier'，表示其尺寸直接依赖于 num_labels，跳过 EWC 正则化
                if "classifier" in n:
                    continue
                # 如果该参数属于任务头，则可能需要进行标签维度的 padding/裁剪
                if "head" in n:
                    # 针对 1D 参数（比如 bias），只需按照 0 维判断
                    if fisher_val.dim() == 1:
                        if fisher_val.size(0) < max_num_labels:
                            pad_size = max_num_labels - fisher_val.size(0)
                            fisher_val = F.pad(fisher_val, (0, pad_size))  # 在末尾 pad zeros
                            optpar_val = F.pad(optpar_val, (0, pad_size))
                        # 只保留当前任务的标签部分
                        fisher_val = fisher_val[:current_task_num_labels]
                        optpar_val = optpar_val[:current_task_num_labels]
                    # 针对多维参数（假设第一维为标签数），进行 padding 与裁剪
                    elif fisher_val.dim() >= 2:
                        if fisher_val.size(0) < max_num_labels:
                            pad_rows = max_num_labels - fisher_val.size(0)
                            # 构造与 fisher_val 除第一维外形状相同的零张量
                            zeros = torch.zeros((pad_rows,) + fisher_val.shape[1:], device=device)
                            fisher_val = torch.cat([fisher_val, zeros], dim=0)
                            zeros2 = torch.zeros((pad_rows,) + optpar_val.shape[1:], device=device)
                            optpar_val = torch.cat([optpar_val, zeros2], dim=0)
                        # 裁剪到当前任务的标签数
                        fisher_val = fisher_val[:current_task_num_labels]
                        optpar_val = optpar_val[:current_task_num_labels]
                # 对于不属于任务头的参数，我们不作 padding直接使用保存的 Fisher 与 optpar

                # 检查处理后的形状是否与当前参数匹配
                if fisher_val.shape != p.shape:
                    # logger.warning(
                    #     f"Fisher 参数 {n} 的形状 {fisher_val.shape} 与模型参数形状 {p.shape} 不匹配，跳过该参数。")
                    continue

                # logger.info(
                #     f"参数 {n}：fisher_val shape: {fisher_val.shape}, optpar_val shape: {optpar_val.shape}, p shape: {p.shape}")
                loss_ewc += (fisher_val * (p - optpar_val) ** 2).sum()

        return lambda_dyn * loss_ewc

    def estimate_and_save_fisher(self, dataset_or_loader, device=None, sample_size=100, batch_save_interval=10):
        """
        在当前任务数据上估计Fisher (只需少量样本), 并保存 {fisher, optpar} 到 ewc_dir
        :param dataset_or_loader: 也可以是Dataset 或Dataloader, 用来估计Fisher
        :param device:
        :param sample_size: 只拿这么多样本估计, 以免太慢
        """
        if device is None:
            device = next(self.model.parameters()).device

        # 保存当前模型参数
        optpar_dict = {n: p.detach().cpu().numpy() for n, p in self.model.named_parameters()}

        # 构建 Fisher 字典
        fisher_dict = {n: torch.zeros_like(p, device=device) for n, p in self.model.named_parameters()}

        # 获取当前任务的标签数量
        current_task_num_labels = self.num_labels  # 假设 `self.num_labels` 是当前任务的标签数量
        max_num_labels = 23  # 假设最大标签数量为 23 (根据所有任务的最大标签数)

        # 遍历样本(只 sample_size 次)
        # 如果 dataset_or_loader 是 Dataset,就手动取sample
        # 如果是 Dataloader, 就 iterate
        self.model.eval()
        count = 0
        sample_data = []

        if hasattr(dataset_or_loader, "__len__") and not hasattr(dataset_or_loader, "batch_size"):
            # dataset
            indices = list(range(len(dataset_or_loader)))
            import random
            random.shuffle(indices)
            indices = indices[:sample_size]
            sample_data = [dataset_or_loader[i] for i in indices]
        else:
            # assume it's DataLoader
            for batch in dataset_or_loader:
                sample_data.append(batch)
                count += 1
                if count>=sample_size:
                    break
        # 将 Fisher 和 Optpar 分批次保存
        for batch in sample_data:
            # 组装输入
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if "token_type_ids" in batch:
                token_type_ids = batch["token_type_ids"].to(device)
            else:
                token_type_ids = None
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["labels"].to(device)

            # forward
            self.model.zero_grad()
            if self.task_name in ["mate", "mner", "mabsa"]:
                fused_feat = self.model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=True
                )

                logits = self.model.head(fused_feat)  # => (batch_size, seq_len, num_labels)

                # 扁平化 logits 和 labels
                logits = logits.view(-1, self.num_labels)  # 扁平化 logits 为 (batch_size * seq_len, num_labels)
                labels = labels.view(-1)  # 扁平化 labels 为 (batch_size * seq_len)

                # 根据任务选择对应的 class_weights（如果需要使用权重的话）
                if self.task_name == "mate":
                    class_weights = torch.tensor([1.0, 15.0, 15.0], device=device)
                elif self.task_name == "mner":
                    class_weights = torch.tensor([0.1, 164.0, 10.0, 270.0, 27.0, 340.0, 16.0, 360.0, 2.0],
                                                 device=device)
                elif self.task_name == "mabsa":
                    class_weights = torch.tensor([1.0, 3700.0, 234.0, 480.0, 34.0, 786.0, 69.0], device=device)
                loss = nn.functional.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    ignore_index=-100
                )
            else:
                # 句级分类: return_sequence=False => (batch_size, fusion_dim)
                fused_feat = self.model.base_model(
                    input_ids, attention_mask, token_type_ids, image_tensor,
                    return_sequence=False
                )
                logits = self.model.head(fused_feat)  # => (batch_size, num_labels)

                loss = nn.functional.cross_entropy(logits, labels)  # => (batch_size)
            loss.backward()

            # 计算 Fisher 信息并累计
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_dict[n] += p.grad ** 2

            # # 每batch_save_interval个批次保存一次
            # batch_count += 1
            # if batch_count % batch_save_interval == 0 or batch_count == count:
            #     # 保存 Fisher 和 optpar 到 .pt 文件
            #     fisher_path = os.path.join(self.ewc_dir, f"{self.session_name}_fisher_batch_{batch_count}.pt")
            #     torch.save({'fisher': fisher_dict, 'optpar': optpar_dict}, fisher_path)
            #     logger.info(
            #         f"Saved Fisher and Optpar (batch {batch_count}) for task={self.session_name} => {fisher_path}")
        # 在保存之前对 fisher_dict 和 optpar_dict 进行规范化
        # 进行填充，确保所有任务的 Fisher 和 optpar 的维度一致
        for n, fisher in fisher_dict.items():
            fisher_shape = fisher.size()
            p_shape = self.model.state_dict()[n].size()

            # 填充至模型参数维度
            if fisher_shape != p_shape:
                fisher_dict[n] = F.pad(fisher, (0, p_shape[2] - fisher_shape[2]))  # 填充特征维度

        # 最后保存
        fisher_path = os.path.join(self.ewc_dir, f"{self.session_name}_fisher.pt")
        torch.save({'fisher': fisher_dict, 'optpar': optpar_dict}, fisher_path)
        logger.info(f"Final Fisher and Optpar saved for task={self.session_name} => {fisher_path}")