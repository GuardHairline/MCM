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
    def __init__(self, model, current_task_name, session_name, num_labels, ewc_lambda=1000, ewc_dir="ewc_params"):
        """
        :param model: 当前要学习的模型
        :param current_task_name: 当前任务的名称（字符串）
        :param session_name: 当前任务的会话名称（字符串）
        :param num_labels: 当前任务的标签数量
        :param ewc_lambda: EWC损失项系数
        :param ewc_dir: 存放 fisher/param 的目录
        """
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
        # 如果没有 train_info 文件，则跳过
        if not os.path.exists(self.ewc_dir):
            os.makedirs(self.ewc_dir)
            logger.info(f"EWC directory '{self.ewc_dir}' created.")

        # 如果没有历史任务信息文件（train_info），则跳过
        if "sessions" not in train_info:
            logger.warning("No previous sessions in train_info, skipping loading of Fisher and optpar.")
            return

        self.fisher_all = {}
        self.optpar_all = {}
        logger.info("Loading Fisher and optpar for previous tasks...")

        for session in train_info["sessions"]:
            fisher_file_path = session["fisher_file"]
            if os.path.exists(fisher_file_path):
                logger.info(f"Loading Fisher and optpar from: {fisher_file_path}")
                checkpoint = torch.load(fisher_file_path)
                fisher_dict = checkpoint["fisher"]
                optpar_dict = checkpoint["optpar"]

                for n, p in fisher_dict.items():
                    p_tensor = torch.tensor(p) if not isinstance(p, torch.Tensor) else p
                    if n not in self.fisher_all:
                        self.fisher_all[n] = p_tensor
                    else:
                        self.fisher_all[n] += p_tensor

                for n, p in optpar_dict.items():
                    p_tensor = torch.tensor(p) if not isinstance(p, torch.Tensor) else p
                    if n not in self.optpar_all:
                        self.optpar_all[n] = p_tensor
                    else:
                        self.optpar_all[n] += p_tensor
            else:
                logger.warning(f"Fisher file for session {session['session_name']} not found at {fisher_file_path}")

        logger.info("Finished loading Fisher and optpar from previous tasks.")

    def penalty(self, model):
        """
        计算 EWC penalty = sum_{n} fisher_all[n] * (p - optpar_all[n])^2
        如果self.fisher_all/optpar_all为空, 返回0
        """
        if not self.fisher_all or not self.optpar_all:
            logger.warning("Fisher or optpar are empty. Returning penalty = 0.")
            return torch.tensor(0., device=next(model.parameters()).device)

        device = next(model.parameters()).device
        loss_ewc = torch.tensor(0., device=device)

        logger.info("Calculating EWC penalty...")

        for n, p in model.named_parameters():
            if n in self.fisher_all:
                fisher_val = self.fisher_all[n].to(device)
                optpar_val = self.optpar_all[n].to(device)
                loss_ewc += (fisher_val * (p - optpar_val)**2).sum()
        logger.info(f"EWC penalty calculation completed with total penalty: {loss_ewc.item()}")
        return self.ewc_lambda * loss_ewc

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
        # batch_count = 0  # 记录保存的批次数
        for batch in sample_data:
            # 组装输入
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"]
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
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

        # 最后保存
        fisher_path = os.path.join(self.ewc_dir, f"{self.session_name}_fisher_final.pt")
        torch.save({'fisher': fisher_dict, 'optpar': optpar_dict}, fisher_path)
        logger.info(f"Final Fisher and Optpar saved for task={self.session_name} => {fisher_path}")