# continual/ewc_new.py
import torch
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
    def __init__(self, model, current_task_name, session_name, ewc_lambda=1000, ewc_dir="ewc_params"):
        """
        :param model: 当前要学习的模型
        :param current_task_name: 当前任务的名称（字符串）
        :param ewc_lambda: EWC损失项系数
        :param ewc_dir: 存放 fisher/param 的目录
        """
        self.model = model
        self.task_name = current_task_name
        self.session_name = session_name
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
        # 如果没有任务历史，则跳过
        if not os.path.exists(self.ewc_dir):
            os.makedirs(self.ewc_dir)
            logger.info(f"EWC directory '{self.ewc_dir}' created.")

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

    def estimate_and_save_fisher(self, dataset_or_loader, device=None, sample_size=200):
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
            logits = self.model(input_ids, attention_mask, token_type_ids, image_tensor)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_dict[n] += p.grad**2

        for n in fisher_dict:
            if count>0:
                fisher_dict[n] /= float(count)

        # 保存 Fisher 和 optpar 到 .pt 文件
        fisher_path = os.path.join(self.ewc_dir, f"{self.session_name}_fisher.pt")
        torch.save({'fisher': fisher_dict, 'optpar': optpar_dict}, fisher_path)
        logger.info(f"Saved Fisher and Optpar for task={self.session_name} => {fisher_path}")
