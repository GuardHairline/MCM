# continual/ewc_new.py
import torch
import torch.nn.functional as F
import json
import os

class MultiTaskEWC:
    """
    多任务版本EWC：每学完一个任务后，估计 Fisher 并保存到磁盘；
    下次学新任务时，加载所有旧任务 Fisher 与旧参数进行 penalty。
    不再需要旧数据集。
    """
    def __init__(self, model, current_task_name, ewc_lambda=1000, ewc_dir="ewc_params"):
        """
        :param model: 当前要学习的模型
        :param current_task_name: 当前任务的名称（字符串）
        :param ewc_lambda: EWC损失项系数
        :param ewc_dir: 存放 fisher/param 的目录
        """
        self.model = model
        self.task_name = current_task_name
        self.ewc_lambda = ewc_lambda
        self.ewc_dir = ewc_dir

        # 用来累加所有旧任务的fisher和optpar
        # fisher_all, optpar_all => {param_name: aggregated_value}
        # 这里仅在 penalty() 时构建
        self.fisher_all = {}
        self.optpar_all = {}

    def load_all_previous_tasks(self):
        """
        加载 ewc_dir 下所有旧任务的 fisher & optpar 文件，
        然后把它们都累加到 self.fisher_all / self.optpar_all
        """
        # 先初始化为空
        self.fisher_all = {}
        self.optpar_all = {}

        if not os.path.exists(self.ewc_dir):
            os.makedirs(self.ewc_dir)

        # 遍历 ewc_dir 下所有json文件
        files = [f for f in os.listdir(self.ewc_dir) if f.endswith(".json")]
        for file in files:
            path = os.path.join(self.ewc_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fisher_dict = data["fisher"]     # {param_name: list or tensor}
            optpar_dict = data["optpar"]     # {param_name: list or tensor}

            # 累加
            for n, p in fisher_dict.items():
                p_tensor = torch.tensor(p) if not isinstance(p, list) else torch.tensor(p)
                if n not in self.fisher_all:
                    self.fisher_all[n] = p_tensor
                else:
                    self.fisher_all[n] += p_tensor

            for n, p in optpar_dict.items():
                p_tensor = torch.tensor(p) if not isinstance(p, list) else torch.tensor(p)
                if n not in self.optpar_all:
                    self.optpar_all[n] = p_tensor
                else:
                    # 这里 optpar 可能更合理的是 "存最后一次" or "存多份"?
                    # 多任务合并时：EWC penality = ∑ fisher_i * (θ - θ_i*)^2
                    # 需要对每个任务 i 的 (fisher_i, optpar_i) 分开 penalty
                    # => 这里可以记录多个
                    #
                    # 但若要一次 penaltySum, 需要区分task i.
                    # 这里演示简单相加(online EWC).
                    self.optpar_all[n] += p_tensor

    def penalty(self, model):
        """
        计算 EWC penalty = sum_{n} fisher_all[n] * (p - optpar_all[n])^2
        如果self.fisher_all/optpar_all为空, 返回0
        """
        if not self.fisher_all or not self.optpar_all:
            return torch.tensor(0., device=next(model.parameters()).device)

        device = next(model.parameters()).device
        loss_ewc = torch.tensor(0., device=device)
        for n, p in model.named_parameters():
            if n in self.fisher_all:
                fisher_val = self.fisher_all[n].to(device)
                optpar_val = self.optpar_all[n].to(device)
                loss_ewc += (fisher_val * (p - optpar_val)**2).sum()
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

        # 1) 先保存当前optpar
        optpar_dict = {}
        for n, p in self.model.named_parameters():
            optpar_dict[n] = p.detach().cpu().numpy()  # or p.detach().tolist()

        # 2) 构建fisher_dict
        fisher_dict = {}
        for n, p in self.model.named_parameters():
            fisher_dict[n] = torch.zeros_like(p, device=device)

        # 3) 遍历样本(只 sample_size 次)
        # 如果 dataset_or_loader 是 Dataset,就手动取sample
        # 如果是 Dataloader, 就 iterate
        self.model.eval()
        count = 0

        if hasattr(dataset_or_loader, "__len__") and not hasattr(dataset_or_loader, "batch_size"):
            # dataset
            indices = list(range(len(dataset_or_loader)))
            import random
            random.shuffle(indices)
            indices = indices[:sample_size]
            sample_data = [dataset_or_loader[i] for i in indices]
        else:
            # assume it's DataLoader
            sample_data = []
            for batch in dataset_or_loader:
                sample_data.append(batch)
                count += 1
                if count>=sample_size:
                    break

        count = 0
        for batch in sample_data:
            # 组装输入
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"]
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["label"].to(device)

            # forward
            self.model.zero_grad()
            logits = self.model(input_ids, attention_mask, token_type_ids, image_tensor)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_dict[n] += p.grad**2
            count += 1

        for n in fisher_dict:
            if count>0:
                fisher_dict[n] /= float(count)

        # 4) 保存到 ewc_dir: self.task_name + ".json"
        result = {
            "fisher": {},
            "optpar": {}
        }
        for n, p in fisher_dict.items():
            result["fisher"][n] = p.cpu().numpy().tolist()  # or .numpy()
        for n, val in optpar_dict.items():
            result["optpar"][n] = val.tolist() if hasattr(val, "tolist") else val

        save_path = os.path.join(self.ewc_dir, f"{self.task_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            import json
            json.dump(result, f)
        print(f"[MultiTaskEWC] Saved fisher & optpar for task={self.task_name} => {save_path}")
