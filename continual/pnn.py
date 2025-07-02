# \continual\pnn.py
import torch

class PNNManager:
    """
    Manage Progressive Neural Networks: freeze old columns, add new.
    """
    def __init__(self, text_model_name, image_model_name, fusion_strategy, num_heads, mode, hidden_dim):
        self.columns = []
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.fusion_strategy = fusion_strategy
        self.num_heads = num_heads
        self.mode = mode
        self.hidden_dim = hidden_dim

    def add_task(self, num_labels):
        # freeze previous
        for col in self.columns:
            for p in col.parameters(): p.requires_grad=False
        # new column
        from models.base_model import BaseMultimodalModel
        from models.task_heads.get_head import get_head
        base = BaseMultimodalModel(
            self.text_model_name,
            self.image_model_name,
            multimodal_fusion=self.fusion_strategy,
            num_heads=self.num_heads,
            mode=self.mode,
            hidden_dim=self.hidden_dim
        )
        head = get_head('', base, {'num_labels':num_labels, 'dropout_prob':0.1, 'hidden_dim':self.hidden_dim})
        col = torch.nn.Module()
        col.base_model = base
        col.head = head
        self.columns.append(col)
        return col

    def forward(self, inputs, task_idx):
        # lateral connections omitted for simplicity
        col = self.columns[task_idx]
        return col.base_model(**inputs, return_sequence=(inputs['labels'].dim()>1))