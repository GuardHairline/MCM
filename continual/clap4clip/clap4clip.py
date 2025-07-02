import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class CLAP4CLIP(nn.Module):
    def __init__(self, text_model_name, image_model_name, num_labels, dropout_prob=0.1):
        super(CLAP4CLIP, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(text_model_name)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.clip_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, image_tensor):
        text_features = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        image_features = self.clip_model.get_image_features(image_tensor)

        # 概率微调过程
        combined_features = text_features + image_features  # 简化的加法融合
        logits = self.classifier(self.dropout(combined_features))
        return logits
