import torch
import torch.nn as nn
import torch.nn.functional as F
from continual.label_embedding import GlobalLabelEmbedding
from torchcrf import CRF
from typing import Dict, Any, Optional

class TokenLabelHead(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            num_labels, 
            label_emb: GlobalLabelEmbedding, 
            task_name: str, 
            use_crf: bool = True,
            use_cosine: bool = False,
            init_temperature: float = 10.0,
            dropout: float = 0.1,
        ):
        super().__init__()
        self.num_labels = num_labels
        self.label_emb = label_emb
        self.task_name = task_name
        self.use_crf = use_crf and CRF is not None
        self.use_cosine = use_cosine

        # ---- Token projection ----
        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Label projection ----
        self.label_proj = nn.Linear(label_emb.emb_dim, hidden_dim, bias=False)

        # ---- Temperature (learnable scalar) ----
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

        # ---- CRF transition ----
        if self.use_crf:
            self.crf = CRF(num_tags=num_labels, batch_first=True)
    def _compute_logits(self, token_h: torch.Tensor, label_h: torch.Tensor) -> torch.Tensor:
        """
        token_h : (B, L, d)       after projection & non‑linearity
        label_h : (num_labels, d) after projection
        Returns
        -------
        logits  : (B, L, num_labels)
        """
        if self.use_cosine:
            token_h = F.normalize(token_h, dim=-1)
            label_h = F.normalize(label_h, dim=-1)
        # (B,L,d) @ (d, num_labels) -> (B,L,num_labels)
        logits = torch.matmul(token_h, label_h.t())
        logits = logits * self.temperature
        return logits
    def forward(
            self,
            seq_feats: torch.Tensor,                 # (B,L,input_dim)
            labels: Optional[torch.Tensor] = None,   # (B,L) or None
            attention_mask: Optional[torch.Tensor] = None,  # (B,L) 1=valid
        ) -> Dict[str, Any]:

        # seq_feats: (batch, seq_len, input_dim)
        token_h = self.token_proj(seq_feats)  # (batch, seq_len, hidden_dim)
        label_embeddings = self.label_emb.get_all_label_embeddings(self.task_name)  # (num_labels, emb_dim)
        label_h = self.label_proj(label_embeddings)  # (num_labels, hidden_dim)

        logits = self._compute_logits(token_h, label_h)   # (B,L,num_labels)

        output: Dict[str, Any] = {"logits": logits}

        if labels is not None:
            if attention_mask is None:
                # Default: treat all tokens valid
                attention_mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()

            if self.use_crf:
                # CRF expects mask = ByteTensor / BoolTensor, 1 for valid
                nll = -self.crf(logits, labels, mask=attention_mask, reduction="mean")
                output["loss"] = nll
            else:
                # Flatten for CrossEntropy
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                active_idx = attention_mask.view(-1)  # 1/0
                active_logits = logits.view(-1, self.num_labels)[active_idx]
                active_labels = labels.view(-1)[active_idx]
                output["loss"] = loss_fct(active_logits, active_labels)

        else:  # inference
            if self.use_crf:
                best_paths = self.crf.decode(
                    logits, mask=attention_mask.bool() if attention_mask is not None else None
                )
                output["predictions"] = best_paths   # List[List[int]]
            else:
                preds = logits.argmax(dim=-1)        # (B,L)
                output["predictions"] = preds

        return logits