from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    encoder_layers: int
    encoder_heads: int
    alpha: float = 0.25
    top_k: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "linear": ModelConfig("linear", 4096, 16, 0, 0, 1),
    "mlp": ModelConfig("mlp", 8192, 32, 96, 0, 1),
    "tiny-2m": ModelConfig("tiny-2m", 8192, 160, 160, 1, 5),
    "tiny-4m": ModelConfig("tiny-4m", 16384, 160, 256, 3, 5),
    "tiny-8m": ModelConfig("tiny-8m", 32768, 192, 256, 3, 6),
}


def masked_mean(values: torch.Tensor, ids: torch.Tensor, dimension: int) -> torch.Tensor:
    mask = ids.ne(0).unsqueeze(-1)
    denominator = mask.sum(dim=dimension).clamp_min(1)
    return (values * mask).sum(dim=dimension) / denominator


class TinyContextReranker(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        if config.encoder_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.encoder_heads,
                dim_feedforward=config.embedding_dim * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder: nn.Module = nn.TransformerEncoder(layer, config.encoder_layers)
        else:
            self.context_encoder = nn.Identity()
        combined = config.embedding_dim * 3 + 4
        if config.hidden_dim:
            self.scorer = nn.Sequential(
                nn.Linear(combined, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.gate = nn.Sequential(
                nn.Linear(config.embedding_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )
        else:
            self.scorer = nn.Linear(combined, 1)
            self.gate = nn.Linear(config.embedding_dim * 2, 1)

    def forward(
        self,
        context_ids: torch.Tensor,
        pinyin_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_embeddings = self.embedding(context_ids)
        if self.config.encoder_layers:
            padding_mask = context_ids.eq(0)
            context_embeddings = self.context_encoder(context_embeddings, src_key_padding_mask=padding_mask)
        context_vector = masked_mean(context_embeddings, context_ids, 1)
        pinyin_vector = masked_mean(self.embedding(pinyin_ids), pinyin_ids, 1)
        candidate_vector = masked_mean(self.embedding(candidate_ids), candidate_ids, 2)
        count = candidate_ids.shape[1]
        combined = torch.cat(
            (
                context_vector.unsqueeze(1).expand(-1, count, -1),
                pinyin_vector.unsqueeze(1).expand(-1, count, -1),
                candidate_vector,
                numeric_features,
            ),
            dim=-1,
        )
        residuals = self.scorer(combined).squeeze(-1)
        confidence = self.gate(torch.cat((context_vector, pinyin_vector), dim=-1)).squeeze(-1)
        return residuals, confidence

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
