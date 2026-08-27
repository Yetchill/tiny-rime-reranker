from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from training.candidate_types import CANDIDATE_TYPES


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    encoder_layers: int
    encoder_heads: int
    type_encoding: str = "legacy_zero"
    token_encoding: str = "hash"
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
PRESETS.update(
    {
        "mlp-scalar-type-hash": replace(
            PRESETS["mlp"], name="mlp-scalar-type-hash", type_encoding="scalar_hash"
        ),
        "tiny-2m-scalar-type-hash": replace(
            PRESETS["tiny-2m"], name="tiny-2m-scalar-type-hash", type_encoding="scalar_hash"
        ),
        "tiny-4m-scalar-type-hash": replace(
            PRESETS["tiny-4m"], name="tiny-4m-scalar-type-hash", type_encoding="scalar_hash"
        ),
        "tiny-8m-scalar-type-hash": replace(
            PRESETS["tiny-8m"], name="tiny-8m-scalar-type-hash", type_encoding="scalar_hash"
        ),
        "mlp-cat-hash": replace(PRESETS["mlp"], name="mlp-cat-hash", type_encoding="categorical"),
        "tiny-2m-cat-hash": replace(
            PRESETS["tiny-2m"], name="tiny-2m-cat-hash", type_encoding="categorical"
        ),
        "tiny-4m-cat-hash": replace(
            PRESETS["tiny-4m"], name="tiny-4m-cat-hash", type_encoding="categorical"
        ),
        "tiny-8m-cat-hash": replace(
            PRESETS["tiny-8m"], name="tiny-8m-cat-hash", type_encoding="categorical"
        ),
    }
)
for base in ("tiny-2m", "tiny-4m", "tiny-8m"):
    categorical = PRESETS[f"{base}-cat-hash"]
    PRESETS[f"{base}-cat-exact"] = replace(
        categorical,
        name=f"{base}-cat-exact",
        token_encoding="exact",
    )


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
        if config.type_encoding not in {"legacy_zero", "scalar_hash", "categorical"}:
            raise ValueError(f"unknown candidate type encoding: {config.type_encoding}")
        self.type_embedding = (
            nn.Embedding(len(CANDIDATE_TYPES), 8, padding_idx=0)
            if config.type_encoding == "categorical"
            else None
        )
        numeric_size = 3 if self.type_embedding is not None else 4
        combined = config.embedding_dim * 3 + numeric_size + (8 if self.type_embedding is not None else 0)
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
        candidate_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_embeddings = self.embedding(context_ids)
        if self.config.encoder_layers:
            padding_mask = context_ids.eq(0)
            context_embeddings = self.context_encoder(context_embeddings, src_key_padding_mask=padding_mask)
        context_vector = masked_mean(context_embeddings, context_ids, 1)
        pinyin_vector = masked_mean(self.embedding(pinyin_ids), pinyin_ids, 1)
        candidate_vector = masked_mean(self.embedding(candidate_ids), candidate_ids, 2)
        count = candidate_ids.shape[1]
        parts = [
            context_vector.unsqueeze(1).expand(-1, count, -1),
            pinyin_vector.unsqueeze(1).expand(-1, count, -1),
            candidate_vector,
            numeric_features,
        ]
        if self.type_embedding is not None:
            if candidate_type_ids is None:
                raise ValueError("categorical type encoding requires candidate_type_ids")
            parts.append(self.type_embedding(candidate_type_ids))
        combined = torch.cat(parts, dim=-1)
        residuals = self.scorer(combined).squeeze(-1)
        confidence = self.gate(torch.cat((context_vector, pinyin_vector), dim=-1)).squeeze(-1)
        return residuals, confidence

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
