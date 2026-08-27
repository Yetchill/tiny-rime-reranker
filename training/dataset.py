from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from data_pipeline.validate_dataset import iter_examples
from tinyrime import RankingExample
from training.candidate_types import candidate_type_id
from training.vocabulary import ExactVocabulary, hash_token_id


def token_id(value: str, vocabulary_size: int, namespace: str) -> int:
    return hash_token_id(value, vocabulary_size, namespace)


def padded(values: list[int], length: int) -> list[int]:
    return (values[:length] + [0] * length)[:length]


class RimeRankingDataset(Dataset):
    def __init__(
        self,
        path: Path,
        vocabulary_size: int,
        top_k: int = 8,
        type_encoding: str = "legacy_zero",
        exact_vocabulary: ExactVocabulary | None = None,
    ):
        self.examples = [RankingExample.from_dict(raw, top_k=top_k) for raw in iter_examples(path)]
        self.vocabulary_size = vocabulary_size
        self.top_k = top_k
        if type_encoding not in {"legacy_zero", "scalar_hash", "categorical"}:
            raise ValueError(f"unknown type encoding: {type_encoding}")
        self.type_encoding = type_encoding
        self.exact_vocabulary = exact_vocabulary
        if exact_vocabulary is not None and exact_vocabulary.embedding_capacity != vocabulary_size:
            raise ValueError("exact vocabulary capacity does not match model embedding capacity")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        context = padded([self.character_id(character) for character in example.context[-32:]], 32)
        pinyin = padded([self.pinyin_id(syllable) for syllable in example.pinyin[:16]], 16)
        candidate_ids = []
        numeric = []
        candidate_mask = []
        candidate_type_ids = []
        for candidate in example.candidates[: self.top_k]:
            candidate_ids.append(
                padded([self.character_id(character) for character in candidate.text[:8]], 8)
            )
            quality = 0.0 if candidate.quality is None else max(-20.0, min(20.0, candidate.quality)) / 20.0
            if self.type_encoding in {"legacy_zero", "scalar_hash"}:
                type_value = (
                    0.0
                    if self.type_encoding == "legacy_zero" or candidate.type is None
                    else token_id(candidate.type, 257, "type:") / 256.0
                )
                numeric.append(
                    [
                        candidate.rank / max(1, self.top_k - 1),
                        quality,
                        min(len(candidate.text), 8) / 8.0,
                        type_value,
                    ]
                )
            else:
                numeric.append(
                    [candidate.rank / max(1, self.top_k - 1), quality, min(len(candidate.text), 8) / 8.0]
                )
            candidate_type_ids.append(candidate_type_id(candidate.type))
            candidate_mask.append(True)
        while len(candidate_ids) < self.top_k:
            candidate_ids.append([0] * 8)
            numeric.append([0.0] * (3 if self.type_encoding == "categorical" else 4))
            candidate_mask.append(False)
            candidate_type_ids.append(0)
        return {
            "context_ids": torch.tensor(context, dtype=torch.long),
            "pinyin_ids": torch.tensor(pinyin, dtype=torch.long),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "numeric_features": torch.tensor(numeric, dtype=torch.float32),
            "candidate_type_ids": torch.tensor(candidate_type_ids, dtype=torch.long),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
            "target": torch.tensor(example.target_index, dtype=torch.long),
            "baseline_correct": torch.tensor(example.target_index == 0, dtype=torch.bool),
            "contested": torch.tensor(example.contested, dtype=torch.bool),
        }

    def character_id(self, value: str) -> int:
        return (
            self.exact_vocabulary.character_id(value)
            if self.exact_vocabulary is not None
            else token_id(value, self.vocabulary_size, "char:")
        )

    def pinyin_id(self, value: str) -> int:
        return (
            self.exact_vocabulary.pinyin_id(value)
            if self.exact_vocabulary is not None
            else token_id(value, self.vocabulary_size, "pinyin:")
        )
