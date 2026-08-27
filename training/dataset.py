from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data_pipeline.validate_dataset import iter_examples
from tinyrime import RankingExample


def token_id(value: str, vocabulary_size: int, namespace: str) -> int:
    digest = hashlib.blake2b((namespace + value).encode(), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % (vocabulary_size - 1)


def padded(values: list[int], length: int) -> list[int]:
    return (values[:length] + [0] * length)[:length]


class RimeRankingDataset(Dataset):
    def __init__(self, path: Path, vocabulary_size: int, top_k: int = 8):
        self.examples = [RankingExample.from_dict(raw, top_k=top_k) for raw in iter_examples(path)]
        self.vocabulary_size = vocabulary_size
        self.top_k = top_k
        targets_by_pinyin: dict[str, set[str]] = {}
        for example in self.examples:
            key = "/".join(example.pinyin)
            targets_by_pinyin.setdefault(key, set()).add(example.candidates[example.target_index].text)
        self.contested_keys = {key for key, targets in targets_by_pinyin.items() if len(targets) >= 2}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        context = padded(
            [token_id(character, self.vocabulary_size, "char:") for character in example.context[-32:]], 32
        )
        pinyin = padded(
            [token_id(syllable, self.vocabulary_size, "pinyin:") for syllable in example.pinyin[:16]], 16
        )
        candidate_ids = []
        numeric = []
        candidate_mask = []
        for candidate in example.candidates[: self.top_k]:
            candidate_ids.append(
                padded([token_id(character, self.vocabulary_size, "char:") for character in candidate.text[:8]], 8)
            )
            quality = 0.0 if candidate.quality is None else max(-20.0, min(20.0, candidate.quality)) / 20.0
            type_value = 0.0 if candidate.type is None else token_id(candidate.type, 257, "type:") / 256.0
            numeric.append([candidate.rank / 7.0, quality, min(len(candidate.text), 8) / 8.0, type_value])
            candidate_mask.append(True)
        while len(candidate_ids) < self.top_k:
            candidate_ids.append([0] * 8)
            numeric.append([0.0] * 4)
            candidate_mask.append(False)
        contested_key = "/".join(example.pinyin)
        return {
            "context_ids": torch.tensor(context, dtype=torch.long),
            "pinyin_ids": torch.tensor(pinyin, dtype=torch.long),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "numeric_features": torch.tensor(numeric, dtype=torch.float32),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
            "target": torch.tensor(example.target_index, dtype=torch.long),
            "baseline_correct": torch.tensor(example.target_index == 0, dtype=torch.bool),
            "contested": torch.tensor(contested_key in self.contested_keys, dtype=torch.bool),
        }
