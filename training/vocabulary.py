from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PAD_ID = 0
UNK_ID = 1


def hash_token_id(value: str, capacity: int, namespace: str) -> int:
    digest = hashlib.blake2b((namespace + value).encode(), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % (capacity - 1)


@dataclass(frozen=True)
class ExactVocabulary:
    embedding_capacity: int
    characters: dict[str, int]
    pinyin: dict[str, int]

    @property
    def assigned_tokens(self) -> int:
        return len(self.characters) + len(self.pinyin)

    @property
    def required_embeddings(self) -> int:
        return 2 + self.assigned_tokens

    def character_id(self, value: str) -> int:
        return self.characters.get(value, UNK_ID)

    def pinyin_id(self, value: str) -> int:
        return self.pinyin.get(value, UNK_ID)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "embedding_capacity": self.embedding_capacity,
            "assigned_tokens": self.assigned_tokens,
            "characters": self.characters,
            "pinyin": self.pinyin,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ExactVocabulary":
        if value.get("pad_id") != PAD_ID or value.get("unk_id") != UNK_ID:
            raise ValueError("exact vocabulary PAD/UNK IDs are incompatible")
        vocabulary = cls(
            embedding_capacity=int(value["embedding_capacity"]),
            characters={str(token): int(index) for token, index in value["characters"].items()},
            pinyin={str(token): int(index) for token, index in value["pinyin"].items()},
        )
        ids = [*vocabulary.characters.values(), *vocabulary.pinyin.values()]
        if len(ids) != len(set(ids)) or any(index < 2 for index in ids):
            raise ValueError("exact vocabulary IDs must be unique and above UNK")
        if ids and max(ids) >= vocabulary.embedding_capacity:
            raise ValueError("exact vocabulary exceeds embedding capacity")
        return vocabulary

    @classmethod
    def load(cls, path: Path) -> "ExactVocabulary":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_exact_vocabulary(
    character_counts: Counter[str], pinyin_counts: Counter[str], embedding_capacity: int
) -> ExactVocabulary:
    ordered_characters = sorted(character_counts, key=lambda token: (-character_counts[token], token))
    ordered_pinyin = sorted(pinyin_counts, key=lambda token: (-pinyin_counts[token], token))
    required = 2 + len(ordered_characters) + len(ordered_pinyin)
    if required > embedding_capacity:
        raise ValueError(f"exact vocabulary requires {required} embeddings, capacity is {embedding_capacity}")
    next_id = 2
    characters = {}
    for token in ordered_characters:
        characters[token] = next_id
        next_id += 1
    pinyin = {}
    for token in ordered_pinyin:
        pinyin[token] = next_id
        next_id += 1
    return ExactVocabulary(embedding_capacity, characters, pinyin)


def collision_report(
    character_counts: Counter[str], pinyin_counts: Counter[str], capacity: int, high_frequency: int = 100
) -> dict:
    tokens = [
        ("char", token, count, hash_token_id(token, capacity, "char:"))
        for token, count in character_counts.items()
    ] + [
        ("pinyin", token, count, hash_token_id(token, capacity, "pinyin:"))
        for token, count in pinyin_counts.items()
    ]
    buckets: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    for namespace, token, count, bucket in tokens:
        buckets[bucket].append((namespace, token, count))
    colliding = {bucket: values for bucket, values in buckets.items() if len(values) > 1}
    frequency_order = sorted(tokens, key=lambda item: (-item[2], item[0], item[1]))
    high_frequency_collisions = []
    for namespace, token, count, bucket in frequency_order[:high_frequency]:
        values = colliding.get(bucket)
        if values:
            high_frequency_collisions.append(
                {
                    "namespace": namespace,
                    "token": token,
                    "frequency": count,
                    "bucket": bucket,
                    "collides_with": [
                        {"namespace": other_namespace, "token": other_token, "frequency": other_count}
                        for other_namespace, other_token, other_count in sorted(values)
                        if (other_namespace, other_token) != (namespace, token)
                    ],
                }
            )
    unique_tokens = len(tokens)
    collision_count = unique_tokens - len(buckets)
    return {
        "embedding_capacity": capacity,
        "usable_hash_buckets": capacity - 1,
        "unique_tokens": unique_tokens,
        "occupied_buckets": len(buckets),
        "collision_count": collision_count,
        "collision_rate": collision_count / unique_tokens if unique_tokens else 0.0,
        "colliding_buckets": len(colliding),
        "top100_tokens_with_collision": len(high_frequency_collisions),
        "high_frequency_collisions": high_frequency_collisions,
    }
