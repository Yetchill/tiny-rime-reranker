from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from data_pipeline.validate_dataset import iter_examples
from training.candidate_types import CANDIDATE_TYPE_TO_ID
from training.vocabulary import build_exact_vocabulary, collision_report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_han(character: str) -> bool:
    return "\u3400" <= character <= "\u9fff"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact compact vocabulary and audit hash collisions")
    parser.add_argument("train_dataset", type=Path)
    parser.add_argument("--embedding-capacity", type=int, default=16384)
    parser.add_argument("--hash-capacities", default="4096,8192,16384,32768")
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    character_counts: Counter[str] = Counter()
    pinyin_counts: Counter[str] = Counter()
    candidate_type_counts: Counter[str] = Counter()
    examples = 0
    for example in iter_examples(args.train_dataset):
        examples += 1
        character_counts.update(example.get("context", "")[-32:])
        pinyin_counts.update(example.get("pinyin", [])[:16])
        for candidate in example.get("candidates", [])[:32]:
            character_counts.update(str(candidate["text"])[:8])
            candidate_type_counts[str(candidate.get("type") or "<UNK>")] += 1
    vocabulary = build_exact_vocabulary(character_counts, pinyin_counts, args.embedding_capacity)
    args.vocabulary.parent.mkdir(parents=True, exist_ok=True)
    args.vocabulary.write_text(json.dumps(vocabulary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    capacities = [int(value) for value in args.hash_capacities.split(",") if value.strip()]
    unknown_types = sum(
        count
        for value, count in candidate_type_counts.items()
        if value == "<UNK>" or value not in CANDIDATE_TYPE_TO_ID
    )
    audit = {
        "schema_version": 1,
        "train_dataset": str(args.train_dataset),
        "train_dataset_sha256": sha256(args.train_dataset),
        "examples": examples,
        "unique_character_tokens": len(character_counts),
        "unique_han_characters": sum(is_han(character) for character in character_counts),
        "unique_pinyin_syllables": len(pinyin_counts),
        "character_token_occurrences": sum(character_counts.values()),
        "pinyin_token_occurrences": sum(pinyin_counts.values()),
        "candidate_type_counts": dict(candidate_type_counts.most_common()),
        "unknown_candidate_type_occurrences": unknown_types,
        "exact_vocabulary": {
            "embedding_capacity": vocabulary.embedding_capacity,
            "assigned_tokens": vocabulary.assigned_tokens,
            "required_embeddings": vocabulary.required_embeddings,
            "unused_embeddings": vocabulary.embedding_capacity - vocabulary.required_embeddings,
            "sha256": sha256(args.vocabulary),
        },
        "hash_collision_audit": {
            str(capacity): collision_report(character_counts, pinyin_counts, capacity)
            for capacity in capacities
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
