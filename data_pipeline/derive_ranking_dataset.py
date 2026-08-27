from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import zstandard

from data_pipeline.benchmark_v1 import canonical_ranking_example, deterministic_train_sample
from data_pipeline.validate_dataset import iter_examples


def encode_line(example: dict) -> bytes:
    return (json.dumps(example, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_split(path: Path, examples: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha256()
    document_ids = set()
    example_ids = []
    with path.open("wb") as raw:
        with zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            for example in examples:
                line = encode_line(example)
                compressed.write(line)
                content_hash.update(line)
                document_ids.add(example["source_document_id"])
                example_ids.append(example["example_id"])
    compressed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    documents_hash = hashlib.sha256("\n".join(sorted(document_ids)).encode()).hexdigest()
    examples_hash = hashlib.sha256("\n".join(example_ids).encode()).hexdigest()
    return {
        "examples": len(examples),
        "source_documents": len(document_ids),
        "contested_examples": sum(example["contested"] for example in examples),
        "content_sha256": content_hash.hexdigest(),
        "compressed_sha256": compressed_hash,
        "source_document_ids_sha256": documents_hash,
        "ordered_example_ids_sha256": examples_hash,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive unbiased TinyRime Context v1 ranking splits")
    parser.add_argument("scored_query_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pool-k", type=int, default=8)
    parser.add_argument("--train-examples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    all_recallable = {}
    selected = {}
    for split in ("train", "val", "test"):
        examples = []
        for scored in iter_examples(args.scored_query_dir / f"{split}.jsonl.zst"):
            example = canonical_ranking_example(scored, args.pool_k)
            if example is not None:
                examples.append(example)
        all_recallable[split] = len(examples)
        selected[split] = (
            deterministic_train_sample(examples, args.train_examples, args.seed)
            if split == "train"
            else examples
        )
    split_stats = {
        split: write_split(args.output_dir / f"{split}.jsonl.zst", examples)
        for split, examples in selected.items()
    }
    stats = {
        "benchmark": "TinyRime-Context-v1",
        "candidate_pool_k": args.pool_k,
        "display_pool_k": 8,
        "train_sampling": {
            "method": "lowest stable SHA-256 priority over all recallable train examples",
            "seed": args.seed,
            "requested": args.train_examples,
        },
        "all_recallable_examples": all_recallable,
        "splits": split_stats,
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
