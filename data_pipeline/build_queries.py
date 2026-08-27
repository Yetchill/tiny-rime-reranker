from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import jieba
import zstandard
from pypinyin import Style, lazy_pinyin

from data_pipeline.build_examples import split_for
from data_pipeline.benchmark_v1 import apply_contested_labels
from data_pipeline.validate_dataset import iter_examples


def is_chinese_word(value: str) -> bool:
    return 2 <= len(value) <= 4 and all("\u3400" <= character <= "\u9fff" for character in value)


def document_queries(document_id: str, text: str, maximum: int, title: str = "") -> list[dict]:
    choices = []
    for target, start, _ in jieba.tokenize(text, mode="default"):
        if not is_chinese_word(target):
            continue
        context = text[max(0, start - 32) : start].strip()
        if not context:
            continue
        syllables = lazy_pinyin(target, style=Style.NORMAL, errors="ignore")
        if len(syllables) != len(target):
            continue
        priority = hashlib.sha256(f"{document_id}\0{start}\0{target}\0{context}".encode()).digest()
        choices.append(
            (
                priority,
                {
                    "context": context,
                    "pinyin": syllables,
                    "target": target,
                    "source_document_id": document_id,
                    "proper_noun_proxy": bool(title and target in title),
                },
            )
        )
    choices.sort(key=lambda item: item[0])
    return [query for _, query in choices[:maximum]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed query pool before candidate generation")
    parser.add_argument("documents", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-queries", type=int, default=250_000)
    parser.add_argument("--val-queries", type=int, default=25_000)
    parser.add_argument("--test-queries", type=int, default=25_000)
    parser.add_argument("--max-queries-per-document", type=int, default=16)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    limits = {"train": args.train_queries, "val": args.val_queries, "test": args.test_queries}
    queries_by_split = {split: [] for split in limits}
    documents_used = {split: set() for split in limits}
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    for document in iter_examples(args.documents):
        document_id = str(document["source_document_id"])
        split = split_for(document_id)
        if len(queries_by_split[split]) >= limits[split]:
            continue
        for query in document_queries(
            document_id,
            str(document["text"]),
            args.max_queries_per_document,
            title=str(document.get("title", "")),
        ):
            if len(queries_by_split[split]) >= limits[split]:
                break
            pair = (query["context"], query["target"])
            if pair in seen_pairs:
                duplicate_pairs += 1
                continue
            seen_pairs.add(pair)
            queries_by_split[split].append(query)
            documents_used[split].add(document_id)
        if all(len(queries_by_split[split]) >= limits[split] for split in limits):
            break
    counts = {split: len(queries) for split, queries in queries_by_split.items()}
    if counts != limits:
        raise RuntimeError(f"insufficient document sample for requested query pool: {counts} != {limits}")
    contested_keys = apply_contested_labels(queries_by_split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, queries in queries_by_split.items():
        with (args.output_dir / f"{split}.jsonl.zst").open("wb") as file:
            with zstandard.ZstdCompressor(level=6).stream_writer(file) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    for query in queries:
                        stream.write(json.dumps(query, ensure_ascii=False, sort_keys=True) + "\n")
    stats = {
        "queries": counts,
        "source_documents": {split: len(values) for split, values in documents_used.items()},
        "contested_pinyin_keys": len(contested_keys),
        "contested_examples": sum(query["contested"] for queries in queries_by_split.values() for query in queries),
        "duplicate_context_target_pairs_discarded": duplicate_pairs,
        "max_queries_per_document": args.max_queries_per_document,
        "selection": "jieba 2-4 Han-character tokens, fixed SHA-256 priority within each document",
        "contested_definition": "global fixed query pool: normalized pinyin has at least two distinct gold targets",
        "pinyin_normalization": "lowercase syllables joined by apostrophe; pypinyin Style.NORMAL",
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
