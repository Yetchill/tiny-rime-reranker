from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import jieba
import zstandard
from pypinyin import Style, lazy_pinyin

from data_pipeline.build_examples import split_for
from data_pipeline.validate_dataset import iter_examples


def is_chinese_word(value: str) -> bool:
    return 2 <= len(value) <= 4 and all("\u3400" <= character <= "\u9fff" for character in value)


def document_queries(document_id: str, text: str, maximum: int) -> list[dict]:
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
    counts = {split: 0 for split in limits}
    documents_used = {split: set() for split in limits}
    targets_by_pinyin: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[tuple[str, str]] = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {split: (args.output_dir / f"{split}.jsonl.zst").open("wb") for split in limits}
    streams = {
        split: io.TextIOWrapper(zstandard.ZstdCompressor(level=6).stream_writer(file), encoding="utf-8")
        for split, file in files.items()
    }
    try:
        for document in iter_examples(args.documents):
            document_id = str(document["source_document_id"])
            split = split_for(document_id)
            if counts[split] >= limits[split]:
                continue
            for query in document_queries(document_id, str(document["text"]), args.max_queries_per_document):
                if counts[split] >= limits[split]:
                    break
                pair = (query["context"], query["target"])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                streams[split].write(json.dumps(query, ensure_ascii=False) + "\n")
                counts[split] += 1
                documents_used[split].add(document_id)
                targets_by_pinyin["/".join(query["pinyin"])].add(query["target"])
            if all(counts[split] >= limits[split] for split in limits):
                break
    finally:
        for stream in streams.values():
            stream.close()
        for file in files.values():
            file.close()
    if counts != limits:
        raise RuntimeError(f"insufficient document sample for requested query pool: {counts} != {limits}")
    stats = {
        "queries": counts,
        "source_documents": {split: len(values) for split, values in documents_used.items()},
        "contested_pinyin_keys": sum(len(targets) >= 2 for targets in targets_by_pinyin.values()),
        "max_queries_per_document": args.max_queries_per_document,
        "selection": "jieba 2-4 Han-character tokens, fixed SHA-256 priority within each document",
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
