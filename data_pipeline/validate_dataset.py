from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, TextIO

from tinyrime import RankingExample


def open_text(path: Path) -> TextIO:
    if path.suffix == ".zst":
        import zstandard

        reader = zstandard.ZstdDecompressor().stream_reader(path.open("rb"))
        import io

        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_examples(path: Path) -> Iterator[dict]:
    with open_text(path) as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def validate(paths: list[Path]) -> dict:
    documents_by_split: dict[str, set[str]] = {}
    exact_seen: set[str] = set()
    pair_seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    recall = Counter()
    contested: Counter[str] = Counter()

    for path in paths:
        split = path.name.split(".", 1)[0]
        documents = documents_by_split.setdefault(split, set())
        for raw in iter_examples(path):
            example = RankingExample.from_dict(raw)
            documents.add(example.source_document_id)
            canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            if digest in exact_seen:
                raise ValueError(f"exact duplicate sample in {path}: {digest}")
            exact_seen.add(digest)
            target = example.candidates[example.target_index].text
            pair = (example.context, target)
            if pair in pair_seen:
                raise ValueError(f"duplicate context-target pair in {path}: {pair!r}")
            pair_seen.add(pair)
            counts[split] += 1
            for k in (1, 3, 5, 8):
                if example.target_index < min(k, len(example.candidates)):
                    recall[f"recall@{k}"] += 1
            contested["/".join(example.pinyin)] += 1

    splits = sorted(documents_by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = documents_by_split[left] & documents_by_split[right]
            if overlap:
                raise ValueError(f"source-document leakage between {left} and {right}: {len(overlap)}")
    total = sum(counts.values())
    return {
        "samples": dict(counts),
        "documents": {split: len(values) for split, values in documents_by_split.items()},
        "candidate_recall": {key: (value / total if total else 0.0) for key, value in recall.items()},
        "contested_pinyin_keys": sum(value >= 2 for value in contested.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.paths)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
