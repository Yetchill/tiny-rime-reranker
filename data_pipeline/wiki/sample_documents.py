from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import random
import re
import time
from pathlib import Path
from typing import Iterator

import zstandard

CHINESE = re.compile(r"[\u3400-\u9fff]")


def open_text(path: Path):
    if path.suffix == ".zst":
        reader = zstandard.ZstdDecompressor().stream_reader(path.open("rb"))
        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open(encoding="utf-8")


def jsonl_documents(path: Path, snapshot: str) -> Iterator[dict]:
    with open_text(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            document_id = value.get("id", value.get("source_document_id"))
            if document_id is None:
                raise ValueError(f"{path}:{line_number}: missing document id")
            yield {
                "source_document_id": f"zhwiki:{snapshot}:{document_id}",
                "title": str(value.get("title", "")),
                "tags": value.get("tags", []),
                "text": str(value.get("text", "")),
            }


def xml_bz2_documents(path: Path, snapshot: str) -> Iterator[dict]:
    from lxml import etree

    from data_pipeline.wiki.stream_extract import clean

    with bz2.open(path, "rb") as source:
        for _, element in etree.iterparse(source, events=("end",), tag="{*}page"):
            namespace = element.findtext("{*}ns")
            redirect = element.find("{*}redirect") is not None
            if namespace == "0" and not redirect:
                document_id = element.findtext("{*}id")
                text = element.findtext(".//{*}text") or ""
                if document_id:
                    yield {
                        "source_document_id": f"zhwiki:{snapshot}:{document_id}",
                        "title": element.findtext("{*}title") or "",
                        "tags": [],
                        "text": clean(text),
                    }
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]


def documents(path: Path, source_format: str, snapshot: str) -> Iterator[dict]:
    if source_format == "jsonl":
        yield from jsonl_documents(path, snapshot)
    elif source_format == "xml-bz2":
        yield from xml_bz2_documents(path, snapshot)
    else:
        raise ValueError(f"unsupported source format: {source_format}")


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return sorted(values)[int((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-seed document reservoir sampling over a full source")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-format", choices=("jsonl", "xml-bz2"), required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sample-documents", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--min-chinese-characters", type=int, default=128)
    parser.add_argument("--max-characters-per-document", type=int, default=12_000)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_documents < 1:
        raise SystemExit("sample-documents must be positive")
    if args.source.stat().st_size < 1024 * 1024:
        raise SystemExit("source is unexpectedly small")

    rng = random.Random(args.seed)
    reservoir: list[dict] = []
    input_documents = 0
    eligible_documents = 0
    discarded_short = 0
    started = time.monotonic()
    for document in documents(args.source, args.source_format, args.snapshot):
        input_documents += 1
        text = document["text"].replace("\x00", " ").strip()
        chinese_characters = sum(1 for _ in CHINESE.finditer(text))
        if chinese_characters < args.min_chinese_characters:
            discarded_short += 1
            continue
        document["text"] = text[: args.max_characters_per_document]
        eligible_documents += 1
        if len(reservoir) < args.sample_documents:
            reservoir.append(document)
        else:
            replacement = rng.randrange(eligible_documents)
            if replacement < args.sample_documents:
                reservoir[replacement] = document

    if len(reservoir) != min(args.sample_documents, eligible_documents):
        raise RuntimeError("reservoir size invariant failed")
    ids = [document["source_document_id"] for document in reservoir]
    if len(ids) != len(set(ids)):
        raise RuntimeError("sampled source document IDs are not unique")
    reservoir.sort(
        key=lambda document: hashlib.sha256(
            f"{args.seed}\0{document['source_document_id']}".encode()
        ).digest()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            for document in reservoir:
                compressed.write((json.dumps(document, ensure_ascii=False) + "\n").encode())
    lengths = [len(document["text"]) for document in reservoir]
    split_counts = {"train": 0, "val": 0, "test": 0}
    from data_pipeline.build_examples import split_for

    for document_id in ids:
        split_counts[split_for(document_id)] += 1
    stats = {
        "source": str(args.source),
        "source_bytes": args.source.stat().st_size,
        "source_format": args.source_format,
        "source_sha256": args.source_sha256,
        "snapshot": args.snapshot,
        "seed": args.seed,
        "input_documents": input_documents,
        "eligible_documents": eligible_documents,
        "discarded_short": discarded_short,
        "sampled_documents": len(reservoir),
        "sampled_source_document_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "sampled_split_documents": split_counts,
        "max_characters_per_document": args.max_characters_per_document,
        "sampled_text_length": {
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
            "total": sum(lengths),
        },
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "sampling": "Algorithm R, one pass over every eligible document",
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
