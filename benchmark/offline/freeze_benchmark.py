from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_pipeline.validate_dataset import iter_examples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_jsonl_hash(path: Path) -> tuple[str, int, str]:
    content = hashlib.sha256()
    document_ids = set()
    count = 0
    for record in iter_examples(path):
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        content.update(line.encode())
        document_ids.add(str(record["source_document_id"]))
        count += 1
    documents_hash = hashlib.sha256("\n".join(sorted(document_ids)).encode()).hexdigest()
    return content.hexdigest(), count, documents_hash


def ensure_new_manifest(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen benchmark manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze immutable TinyRime-Context-v1 evidence")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--reservoir-stats", type=Path, required=True)
    parser.add_argument("--query-stats", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--scored-query-dir", type=Path, required=True)
    parser.add_argument("--ranking-stats", type=Path, required=True)
    parser.add_argument("--ranking-dir", type=Path, required=True)
    parser.add_argument("--query-generation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        ensure_new_manifest(args.output)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    dataset = json.loads(args.dataset_manifest.read_text())
    reservoir = json.loads(args.reservoir_stats.read_text())
    query_stats = json.loads(args.query_stats.read_text())
    ranking_stats = json.loads(args.ranking_stats.read_text())
    query_splits = {}
    scored_splits = {}
    ranking_splits = {}
    for split in ("train", "val", "test"):
        query_path = args.query_dir / f"{split}.jsonl.zst"
        scored_path = args.scored_query_dir / f"{split}.jsonl.zst"
        ranking_path = args.ranking_dir / f"{split}.jsonl.zst"
        query_content, query_count, query_documents = canonical_jsonl_hash(query_path)
        scored_content, scored_count, scored_documents = canonical_jsonl_hash(scored_path)
        ranking_content, ranking_count, ranking_documents = canonical_jsonl_hash(ranking_path)
        query_splits[split] = {
            "examples": query_count,
            "content_sha256": query_content,
            "compressed_sha256": sha256(query_path),
            "source_document_ids_sha256": query_documents,
        }
        scored_splits[split] = {
            "examples": scored_count,
            "content_sha256": scored_content,
            "compressed_sha256": sha256(scored_path),
            "source_document_ids_sha256": scored_documents,
        }
        ranking_splits[split] = {
            "examples": ranking_count,
            "content_sha256": ranking_content,
            "compressed_sha256": sha256(ranking_path),
            "source_document_ids_sha256": ranking_documents,
        }
    manifest = {
        "schema_version": 1,
        "benchmark": "TinyRime-Context-v1",
        "immutability": "This file must never be overwritten; protocol changes require a new benchmark name.",
        "source": {
            "dataset_repo": dataset["repository"]["id"],
            "snapshot_date": dataset["repository"]["snapshot_date"],
            "revision": dataset["repository"]["revision"],
            "file": dataset["file"]["name"],
            "sha256": dataset["file"]["sha256"],
        },
        "reservoir": {
            "seed": reservoir["seed"],
            "sampled_documents": reservoir["sampled_documents"],
            "source_document_ids_sha256": reservoir["sampled_source_document_ids_sha256"],
        },
        "query_generation": {
            "code_commit": args.query_generation_commit,
            "normalization": [
                "target is a Jieba token containing 2-4 characters in U+3400..U+9FFF",
                "left context is stripped and capped at 32 Python Unicode code points",
                "duplicate context-target pairs are removed globally",
                "maximum 16 queries per document selected by stable SHA-256 priority",
            ],
            "pinyin": "pypinyin Style.NORMAL; lowercase syllables joined by apostrophe for contested keys",
            "contested": "global fixed query pool normalized pinyin has >=2 distinct gold targets",
            "stats": query_stats,
            "splits": query_splits,
        },
        "candidate_generation": {
            "internal_pool_k": 32,
            "measured_recall_k": [1, 3, 5, 8, 12, 16, 24, 32],
            "scored_splits": scored_splits,
            "librime_commit": "13faefe2819d01fce208752c2539744094bb4787",
            "rime_ice_commit": "75e6572bebc05b49021e842949ce947882e3e4b2",
            "librime_octagram_commit": "bfb168ca33d8b372596fdf2007933f3da1cf360e",
            "wanxiang_model_sha256": "01ffe37f22607bf8a5cd5d82a3349f6df97744369464aee4577585112d85469d",
        },
        "ranking_dataset": {
            "candidate_pool_k": ranking_stats["candidate_pool_k"],
            "display_pool_k": ranking_stats["display_pool_k"],
            "train_sampling": ranking_stats["train_sampling"],
            "splits": ranking_splits,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
