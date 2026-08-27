from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import time
from collections import defaultdict
from pathlib import Path

import zstandard

from data_pipeline.build_examples import initialize_worker, score_query
from data_pipeline.validate_dataset import iter_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Rime Top-K generation over a fixed gold query pool")
    parser.add_argument("query_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--runner", nargs=argparse.REMAINDER, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--val-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=10_000)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    stored = {split: 0 for split in limits}
    attempts = {split: 0 for split in limits}
    misses = {split: 0 for split in limits}
    recall_hits = {split: {1: 0, 3: 0, 5: 0, 8: 0} for split in limits}
    candidate_seconds = 0.0
    recorded_misses = []
    ranking = {
        split: {
            "mrr_sum": 0.0,
            "contested_attempts": 0,
            "contested_top1": 0,
            "contested_mrr_sum": 0.0,
            "wins": 0,
            "losses": 0,
            "reorders": 0,
            "correct_promotions": 0,
        }
        for split in limits
    }
    contested = {split: contested_keys(args.query_dir / f"{split}.jsonl.zst") for split in limits}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {split: (args.output_dir / f"{split}.jsonl.zst").open("wb") for split in limits}
    streams = {
        split: io.TextIOWrapper(zstandard.ZstdCompressor(level=6).stream_writer(file), encoding="utf-8")
        for split, file in files.items()
    }
    pool = multiprocessing.get_context("spawn").Pool(
        processes=args.workers, initializer=initialize_worker, initargs=(args.runner,)
    )
    wall_started = time.monotonic()
    try:
        for split in ("train", "val", "test"):
            batch = []
            for query in iter_examples(args.query_dir / f"{split}.jsonl.zst"):
                batch.append(query)
                if len(batch) < args.batch_size:
                    continue
                candidate_seconds += process_batch(
                    split,
                    batch,
                    pool,
                    streams[split],
                    limits,
                    stored,
                    attempts,
                    misses,
                    recall_hits,
                    recorded_misses,
                    contested[split],
                    ranking[split],
                )
                batch = []
            if batch:
                candidate_seconds += process_batch(
                    split,
                    batch,
                    pool,
                    streams[split],
                    limits,
                    stored,
                    attempts,
                    misses,
                    recall_hits,
                    recorded_misses,
                    contested[split],
                    ranking[split],
                )
    finally:
        pool.close()
        pool.join()
        for stream in streams.values():
            stream.close()
        for file in files.values():
            file.close()
    with (args.output_dir / "candidate_misses.jsonl.zst").open("wb") as raw:
        with zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            for miss in recorded_misses:
                compressed.write((json.dumps(miss, ensure_ascii=False) + "\n").encode())
    wall_seconds = time.monotonic() - wall_started
    profile = {
        "stored_examples": stored,
        "attempts": attempts,
        "misses": misses,
        "candidate_recall": {
            split: {
                f"recall@{k}": recall_hits[split][k] / attempts[split] if attempts[split] else 0.0
                for k in (1, 3, 5, 8)
            }
            for split in limits
        },
        "ranking_metrics": {
            split: {
                "mrr@8": ranking[split]["mrr_sum"] / attempts[split] if attempts[split] else 0.0,
                "contested_attempts": ranking[split]["contested_attempts"],
                "contested_top1": (
                    ranking[split]["contested_top1"] / ranking[split]["contested_attempts"]
                    if ranking[split]["contested_attempts"]
                    else 0.0
                ),
                "contested_mrr@8": (
                    ranking[split]["contested_mrr_sum"] / ranking[split]["contested_attempts"]
                    if ranking[split]["contested_attempts"]
                    else 0.0
                ),
                "wins": ranking[split]["wins"],
                "losses": ranking[split]["losses"],
                "net_wins": ranking[split]["wins"] - ranking[split]["losses"],
                "reorder_coverage": (
                    ranking[split]["reorders"] / attempts[split] if attempts[split] else 0.0
                ),
                "promotion_precision": (
                    ranking[split]["correct_promotions"] / ranking[split]["reorders"]
                    if ranking[split]["reorders"]
                    else 0.0
                ),
            }
            for split in limits
        },
        "workers": args.workers,
        "batch_size": args.batch_size,
        "candidate_scoring_seconds_sum": candidate_seconds,
        "wall_seconds": wall_seconds,
        "queries_per_wall_second": sum(attempts.values()) / wall_seconds if wall_seconds else 0.0,
    }
    if any(stored[split] < limits[split] for split in limits):
        profile["warning"] = "one or more splits did not reach the requested stored-example limit"
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, sort_keys=True))


def contested_keys(path: Path) -> set[str]:
    targets: dict[str, set[str]] = defaultdict(set)
    for query in iter_examples(path):
        targets["/".join(query["pinyin"])].add(query["target"])
    return {key for key, values in targets.items() if len(values) >= 2}


def process_batch(
    split,
    batch,
    pool,
    stream,
    limits,
    stored,
    attempts,
    misses,
    recall_hits,
    recorded_misses,
    contested,
    ranking,
):
    scored = pool.map(score_query, [("".join(query["pinyin"]), query["context"]) for query in batch])
    candidate_seconds = 0.0
    for query, (candidate_values, elapsed) in zip(batch, scored):
        attempts[split] += 1
        candidate_seconds += elapsed
        candidates = candidate_values[:8]
        target_index = next(
            (index for index, candidate in enumerate(candidates) if candidate["text"] == query["target"]), -1
        )
        reciprocal_rank = 0.0 if target_index < 0 else 1.0 / (target_index + 1)
        ranking["mrr_sum"] += reciprocal_rank
        is_contested = "/".join(query["pinyin"]) in contested
        if is_contested:
            ranking["contested_attempts"] += 1
            ranking["contested_top1"] += target_index == 0
            ranking["contested_mrr_sum"] += reciprocal_rank
        if "baseline_target_index" in query:
            baseline_correct = query["baseline_target_index"] == 0
            new_correct = target_index == 0
            ranking["wins"] += not baseline_correct and new_correct
            ranking["losses"] += baseline_correct and not new_correct
            changed = bool(candidates) and candidates[0]["text"] != query.get("baseline_top1")
            ranking["reorders"] += changed
            ranking["correct_promotions"] += changed and new_correct
        if target_index < 0:
            misses[split] += 1
            if len(recorded_misses) < 10_000:
                recorded_misses.append(
                    {
                        **query,
                        "category": "pinyin-or-decoder-empty" if not candidates else "oov-or-below-top8",
                    }
                )
            continue
        for k in (1, 3, 5, 8):
            recall_hits[split][k] += target_index < k
        if stored[split] >= limits[split]:
            continue
        example = {
            "context": query["context"],
            "pinyin": query["pinyin"],
            "candidates": candidates,
            "target_index": target_index,
            "source_document_id": query["source_document_id"],
        }
        stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        stored[split] += 1
    return candidate_seconds


if __name__ == "__main__":
    main()
