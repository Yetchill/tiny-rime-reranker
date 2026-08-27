from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path

import zstandard

from data_pipeline.build_examples import initialize_worker, score_query
from data_pipeline.benchmark_v1 import StableRecordSample, miss_diagnostic
from data_pipeline.validate_dataset import iter_examples


def parse_recall_k(value: str, top_k: int) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not values or values[0] < 1 or values[-1] > top_k:
        raise argparse.ArgumentTypeError(f"recall-k values must be in [1, {top_k}]")
    return values


def write_zstd_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            for record in records:
                compressed.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyRime Context v1 full-query candidate scorer")
    parser.add_argument("query_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--runner", nargs=argparse.REMAINDER, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--recall-k", default="1,3,5,8,12,16,24,32")
    parser.add_argument("--display-k", type=int, default=8)
    parser.add_argument("--miss-sample-per-split", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    recall_k = parse_recall_k(args.recall_k, args.top_k)
    if not 1 <= args.display_k <= args.top_k:
        raise SystemExit("display-k must be within candidate pool")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = multiprocessing.get_context("spawn").Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(args.runner,),
    )
    profile = {"splits": {}, "top_k": args.top_k, "display_k": args.display_k, "recall_k": recall_k}
    wall_started = time.monotonic()
    try:
        for split_index, split in enumerate(("train", "val", "test")):
            profile["splits"][split] = score_split(
                split=split,
                query_path=args.query_dir / f"{split}.jsonl.zst",
                output_path=args.output_dir / f"{split}.jsonl.zst",
                miss_path=args.output_dir / "misses" / f"{split}.jsonl.zst",
                pool=pool,
                batch_size=args.batch_size,
                recall_k=recall_k,
                top_k=args.top_k,
                display_k=args.display_k,
                miss_sample_size=args.miss_sample_per_split,
                seed=args.seed + split_index,
            )
    finally:
        pool.close()
        pool.join()
    profile["workers"] = args.workers
    profile["batch_size"] = args.batch_size
    profile["wall_seconds"] = time.monotonic() - wall_started
    total_attempts = sum(value["attempts"] for value in profile["splits"].values())
    profile["queries_per_wall_second"] = total_attempts / profile["wall_seconds"] if profile["wall_seconds"] else 0.0
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, sort_keys=True))


def score_split(
    *,
    split: str,
    query_path: Path,
    output_path: Path,
    miss_path: Path,
    pool,
    batch_size: int,
    recall_k: tuple[int, ...],
    top_k: int,
    display_k: int,
    miss_sample_size: int,
    seed: int,
) -> dict:
    attempts = candidate_seconds = 0
    recall_hits = Counter()
    rank_histogram = Counter()
    diagnostic_counts = Counter()
    diagnostics = StableRecordSample(miss_sample_size, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw:
        with zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as output:
                batch = []
                for query in iter_examples(query_path):
                    if query.get("split") != split:
                        raise ValueError(f"query split mismatch: expected {split}, got {query.get('split')}")
                    batch.append(query)
                    if len(batch) >= batch_size:
                        count, elapsed = process_batch(
                            batch,
                            pool,
                            output,
                            recall_k,
                            top_k,
                            display_k,
                            recall_hits,
                            rank_histogram,
                            diagnostic_counts,
                            diagnostics,
                        )
                        attempts += count
                        candidate_seconds += elapsed
                        batch = []
                if batch:
                    count, elapsed = process_batch(
                        batch,
                        pool,
                        output,
                        recall_k,
                        top_k,
                        display_k,
                        recall_hits,
                        rank_histogram,
                        diagnostic_counts,
                        diagnostics,
                    )
                    attempts += count
                    candidate_seconds += elapsed
    write_zstd_jsonl(miss_path, diagnostics.records())
    return {
        "attempts": attempts,
        "candidate_recall": {
            f"recall@{k}": recall_hits[k] / attempts if attempts else 0.0 for k in recall_k
        },
        "gold_rank_histogram": dict(sorted(rank_histogram.items())),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "miss_sample_records": len(diagnostics.records()),
        "candidate_scoring_seconds_sum": candidate_seconds,
        "scored_output_bytes": output_path.stat().st_size,
    }


def process_batch(
    batch: list[dict],
    pool,
    output,
    recall_k: tuple[int, ...],
    top_k: int,
    display_k: int,
    recall_hits: Counter,
    rank_histogram: Counter,
    diagnostic_counts: Counter,
    diagnostics: StableRecordSample,
) -> tuple[int, float]:
    scored = pool.map(score_query, [("".join(query["pinyin"]), query["context"]) for query in batch])
    elapsed_sum = 0.0
    for query, (candidate_values, elapsed) in zip(batch, scored):
        elapsed_sum += elapsed
        candidates = candidate_values[:top_k]
        target_index = next(
            (index for index, candidate in enumerate(candidates) if candidate["text"] == query["target"]),
            -1,
        )
        for k in recall_k:
            recall_hits[k] += 0 <= target_index < k
        rank_histogram["miss" if target_index < 0 else str(target_index + 1)] += 1
        scored_record = {
            **query,
            "candidates": candidates,
            "target_index": target_index,
            "candidate_pool_k": top_k,
        }
        output.write(json.dumps(scored_record, ensure_ascii=False, sort_keys=True) + "\n")
        if target_index < 0 or target_index >= display_k:
            diagnostic = miss_diagnostic(query, candidates, target_index, display_k)
            diagnostic_counts[f"category:{diagnostic['category']}"] += 1
            diagnostic_counts[f"target_length:{diagnostic['target_length_category']}"] += 1
            diagnostic_counts[f"proper_noun_proxy:{str(diagnostic['proper_noun_proxy']).lower()}"] += 1
            diagnostic_counts[f"normalization:{diagnostic['normalization_status']}"] += 1
            diagnostic_counts[f"oov:{diagnostic['oov_status']}"] += 1
            diagnostics.add(diagnostic)
    return len(batch), elapsed_sum


if __name__ == "__main__":
    main()
