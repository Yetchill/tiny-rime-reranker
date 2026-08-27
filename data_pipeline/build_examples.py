from __future__ import annotations

import argparse
import atexit
import hashlib
import io
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import zstandard
from pypinyin import Style, lazy_pinyin

from data_pipeline.validate_dataset import iter_examples

CHINESE_MIN = 0x3400
CHINESE_MAX = 0x9FFF


def split_for(document_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(document_id.encode()).digest()[:8], "big") % 120
    if bucket < 100:
        return "train"
    if bucket < 110:
        return "val"
    return "test"


def chinese_windows(text: str):
    run = ""
    for character in text:
        if CHINESE_MIN <= ord(character) <= CHINESE_MAX:
            run += character
        else:
            if len(run) >= 4:
                yield run
            run = ""
    if len(run) >= 4:
        yield run


class RimeRunner:
    def __init__(self, command: list[str]):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def candidates(self, pinyin: str, context: str) -> list[dict]:
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps({"pinyin": pinyin, "context": context}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response["candidates"]

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=10)


_worker_runner: RimeRunner | None = None


def initialize_worker(command: list[str]) -> None:
    global _worker_runner
    resolved = [value.replace("{worker}", str(os.getpid())) for value in command]
    if "--user-data" in resolved:
        user_data = Path(resolved[resolved.index("--user-data") + 1])
        user_data.mkdir(parents=True, exist_ok=True)
    _worker_runner = RimeRunner(resolved)
    atexit.register(_worker_runner.close)


def score_query(query: tuple[str, str]) -> tuple[list[dict], float]:
    if _worker_runner is None:
        raise RuntimeError("candidate worker is not initialized")
    pinyin, context = query
    started = time.monotonic()
    candidates = _worker_runner.candidates(pinyin, context)
    return candidates, time.monotonic() - started


def queries_for_document(document_id: str, text: str, maximum: int) -> list[tuple[str, str, list[str], str]]:
    choices = []
    for run_index, run in enumerate(chinese_windows(text)):
        for end in range(4, len(run) + 1, 2):
            length = 2 + (end % 3)
            target = run[max(0, end - length) : end]
            context = run[max(0, end - length - 32) : max(0, end - length)]
            syllables = lazy_pinyin(target, style=Style.NORMAL, errors="ignore")
            if len(syllables) != len(target) or not context:
                continue
            priority = hashlib.sha256(
                f"{document_id}\0{run_index}\0{end}\0{target}\0{context}".encode()
            ).digest()
            choices.append((priority, target, context, syllables))
    choices.sort(key=lambda item: item[0])
    return [(target, context, syllables, "".join(syllables)) for _, target, context, syllables in choices[:maximum]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--runner", nargs=argparse.REMAINDER, required=True)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--val-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-queries-per-document", type=int, default=12)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    counts = {key: 0 for key in limits}
    attempts = {key: 0 for key in limits}
    miss_counts = {key: 0 for key in limits}
    recall_hits = {split: {1: 0, 3: 0, 5: 0, 8: 0} for split in limits}
    misses: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {split: (args.output_dir / f"{split}.jsonl.zst").open("wb") for split in limits}
    streams = {
        split: io.TextIOWrapper(zstandard.ZstdCompressor(level=6).stream_writer(file), encoding="utf-8")
        for split, file in files.items()
    }
    runner = None
    pool = None
    if args.workers > 1:
        pool = multiprocessing.get_context("spawn").Pool(
            processes=args.workers, initializer=initialize_worker, initargs=(args.runner,)
        )
    else:
        runner = RimeRunner(args.runner)
    wall_started = time.monotonic()
    candidate_seconds = 0.0
    try:
        for document in iter_examples(args.documents):
            document_id = str(document["source_document_id"])
            split = split_for(document_id)
            if counts[split] >= limits[split]:
                continue
            document_queries = queries_for_document(
                document_id, str(document["text"]), args.max_queries_per_document
            )
            scored_inputs = [(pinyin, context) for _, context, _, pinyin in document_queries]
            if pool is not None:
                scored = pool.map(score_query, scored_inputs)
            else:
                assert runner is not None
                scored = []
                for pinyin, context in scored_inputs:
                    started = time.monotonic()
                    scored.append((runner.candidates(pinyin, context), time.monotonic() - started))
            for (target, context, syllables, _), (candidate_values, elapsed) in zip(document_queries, scored):
                    if counts[split] >= limits[split]:
                        break
                    pair = (context, target)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    attempts[split] += 1
                    candidate_seconds += elapsed
                    candidates = candidate_values[:8]
                    target_index = next((i for i, item in enumerate(candidates) if item["text"] == target), -1)
                    if target_index < 0:
                        miss_counts[split] += 1
                        if len(misses) < 10_000:
                            misses.append(
                                {
                                    "source_document_id": document_id,
                                    "target": target,
                                    "pinyin": syllables,
                                    "category": (
                                        "pinyin-or-decoder-empty" if not candidates else "oov-or-below-top8"
                                    ),
                                }
                            )
                        continue
                    for k in (1, 3, 5, 8):
                        recall_hits[split][k] += target_index < k
                    row = {
                        "context": context,
                        "pinyin": syllables,
                        "candidates": candidates,
                        "target_index": target_index,
                        "source_document_id": document_id,
                    }
                    streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[split] += 1
            if all(counts[key] >= limits[key] for key in limits):
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if runner is not None:
            runner.close()
        for stream in streams.values():
            stream.close()
        for file in files.values():
            file.close()
    miss_path = args.output_dir / "candidate_misses.jsonl.zst"
    with miss_path.open("wb") as raw, zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
        for miss in misses:
            compressed.write((json.dumps(miss, ensure_ascii=False) + "\n").encode())
    wall_seconds = time.monotonic() - wall_started
    report = {
                "samples": counts,
                "attempts": attempts,
                "misses": miss_counts,
                "candidate_recall": {
                    split: {
                        f"recall@{k}": (recall_hits[split][k] / attempts[split] if attempts[split] else 0.0)
                        for k in (1, 3, 5, 8)
                    }
                    for split in limits
                },
                "recorded_misses": len(misses),
                "workers": args.workers,
                "candidate_scoring_seconds_sum": candidate_seconds,
                "wall_seconds": wall_seconds,
                "queries_per_wall_second": sum(attempts.values()) / wall_seconds if wall_seconds else 0.0,
            }
    if args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        args.profile.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
