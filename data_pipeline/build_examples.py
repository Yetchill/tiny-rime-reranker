from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--runner", nargs="+", required=True)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--val-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=10_000)
    args = parser.parse_args()
    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    counts = {key: 0 for key in limits}
    attempts = {key: 0 for key in limits}
    miss_counts = {key: 0 for key in limits}
    misses: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {split: (args.output_dir / f"{split}.jsonl.zst").open("wb") for split in limits}
    streams = {
        split: io.TextIOWrapper(zstandard.ZstdCompressor(level=6).stream_writer(file), encoding="utf-8")
        for split, file in files.items()
    }
    runner = RimeRunner(args.runner)
    try:
        for document in iter_examples(args.documents):
            document_id = str(document["source_document_id"])
            split = split_for(document_id)
            if counts[split] >= limits[split]:
                continue
            for run in chinese_windows(str(document["text"])):
                for end in range(4, len(run) + 1, 3):
                    length = 2 + (end % 3)
                    target = run[max(0, end - length) : end]
                    context = run[max(0, end - length - 32) : max(0, end - length)]
                    syllables = lazy_pinyin(target, style=Style.NORMAL, errors="ignore")
                    if len(syllables) != len(target):
                        continue
                    attempts[split] += 1
                    candidates = runner.candidates("".join(syllables), context)[:8]
                    target_index = next((i for i, item in enumerate(candidates) if item["text"] == target), -1)
                    if target_index < 0:
                        miss_counts[split] += 1
                        if len(misses) < 10_000:
                            misses.append({"source_document_id": document_id, "target": target, "pinyin": syllables})
                        continue
                    row = {
                        "context": context,
                        "pinyin": syllables,
                        "candidates": candidates,
                        "target_index": target_index,
                        "source_document_id": document_id,
                    }
                    streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    if counts[split] >= limits[split]:
                        break
                if counts[split] >= limits[split]:
                    break
            if all(counts[key] >= limits[key] for key in limits):
                break
    finally:
        runner.close()
        for stream in streams.values():
            stream.close()
        for file in files.values():
            file.close()
    miss_path = args.output_dir / "candidate_misses.jsonl.zst"
    with miss_path.open("wb") as raw, zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
        for miss in misses:
            compressed.write((json.dumps(miss, ensure_ascii=False) + "\n").encode())
    print(
        json.dumps(
            {
                "samples": counts,
                "attempts": attempts,
                "misses": miss_counts,
                "recall@8": {
                    split: (counts[split] / attempts[split] if attempts[split] else 0.0) for split in limits
                },
                "recorded_misses": len(misses),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
