"""Build a mechanical training smoke set from Gate 1 output.

This is not an accuracy benchmark: target_index is deliberately the Rime Top-1
to exercise loading, masking, listwise loss, baseline-protection loss, saving,
and evaluation without inventing contextual ground truth.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import zstandard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.fixtures.read_text(encoding="utf-8").splitlines() if line]
    args.output.mkdir(parents=True, exist_ok=True)
    for split, selected in (("train", rows[:80]), ("val", rows[80:])):
        with (args.output / f"{split}.jsonl.zst").open("wb") as raw:
            with zstandard.ZstdCompressor(level=3).stream_writer(raw) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as sink:
                    for index, row in enumerate(selected):
                        example = {
                            "context": row["context"],
                            "pinyin": [row["pinyin"]],
                            "candidates": row["candidates"],
                            "target_index": 0,
                            "source_document_id": f"smoke:{split}:{index}",
                        }
                        sink.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(json.dumps({"purpose": "mechanical-smoke-only", "train": 80, "val": 20}))


if __name__ == "__main__":
    main()
