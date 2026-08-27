from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import zstandard

from data_pipeline.validate_dataset import iter_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover the fixed gold queries represented by a candidate dataset")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        count = 0
        with (args.output_dir / f"{split}.jsonl.zst").open("wb") as raw:
            with zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as sink:
                    for example in iter_examples(args.dataset_dir / f"{split}.jsonl.zst"):
                        target = example["candidates"][example["target_index"]]["text"]
                        query = {
                            "context": example["context"],
                            "pinyin": example["pinyin"],
                            "target": target,
                            "source_document_id": example["source_document_id"],
                            "baseline_target_index": example["target_index"],
                            "baseline_top1": example["candidates"][0]["text"],
                        }
                        sink.write(json.dumps(query, ensure_ascii=False) + "\n")
                        count += 1
        counts[split] = count
    print(json.dumps({"queries": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
