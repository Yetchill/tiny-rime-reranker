from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from data_pipeline.validate_dataset import iter_examples


def bounded_union(rime: list[dict], wanxiang: list[dict], budget: int) -> list[dict]:
    if budget < 1:
        raise ValueError("union budget must be positive")
    result = []
    by_text = {}
    maximum = max(len(rime), len(wanxiang))
    for rank in range(maximum):
        for source, candidates in (("rime", rime), ("wanxiang", wanxiang)):
            if rank >= len(candidates):
                continue
            candidate = candidates[rank]
            text = candidate["text"]
            if text in by_text:
                by_text[text][f"{source}_rank"] = rank
                continue
            value = {
                "text": text,
                "rime_rank": rank if source == "rime" else None,
                "wanxiang_rank": rank if source == "wanxiang" else None,
            }
            result.append(value)
            by_text[text] = value
            if len(result) >= budget:
                return result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic bounded Rime/Wanxiang candidate unions")
    parser.add_argument("--rime-scored-dir", type=Path, required=True)
    parser.add_argument("--wanxiang-scored-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="16,24,32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value.strip()}))
    report = {"budgets": budgets, "splits": {}}
    for split in ("train", "val", "test"):
        attempts = 0
        hits = {budget: 0 for budget in budgets}
        rime_top1 = wanxiang_top1 = 0
        rime_records = iter_examples(args.rime_scored_dir / f"{split}.jsonl.zst")
        wanxiang_records = iter_examples(args.wanxiang_scored_dir / f"{split}.jsonl.zst")
        for rime, wanxiang in itertools.zip_longest(rime_records, wanxiang_records):
            if rime is None or wanxiang is None or rime["example_id"] != wanxiang["example_id"]:
                raise ValueError(f"Rime/Wanxiang scored-query alignment failed in {split}")
            if rime["target"] != wanxiang["target"]:
                raise ValueError(f"gold mismatch for {rime['example_id']}")
            attempts += 1
            gold = rime["target"]
            rime_top1 += bool(rime["candidates"] and rime["candidates"][0]["text"] == gold)
            wanxiang_top1 += bool(wanxiang["candidates"] and wanxiang["candidates"][0]["text"] == gold)
            for budget in budgets:
                union = bounded_union(rime["candidates"], wanxiang["candidates"], budget)
                hits[budget] += any(candidate["text"] == gold for candidate in union)
        report["splits"][split] = {
            "attempts": attempts,
            "rime_top1": rime_top1 / attempts if attempts else 0.0,
            "wanxiang_top1": wanxiang_top1 / attempts if attempts else 0.0,
            "union_recall": {
                str(budget): hits[budget] / attempts if attempts else 0.0 for budget in budgets
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
