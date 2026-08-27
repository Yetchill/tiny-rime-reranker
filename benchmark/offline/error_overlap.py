from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_METHODS = ("rime", "wanxiang", "mlp", "tiny_2m", "tiny_4m", "tiny_8m")
DEFAULT_COMPARISONS = (
    ("rime", "wanxiang"),
    ("wanxiang", "tiny_2m"),
    ("wanxiang", "tiny_4m"),
    ("wanxiang", "tiny_8m"),
    ("rime", "tiny_8m"),
)
CONTEXT_BUCKETS = ((0, 4, "0-4"), (5, 8, "5-8"), (9, 16, "9-16"), (17, 32, "17-32"))


class PredictionArtifactError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".zst":
        import io

        import zstandard

        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(path.open("rb")), encoding="utf-8")
    else:
        stream = path.open(encoding="utf-8")
    try:
        records = []
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise PredictionArtifactError(f"{path}:{line_number}: invalid JSON") from error
        return records
    finally:
        stream.close()


def validate_records(records: list[dict], methods: Iterable[str]) -> None:
    methods = tuple(methods)
    seen_ids = set()
    for index, record in enumerate(records):
        location = f"record {index}"
        for field in (
            "example_id",
            "split",
            "source_document_id",
            "context",
            "pinyin",
            "gold",
            "candidates",
            "contested",
            "methods",
        ):
            if field not in record:
                raise PredictionArtifactError(f"{location}: missing field {field!r}")
        if record["example_id"] in seen_ids:
            raise PredictionArtifactError(f"duplicate example_id: {record['example_id']}")
        seen_ids.add(record["example_id"])
        if record["split"] not in {"val", "test"}:
            raise PredictionArtifactError(f"{location}: split must be val or test")
        if not isinstance(record["pinyin"], list) or not record["pinyin"]:
            raise PredictionArtifactError(f"{location}: pinyin must be a non-empty list")
        if not isinstance(record["candidates"], list) or not record["candidates"]:
            raise PredictionArtifactError(f"{location}: candidates must be a non-empty list")
        candidate_texts = []
        for candidate in record["candidates"]:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
                raise PredictionArtifactError(f"{location}: candidate requires string text")
            if "rime_rank" not in candidate or not (
                candidate["rime_rank"] is None
                or isinstance(candidate["rime_rank"], int)
                and candidate["rime_rank"] >= 0
            ):
                raise PredictionArtifactError(f"{location}: candidate requires non-negative or null rime_rank")
            candidate_texts.append(candidate["text"])
        if len(candidate_texts) != len(set(candidate_texts)):
            raise PredictionArtifactError(f"{location}: duplicate candidate text")
        if record["gold"] not in candidate_texts:
            raise PredictionArtifactError(f"{location}: gold is absent from candidate list")
        if not isinstance(record["contested"], bool):
            raise PredictionArtifactError(f"{location}: contested must be boolean")
        if not isinstance(record["methods"], dict):
            raise PredictionArtifactError(f"{location}: methods must be an object")
        for method in methods:
            prediction = record["methods"].get(method)
            if not isinstance(prediction, dict) or not isinstance(prediction.get("text"), str):
                raise PredictionArtifactError(f"{location}: missing prediction text for {method}")
            if prediction["text"] not in candidate_texts:
                raise PredictionArtifactError(f"{location}: {method} predicts outside candidate list")
            for numeric in ("confidence", "margin"):
                if numeric in prediction and (
                    not isinstance(prediction[numeric], (int, float)) or not math.isfinite(prediction[numeric])
                ):
                    raise PredictionArtifactError(f"{location}: invalid {method}.{numeric}")
            if "confidence" in prediction and not 0.0 <= float(prediction["confidence"]) <= 1.0:
                raise PredictionArtifactError(f"{location}: {method}.confidence must be in [0, 1]")


def correct(record: dict, method: str) -> bool:
    return record["methods"][method]["text"] == record["gold"]


def accuracy(records: list[dict], method: str) -> float | None:
    return sum(correct(record, method) for record in records) / len(records) if records else None


def overlap_matrix(records: list[dict], left: str, right: str) -> dict:
    counts = Counter()
    for record in records:
        left_correct = correct(record, left)
        right_correct = correct(record, right)
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif left_correct:
            counts["left_only_correct"] += 1
        elif right_correct:
            counts["right_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
    total = len(records)
    result = {
        "left": left,
        "right": right,
        "samples": total,
        "both_correct": counts["both_correct"],
        "left_only_correct": counts["left_only_correct"],
        "right_only_correct": counts["right_only_correct"],
        "both_wrong": counts["both_wrong"],
    }
    result["proportions"] = {
        key: result[key] / total if total else None
        for key in ("both_correct", "left_only_correct", "right_only_correct", "both_wrong")
    }
    return result


def oracle_metrics(records: list[dict], left: str, right: str) -> dict:
    left_correct = sum(correct(record, left) for record in records)
    right_correct = sum(correct(record, right) for record in records)
    oracle_correct = sum(correct(record, left) or correct(record, right) for record in records)
    total = len(records)
    return {
        "left": left,
        "right": right,
        "samples": total,
        "left_accuracy": left_correct / total if total else None,
        "right_accuracy": right_correct / total if total else None,
        "oracle_accuracy": oracle_correct / total if total else None,
        "oracle_gain_over_left": (oracle_correct - left_correct) / total if total else None,
        "oracle_gain_count": oracle_correct - left_correct,
    }


def win_loss(records: list[dict], baseline: str, method: str) -> dict:
    wins = sum(not correct(record, baseline) and correct(record, method) for record in records)
    losses = sum(correct(record, baseline) and not correct(record, method) for record in records)
    return {"baseline": baseline, "method": method, "wins": wins, "losses": losses, "net_wins": wins - losses}


def context_bucket(record: dict) -> str:
    length = len(record["context"])
    for lower, upper, label in CONTEXT_BUCKETS:
        if lower <= length <= upper:
            return label
    return "33+"


def target_bucket(record: dict) -> str:
    length = len(record["gold"])
    return f"{length} char" if length <= 4 else "5+ chars"


def ambiguity_boundaries(records: list[dict]) -> tuple[int, int]:
    values = sorted(ambiguity_count(record) for record in records)
    if not values:
        return 0, 0
    return values[(len(values) - 1) // 3], values[(len(values) - 1) * 2 // 3]


def ambiguity_count(record: dict) -> int:
    value = record.get("ambiguity_count", len(record["candidates"]))
    if not isinstance(value, int) or value < 1:
        raise PredictionArtifactError(f"{record['example_id']}: ambiguity_count must be a positive integer")
    return value


def grouped_accuracy(
    records: list[dict], methods: Iterable[str], grouping: Callable[[dict], str], expected: Iterable[str] = ()
) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[grouping(record)].append(record)
    labels = list(dict.fromkeys([*expected, *sorted(groups)]))
    return {
        label: {
            "samples": len(groups[label]),
            "accuracy": {method: accuracy(groups[label], method) for method in methods},
        }
        for label in labels
    }


def reliability_bins(records: list[dict], method: str, bins: int = 10) -> list[dict] | None:
    if not records or any("confidence" not in record["methods"][method] for record in records):
        return None
    values = [Counter() for _ in range(bins)]
    for record in records:
        confidence = float(record["methods"][method]["confidence"])
        index = min(bins - 1, max(0, int(confidence * bins)))
        values[index]["samples"] += 1
        values[index]["confidence_sum"] += confidence
        values[index]["correct"] += correct(record, method)
    return [
        {
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "samples": value["samples"],
            "mean_confidence": value["confidence_sum"] / value["samples"] if value["samples"] else None,
            "accuracy": value["correct"] / value["samples"] if value["samples"] else None,
        }
        for index, value in enumerate(values)
    ]


def hybrid_metrics(records: list[dict], baseline: str, challenger: str, threshold: float) -> dict:
    samples = len(records)
    correct_count = wins = losses = changed = correct_promotions = 0
    for record in records:
        base = record["methods"][baseline]["text"]
        candidate = record["methods"][challenger]
        use_challenger = candidate["text"] != base and float(candidate["confidence"]) >= threshold
        prediction = candidate["text"] if use_challenger else base
        is_correct = prediction == record["gold"]
        baseline_correct = base == record["gold"]
        correct_count += is_correct
        wins += not baseline_correct and is_correct
        losses += baseline_correct and not is_correct
        changed += use_challenger
        correct_promotions += use_challenger and is_correct
    contested = [record for record in records if record["contested"]]
    contested_correct = 0
    for record in contested:
        base = record["methods"][baseline]["text"]
        challenger_value = record["methods"][challenger]
        prediction = (
            challenger_value["text"]
            if challenger_value["text"] != base and float(challenger_value["confidence"]) >= threshold
            else base
        )
        contested_correct += prediction == record["gold"]
    return {
        "threshold": threshold,
        "samples": samples,
        "top1": correct_count / samples if samples else None,
        "contested_samples": len(contested),
        "contested_top1": contested_correct / len(contested) if contested else None,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "coverage": changed / samples if samples else None,
        "promotion_precision": correct_promotions / changed if changed else None,
    }


def tune_hybrid_threshold(records: list[dict], baseline: str, challenger: str) -> dict:
    if any(record["split"] != "val" for record in records):
        raise PredictionArtifactError("hybrid threshold tuning accepts val records only")
    if any("confidence" not in record["methods"][challenger] for record in records):
        return {"status": "NOT AVAILABLE", "reason": f"{challenger} confidence is missing"}
    if not records:
        return {"status": "NOT AVAILABLE", "reason": "validation subset is empty"}
    groups: dict[float, list[dict]] = defaultdict(list)
    for record in records:
        if record["methods"][challenger]["text"] != record["methods"][baseline]["text"]:
            groups[float(record["methods"][challenger]["confidence"])].append(record)
    samples = len(records)
    contested_samples = sum(record["contested"] for record in records)
    correct_count = sum(correct(record, baseline) for record in records)
    contested_correct = sum(correct(record, baseline) for record in records if record["contested"])
    wins = losses = changed = correct_promotions = 0

    def current(threshold: float) -> dict:
        return {
            "threshold": threshold,
            "samples": samples,
            "top1": correct_count / samples,
            "contested_samples": contested_samples,
            "contested_top1": contested_correct / contested_samples if contested_samples else None,
            "wins": wins,
            "losses": losses,
            "net_wins": wins - losses,
            "coverage": changed / samples,
            "promotion_precision": correct_promotions / changed if changed else None,
        }

    candidates = [current(1.0000001)]
    for threshold in sorted(groups, reverse=True):
        for record in groups[threshold]:
            baseline_correct = correct(record, baseline)
            challenger_correct = correct(record, challenger)
            correct_count += challenger_correct - baseline_correct
            if record["contested"]:
                contested_correct += challenger_correct - baseline_correct
            wins += not baseline_correct and challenger_correct
            losses += baseline_correct and not challenger_correct
            changed += 1
            correct_promotions += challenger_correct
        candidates.append(current(threshold))

    def objective(item: dict) -> tuple:
        precision = -1.0 if item["promotion_precision"] is None else item["promotion_precision"]
        return item["top1"], precision, -item["coverage"], item["threshold"]

    best = max(candidates, key=objective)
    return {"status": "AVAILABLE", "selected_on": "val", **best}


def deterministic_sample(records: list[dict], maximum: int, seed: int) -> list[dict]:
    ordered = sorted(records, key=lambda record: record["example_id"])
    if len(ordered) <= maximum:
        return ordered
    return sorted(random.Random(seed).sample(ordered, maximum), key=lambda record: record["example_id"])


def feature_summary(records: list[dict]) -> dict:
    if not records:
        return {"samples": 0}
    ranks = []
    for record in records:
        gold_candidate = next(candidate for candidate in record["candidates"] if candidate["text"] == record["gold"])
        ranks.append(gold_candidate["rime_rank"])
    return {
        "samples": len(records),
        "target_length": dict(Counter(map(lambda record: len(record["gold"]), records))),
        "pinyin_length": dict(Counter(map(lambda record: len(record["pinyin"]), records))),
        "context_bucket": dict(Counter(map(context_bucket, records))),
        "ambiguity_count": dict(Counter(map(ambiguity_count, records))),
        "gold_original_rank": dict(Counter(ranks)),
        "contested": sum(record["contested"] for record in records),
        "target_has_ascii_or_digit": sum(any(character.isascii() and character.isalnum() for character in record["gold"]) for record in records),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def analyze(records: list[dict], methods: tuple[str, ...], seed: int) -> dict:
    validate_records(records, methods)
    test = [record for record in records if record["split"] == "test"]
    val = [record for record in records if record["split"] == "val"]
    comparisons = {}
    for left, right in DEFAULT_COMPARISONS:
        if left in methods and right in methods:
            key = f"{left}_vs_{right}"
            comparisons[key] = {
                "overlap": overlap_matrix(test, left, right),
                "oracle": oracle_metrics(test, left, right),
                "win_loss": win_loss(test, left, right),
            }
    low, high = ambiguity_boundaries(test)
    ambiguity = lambda record: "low" if ambiguity_count(record) <= low else (
        "medium" if ambiguity_count(record) <= high else "high"
    )
    contested = [record for record in test if record["contested"]]
    reliability = {method: reliability_bins(test, method) for method in methods}
    hybrid = {"status": "NOT AVAILABLE", "reason": "val and test records are both required"}
    if val and test and "wanxiang" in methods and "tiny_8m" in methods:
        tuned = tune_hybrid_threshold(val, "wanxiang", "tiny_8m")
        hybrid = {"tuning": tuned}
        if tuned["status"] == "AVAILABLE":
            hybrid["test"] = hybrid_metrics(test, "wanxiang", "tiny_8m", tuned["threshold"])
    primary = comparisons.get("wanxiang_vs_tiny_8m")
    primary_quadrants = None
    error_groups = {}
    if primary is not None:
        overlap = primary["overlap"]
        primary_quadrants = {
            "samples": overlap["samples"],
            "both_correct": overlap["both_correct"],
            "wanxiang_only_correct": overlap["left_only_correct"],
            "tiny8_only_correct": overlap["right_only_correct"],
            "both_wrong": overlap["both_wrong"],
            "proportions": {
                "both_correct": overlap["proportions"]["both_correct"],
                "wanxiang_only_correct": overlap["proportions"]["left_only_correct"],
                "tiny8_only_correct": overlap["proportions"]["right_only_correct"],
                "both_wrong": overlap["proportions"]["both_wrong"],
            },
        }
        tiny_only = [record for record in test if not correct(record, "wanxiang") and correct(record, "tiny_8m")]
        wanxiang_only = [record for record in test if correct(record, "wanxiang") and not correct(record, "tiny_8m")]
        error_groups = {
            "wanxiang_wrong_tiny_right": {
                "features": feature_summary(tiny_only),
                "sample": deterministic_sample(tiny_only, 200, seed),
            },
            "wanxiang_right_tiny_wrong": {
                "features": feature_summary(wanxiang_only),
                "sample": deterministic_sample(wanxiang_only, 200, seed + 1),
            },
        }
    return {
        "schema_version": 1,
        "samples": {"val": len(val), "test": len(test), "contested_test": len(contested)},
        "methods": {method: {"test_top1": accuracy(test, method)} for method in methods},
        "comparisons": comparisons,
        "wanxiang_vs_tiny8_quadrants": primary_quadrants,
        "groups": {
            "context_length": grouped_accuracy(test, methods, context_bucket, [label for *_, label in CONTEXT_BUCKETS]),
            "target_length": grouped_accuracy(
                test, methods, target_bucket, ["1 char", "2 char", "3 char", "4 char", "5+ chars"]
            ),
            "ambiguity": {
                "thresholds": {"low_max_candidates": low, "medium_max_candidates": high},
                "groups": grouped_accuracy(test, methods, ambiguity, ["low", "medium", "high"]),
            },
        },
        "contested": {
            "samples": len(contested),
            "methods": {method: accuracy(contested, method) for method in methods},
            "wanxiang_vs_tiny_8m": (
                {
                    "overlap": overlap_matrix(contested, "wanxiang", "tiny_8m"),
                    "oracle": oracle_metrics(contested, "wanxiang", "tiny_8m"),
                }
                if "wanxiang" in methods and "tiny_8m" in methods
                else None
            ),
        },
        "reliability": reliability,
        "simple_hybrid": hybrid,
        "error_groups": error_groups,
    }


def render_markdown(summary: dict) -> str:
    lines = ["# Error overlap analysis", "", f"Test samples: {summary['samples']['test']}", ""]
    primary = summary["comparisons"].get("wanxiang_vs_tiny_8m")
    if primary:
        overlap = primary["overlap"]
        oracle = primary["oracle"]
        lines.extend(
            [
                "## Wanxiang vs Tiny-8M",
                "",
                f"- both correct: {overlap['both_correct']}",
                f"- Wanxiang only: {overlap['left_only_correct']}",
                f"- Tiny-8M only: {overlap['right_only_correct']}",
                f"- both wrong: {overlap['both_wrong']}",
                f"- oracle accuracy: {format_metric(oracle['oracle_accuracy'])}",
                f"- oracle gain over Wanxiang: {format_metric(oracle['oracle_gain_over_left'])}",
                "",
            ]
        )
    hybrid = summary["simple_hybrid"]
    lines.extend(["## Simple hybrid", "", f"```json\n{json.dumps(hybrid, indent=2, sort_keys=True)}\n```", ""])
    return "\n".join(lines)


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    records = read_jsonl(args.predictions)
    summary = analyze(records, methods, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in summary["error_groups"].items():
        sample_file = f"{name.replace('_', '-')}.jsonl"
        write_jsonl(args.output / sample_file, value["sample"])
        value["sample_count"] = len(value["sample"])
        value["sample_file"] = sample_file
        del value["sample"]
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (args.output / "summary.md").write_text(render_markdown(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
