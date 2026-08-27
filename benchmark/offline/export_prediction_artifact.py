from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path


MODEL_METHODS = ("linear", "mlp", "tiny_2m", "tiny_4m", "tiny_8m")
PRESET_FOR_METHOD = {
    "linear": "linear",
    "mlp": "mlp",
    "tiny_2m": "tiny-2m",
    "tiny_4m": "tiny-4m",
    "tiny_8m": "tiny-8m",
}


def stable_example_id(split: str, source_document_id: str, context: str, pinyin: list[str], gold: str) -> str:
    value = "\0".join((split, source_document_id, context, "/".join(pinyin), gold))
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def output_stream(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zst":
        import zstandard

        return io.TextIOWrapper(zstandard.ZstdCompressor(level=6).stream_writer(path.open("wb")), encoding="utf-8")
    return path.open("w", encoding="utf-8")


def model_predictions(dataset_dir: Path, split: str, method: str, checkpoint: Path, batch_size: int) -> list[dict]:
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader

    from training.dataset import RimeRankingDataset
    from training.models import PRESETS, TinyContextReranker

    config = PRESETS[PRESET_FOR_METHOD[method]]
    dataset = RimeRankingDataset(dataset_dir / f"{split}.jsonl.zst", config.vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyContextReranker(config).to(device)
    model.load_state_dict(load_file(checkpoint, device=str(device)))
    model.eval()
    results = []
    with torch.inference_mode():
        for batch in loader:
            tensors = {key: value.to(device) for key, value in batch.items()}
            residuals, gate_logits = model(
                tensors["context_ids"],
                tensors["pinyin_ids"],
                tensors["candidate_ids"],
                tensors["numeric_features"],
            )
            mask = tensors["candidate_mask"]
            base = -torch.arange(mask.shape[1], device=device).float().unsqueeze(0).expand_as(residuals)
            final = (base + config.alpha * residuals).masked_fill(~mask, -1e9)
            confidence = gate_logits.sigmoid()
            proposed = final.argmax(dim=1)
            margin = final.gather(1, proposed[:, None]).squeeze(1) - final[:, 0]
            changed = proposed.ne(0) & confidence.ge(0.80) & margin.ge(0.15)
            prediction = torch.where(changed, proposed, torch.zeros_like(proposed))
            for index in range(prediction.shape[0]):
                valid = int(mask[index].sum())
                results.append(
                    {
                        "prediction_index": int(prediction[index]),
                        "proposed_index": int(proposed[index]),
                        "changed": bool(changed[index]),
                        "confidence": float(confidence[index]),
                        "margin": float(margin[index]),
                        "residual_scores": [float(value) for value in residuals[index, :valid]],
                        "final_scores": [float(value) for value in final[index, :valid]],
                    }
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export minimal val/test predictions for offline overlap analysis")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--wanxiang-runner", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        method, separator, path = value.partition("=")
        if not separator or method not in MODEL_METHODS:
            raise SystemExit(f"invalid checkpoint mapping: {value}")
        checkpoints[method] = Path(path)
    missing = set(MODEL_METHODS) - set(checkpoints)
    if missing:
        raise SystemExit(f"missing checkpoints: {sorted(missing)}")

    from data_pipeline.build_examples import RimeRunner
    from data_pipeline.validate_dataset import iter_examples

    if "--user-data" in args.wanxiang_runner:
        Path(args.wanxiang_runner[args.wanxiang_runner.index("--user-data") + 1]).mkdir(parents=True, exist_ok=True)
    runner = RimeRunner(args.wanxiang_runner)
    try:
        with output_stream(args.output) as output:
            for split in ("val", "test"):
                raw = list(iter_examples(args.dataset_dir / f"{split}.jsonl.zst"))
                learned = {
                    method: model_predictions(args.dataset_dir, split, method, checkpoint, args.batch_size)
                    for method, checkpoint in checkpoints.items()
                }
                if any(len(values) != len(raw) for values in learned.values()):
                    raise RuntimeError("model prediction count mismatch")
                targets_by_pinyin: dict[str, set[str]] = {}
                for example in raw:
                    target = example["candidates"][example["target_index"]]["text"]
                    targets_by_pinyin.setdefault("/".join(example["pinyin"]), set()).add(target)
                contested_keys = {key for key, targets in targets_by_pinyin.items() if len(targets) >= 2}
                for index, example in enumerate(raw):
                    gold = example["candidates"][example["target_index"]]["text"]
                    wanxiang = runner.candidates("".join(example["pinyin"]), example["context"])[:8]
                    union = []
                    by_text = {}
                    for rank, candidate in enumerate(example["candidates"]):
                        value = {
                            "text": candidate["text"],
                            "rime_rank": rank,
                            "quality": candidate.get("quality"),
                            "type": candidate.get("type"),
                        }
                        union.append(value)
                        by_text[value["text"]] = value
                    for candidate in wanxiang:
                        if candidate["text"] not in by_text:
                            value = {
                                "text": candidate["text"],
                                "rime_rank": None,
                                "quality": candidate.get("quality"),
                                "type": candidate.get("type"),
                            }
                            union.append(value)
                            by_text[value["text"]] = value
                    methods = {
                        "rime": {"text": example["candidates"][0]["text"]},
                        "wanxiang": {"text": wanxiang[0]["text"] if wanxiang else example["candidates"][0]["text"]},
                    }
                    for method in MODEL_METHODS:
                        value = learned[method][index]
                        candidates = example["candidates"]
                        methods[method] = {
                            "text": candidates[value["prediction_index"]]["text"],
                            "proposed_text": candidates[value["proposed_index"]]["text"],
                            "changed": value["changed"],
                            "confidence": value["confidence"],
                            "margin": value["margin"],
                            "residual_scores": value["residual_scores"],
                            "final_scores": value["final_scores"],
                        }
                    record = {
                        "schema_version": 1,
                        "example_id": stable_example_id(
                            split,
                            example["source_document_id"],
                            example["context"],
                            example["pinyin"],
                            gold,
                        ),
                        "split": split,
                        "source_document_id": example["source_document_id"],
                        "context": example["context"],
                        "pinyin": example["pinyin"],
                        "gold": gold,
                        "candidates": union,
                        "ambiguity_count": len(union),
                        "contested": "/".join(example["pinyin"]) in contested_keys,
                        "methods": methods,
                    }
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
