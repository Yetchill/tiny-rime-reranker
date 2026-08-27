from __future__ import annotations

from collections import Counter

import torch


def reciprocal_rank(order: torch.Tensor, target: int) -> float:
    return 1.0 / (int((order == target).nonzero(as_tuple=False)[0]) + 1)


@torch.inference_mode()
def evaluate(model, loader, confidence_threshold: float = 0.80, margin_threshold: float = 0.15) -> dict:
    model.eval()
    counts = Counter()
    confidence_bins = [Counter() for _ in range(10)]
    contested_results = []
    for batch in loader:
        device = next(model.parameters()).device
        tensors = {key: value.to(device) for key, value in batch.items()}
        residuals, gate_logits = model(
            tensors["context_ids"], tensors["pinyin_ids"], tensors["candidate_ids"], tensors["numeric_features"]
        )
        mask = tensors["candidate_mask"]
        base_scores = -torch.arange(mask.shape[1], device=device).float().unsqueeze(0).expand_as(residuals)
        final_scores = base_scores + model.config.alpha * residuals
        final_scores = final_scores.masked_fill(~mask, -1e9)
        confidence = gate_logits.sigmoid()
        proposed = final_scores.argmax(dim=1)
        margin = final_scores.gather(1, proposed[:, None]).squeeze(1) - final_scores[:, 0]
        changed = proposed.ne(0) & confidence.ge(confidence_threshold) & margin.ge(margin_threshold)
        predictions = torch.where(changed, proposed, torch.zeros_like(proposed))
        for index in range(predictions.shape[0]):
            target = int(tensors["target"][index])
            prediction = int(predictions[index])
            order = (
                torch.argsort(final_scores[index], descending=True)
                if bool(changed[index])
                else torch.arange(mask.shape[1], device=device)[mask[index]]
            )
            counts["samples"] += 1
            counts["baseline_top1"] += target == 0
            counts["model_top1"] += prediction == target
            counts["model_top3"] += target in order[:3].tolist()
            counts["mrr_sum"] += reciprocal_rank(order, target)
            counts["wins"] += target != 0 and prediction == target
            counts["losses"] += target == 0 and prediction != 0
            counts["reorders"] += bool(changed[index])
            counts["correct_promotions"] += bool(changed[index]) and prediction == target
            if bool(tensors["contested"][index]):
                contested_results.append((target, prediction, reciprocal_rank(order, target)))
            probability = float(confidence[index])
            bin_index = min(9, int(probability * 10))
            confidence_bins[bin_index]["count"] += 1
            confidence_bins[bin_index]["probability_sum"] += probability
            confidence_bins[bin_index]["correct"] += target != 0
    n = counts["samples"] or 1
    contested_n = len(contested_results) or 1
    ece = 0.0
    for values in confidence_bins:
        if values["count"]:
            accuracy = values["correct"] / values["count"]
            mean_probability = values["probability_sum"] / values["count"]
            ece += values["count"] / n * abs(accuracy - mean_probability)
    return {
        "samples": counts["samples"],
        "recall@8_ceiling": 1.0,
        "baseline_top1": counts["baseline_top1"] / n,
        "top1": counts["model_top1"] / n,
        "top3": counts["model_top3"] / n,
        "mrr": counts["mrr_sum"] / n,
        "contested_samples": len(contested_results),
        "contested_top1": sum(prediction == target for target, prediction, _ in contested_results) / contested_n,
        "contested_mrr": sum(mrr for _, _, mrr in contested_results) / contested_n,
        "wins": counts["wins"],
        "losses": counts["losses"],
        "net_wins": counts["wins"] - counts["losses"],
        "reorder_coverage": counts["reorders"] / n,
        "promotion_precision": (
            counts["correct_promotions"] / counts["reorders"] if counts["reorders"] else 0.0
        ),
        "calibration_ece": ece,
    }
