from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from training.dataset import RimeRankingDataset
from training.evaluate import evaluate
from training.models import PRESETS, TinyContextReranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", choices=sorted(PRESETS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.model:
        config["model"] = args.model
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    seed = int(config.get("seed", 20260827))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    model_config = PRESETS[config["model"]]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    train_data = RimeRankingDataset(Path(config["train_data"]), model_config.vocab_size)
    val_data = RimeRankingDataset(Path(config["val_data"]), model_config.vocab_size)
    test_data = (
        RimeRankingDataset(Path(config["test_data"]), model_config.vocab_size)
        if config.get("test_data")
        else None
    )
    train_loader = DataLoader(train_data, batch_size=int(config.get("batch_size", 512)), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=int(config.get("batch_size", 512)), num_workers=2)
    test_loader = (
        DataLoader(test_data, batch_size=int(config.get("batch_size", 512)), num_workers=2)
        if test_data is not None
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("training is remote-GPU-only and CUDA is unavailable")
    model = TinyContextReranker(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 2e-3)), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    best_net_wins = -10**9
    started = time.monotonic()
    history = []
    for epoch in range(int(config.get("epochs", 4))):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                residuals, gate_logits = model(
                    batch["context_ids"], batch["pinyin_ids"], batch["candidate_ids"], batch["numeric_features"]
                )
                base = -torch.arange(model_config.top_k, device=device).float().unsqueeze(0)
                logits = (base + model_config.alpha * residuals).masked_fill(~batch["candidate_mask"], -1e4)
                listwise = functional.cross_entropy(logits, batch["target"])
                gate_target = batch["target"].ne(0).float()
                gate_loss = functional.binary_cross_entropy_with_logits(gate_logits, gate_target)
                correct_baseline = batch["target"].eq(0)
                challenger = residuals[:, 1:].max(dim=1).values
                protection = functional.relu(challenger - residuals[:, 0] + 0.25)
                protection_loss = protection[correct_baseline].mean() if correct_baseline.any() else protection.mean() * 0
                loss = listwise + 0.25 * gate_loss + 0.5 * protection_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())
        metrics = evaluate(model, val_loader)
        metrics.update({"epoch": epoch + 1, "train_loss_sum": epoch_loss})
        history.append(metrics)
        if metrics["net_wins"] > best_net_wins:
            best_net_wins = metrics["net_wins"]
            save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, output_dir / "best.safetensors")
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, output_dir / "last.safetensors")
    model.load_state_dict(load_file(output_dir / "best.safetensors", device=str(device)))
    test_metrics = evaluate(model, test_loader) if test_loader is not None else None
    summary = {
        "model": model_config.to_dict(),
        "parameter_count": model.parameter_count,
        "weight_bytes": (output_dir / "best.safetensors").stat().st_size,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "best_net_wins": best_net_wins,
        "epochs": history,
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
