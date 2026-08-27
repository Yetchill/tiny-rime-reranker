from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from training.dataset import RimeRankingDataset
from training.evaluate import evaluate
from training.models import PRESETS, TinyContextReranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(PRESETS))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PRESETS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyContextReranker(config).to(device)
    model.load_state_dict(load_file(args.checkpoint, device=str(device)))
    dataset = RimeRankingDataset(args.dataset, config.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)
    metrics = evaluate(model, loader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
