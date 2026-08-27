from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a verified FP16 deployment candidate")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    tensors = {}
    with safe_open(args.source, framework="pt", device="cpu") as archive:
        for key in archive.keys():
            value = archive.get_tensor(key)
            tensors[key] = value.half() if value.is_floating_point() else value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)
    with safe_open(args.output, framework="pt", device="cpu") as archive:
        verified = {key: tuple(archive.get_tensor(key).shape) for key in archive.keys()}
    report = {
        "source": str(args.source),
        "output": str(args.output),
        "source_bytes": args.source.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "tensor_count": len(verified),
        "all_floating_tensors_fp16": all(
            not tensor.is_floating_point() or tensor.dtype == torch.float16 for tensor in tensors.values()
        ),
    }
    args.metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
