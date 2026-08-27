from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a safetensors container without a local PyTorch install")
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    file_size = args.path.stat().st_size
    digest = hashlib.sha256(args.path.read_bytes()).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise SystemExit("SHA-256 mismatch")
    with args.path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        if header_size <= 2 or header_size > file_size - 8:
            raise SystemExit("invalid safetensors header size")
        header = json.loads(source.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    maximum_end = 0
    for name, value in tensors.items():
        start, end = value["data_offsets"]
        if not 0 <= start <= end:
            raise SystemExit(f"invalid offsets for {name}")
        maximum_end = max(maximum_end, end)
    if 8 + header_size + maximum_end != file_size:
        raise SystemExit("tensor data does not exactly cover the container")
    print(
        json.dumps(
            {
                "path": str(args.path),
                "bytes": file_size,
                "sha256": digest,
                "tensor_count": len(tensors),
                "dtypes": sorted({value["dtype"] for value in tensors.values()}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
