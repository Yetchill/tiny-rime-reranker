from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROHIBITED_PARTS = {
    "corpus",
    "datasets",
    "candidate-cache",
    "optimizer-state",
    "training-tmp",
    "remote-build-cache",
    "pip-cache",
    "raw-logs",
    "upstream",
}
PROHIBITED_SUFFIXES = {".bz2", ".xml", ".tar", ".gz"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review an explicit AutoDL-to-Mac transfer set")
    parser.add_argument("root", type=Path)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    files: list[Path] = []
    for relative in args.paths:
        target = (args.root / relative).resolve()
        if args.root.resolve() not in target.parents and target != args.root.resolve():
            raise SystemExit(f"path escapes root: {relative}")
        files.extend(path for path in ([target] if target.is_file() else target.rglob("*")) if path.is_file())
    rows = []
    prohibited = []
    total = 0
    for path in sorted(set(files)):
        relative = path.relative_to(args.root.resolve())
        size = path.stat().st_size
        row = {"path": str(relative), "bytes": size, "sha256": sha256(path)}
        rows.append(row)
        total += size
        if PROHIBITED_PARTS.intersection(relative.parts) or path.suffix.lower() in PROHIBITED_SUFFIXES:
            prohibited.append(str(relative))
    result = {
        "root": str(args.root.resolve()),
        "total_bytes": total,
        "over_500mb_review_required": total > 500 * 1024 * 1024,
        "reason_if_over_500mb": args.reason,
        "prohibited_matches": prohibited,
        "files": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("total_bytes", "over_500mb_review_required", "prohibited_matches")}))
    if prohibited or (result["over_500mb_review_required"] and not args.reason.strip()):
        sys.exit(2)


if __name__ == "__main__":
    main()
