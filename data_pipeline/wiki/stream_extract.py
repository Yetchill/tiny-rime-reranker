from __future__ import annotations

import argparse
import bz2
import json
import re
from pathlib import Path

from lxml import etree
import zstandard

CHINESE_RUN = re.compile(r"[\u3400-\u9fff][\u3400-\u9fff，。！？；：、“”‘’（）《》\s]{30,}")
MARKUP = re.compile(r"\{\{.*?\}\}|\[\[(?:[^]|]*\|)?([^]]+)\]\]|<[^>]+>|={2,}[^=]+={2,}", re.S)


def clean(text: str) -> str:
    text = MARKUP.sub(lambda match: match.group(1) or " " if match.lastindex else " ", text)
    return "\n".join("".join(run.split()) for run in CHINESE_RUN.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a pages-articles XML.bz2 directly to JSONL.zst")
    parser.add_argument("dump", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with bz2.open(args.dump, "rb") as source, args.output.open("wb") as raw:
        with zstandard.ZstdCompressor(level=6).stream_writer(raw) as sink:
            for _, element in etree.iterparse(source, events=("end",), tag="{*}page"):
                namespace = element.findtext("{*}ns")
                text = element.findtext(".//{*}text") or ""
                document_id = element.findtext("{*}id") or ""
                cleaned = clean(text) if namespace == "0" else ""
                if cleaned:
                    row = {"source_document_id": f"zhwiki:{document_id}", "text": cleaned}
                    sink.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
                    written += 1
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]
                if args.max_documents and written >= args.max_documents:
                    break
    print(json.dumps({"documents": written, "output": str(args.output)}))


if __name__ == "__main__":
    main()
