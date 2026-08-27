from __future__ import annotations

import argparse
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import zstandard

TITLES = """
人工智能 汉语拼音 输入法 自然语言处理 机器学习 深度学习 人工神经网络
中文维基百科 计算机科学 软件工程 苹果公司 中华人民共和国 北京市 上海市
互联网 数据库 操作系统 信息安全 隐私权 语言模型 大型语言模型 图形处理器
中央处理器 开放源代码 编程语言 Python C++ Swift macOS 中文
""".split()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a small public Wikipedia reproducibility fixture")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|revisions",
            "explaintext": "1",
            "exsectionformat": "plain",
            "rvprop": "ids|timestamp",
            "redirects": "1",
            "titles": "|".join(TITLES),
        }
    )
    request = urllib.request.Request(
        "https://zh.wikipedia.org/w/api.php?" + query,
        headers={"User-Agent": "TinyRime-research/0.1 (public reproducibility fixture)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("wb") as raw, zstandard.ZstdCompressor(level=6).stream_writer(raw) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8") as sink:
            for page in payload["query"]["pages"]:
                text = page.get("extract", "")
                if page.get("missing") or not text:
                    continue
                revision = (page.get("revisions") or [{}])[0]
                row = {
                    "source_document_id": f"zhwiki:{page['pageid']}:{revision.get('revid', 'unknown')}",
                    "title": page["title"],
                    "revision_timestamp": revision.get("timestamp"),
                    "source_url": "https://zh.wikipedia.org/?curid=" + str(page["pageid"]),
                    "text": text,
                }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(json.dumps({"documents": written, "license": "CC-BY-SA-4.0/GFDL", "output": str(args.output)}))


if __name__ == "__main__":
    main()
