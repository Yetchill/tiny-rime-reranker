# Third-party software notices

No third-party source is vendored in this repository. The following projects were inspected or used externally during research; a future distribution must carry complete notices for anything it bundles.

| Component | License | Current use |
|---|---|---|
| librime | BSD-3-Clause | External build dependency and API reference |
| Squirrel | GPL-3.0 | Source analysis only |
| rime-ice | GPL-3.0-only | External baseline data/config; no dictionary vendored |
| RIME-LMDG | CC-BY-4.0 | External Wanxiang grammar baseline on AutoDL; no model vendored or copied to Mac |
| librime-octagram | GPL-3.0 | External AutoDL build for the measured grammar baseline; no source vendored |
| librime-ai-predict | BSD-3-Clause | Source analysis only |
| Cassotis IME | GPL-3.0 | Source analysis only; no code/model copied |
| OpenIME | No license file found at locked commit | Not used |

Transitive libraries in the external librime build retain their upstream licenses, including Boost, glog, LevelDB, marisa-trie, OpenCC, yaml-cpp, and GoogleTest. Consult the locked upstream tree before distributing binaries.

## Python research toolchain

These packages are not vendored. They are used in the remote data/training environment or optional development environment and retain their own licenses.

| Package | License | Use |
|---|---|---|
| PyTorch | BSD-style; distributed wheels include multiple separately licensed components | Remote training/evaluation only; not required by the intended native runtime |
| NumPy | BSD-3-Clause plus separately identified bundled components | Remote training/data statistics |
| PyYAML | MIT | Experiment configuration |
| python-zstandard | BSD-3-Clause | Streaming JSONL.zst datasets and reports |
| lxml | BSD-3-Clause | Optional streaming Wikimedia XML parser |
| pypinyin | MIT | Gold target to pinyin conversion |
| jieba | MIT | Deterministic 2–4 character word tokenization for the benchmark query pool |
| safetensors | Apache-2.0 | Checkpoint serialization |
| huggingface_hub | Apache-2.0 | Remote-only dataset download utility |
| pytest | MIT | Development tests |

TinyRime's own code is released under the BSD-3-Clause license in `LICENSE`. This does not relicense external GPL/CC/data components; each remains governed by the license listed above.
