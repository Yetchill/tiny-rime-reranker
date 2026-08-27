# Session final report

## Environment

- Mac: macOS 26.6.2 (25G83), arm64 Darwin 25.6.0, Apple clang 21.0.0.
- AutoDL: Ubuntu 22.04.5, 12 allocated vCPU, 43GB allocated RAM, RTX 3090 24GB, driver 580.76.05.
- Python 3.12.3, PyTorch 2.8.0+cu128, CUDA available, CUDA runtime 12.8.
- Remote project peak observed: 420MB on the 50GB `/root/autodl-tmp` data disk; free space remained approximately 50GB by rounded `df` output.
- Local project size before final reports: 748KB; the temporary 1.1GB upstream review checkout in `/tmp` was deleted after SHA/license/source inspection.

## Engineering

- Initialized branch `codex/tinyrime-v0` with deny-by-default data/checkpoint ignores and explicit Mac disk policy.
- Built locked librime statically, deployed a minimal deterministic rime-ice schema, and implemented a streaming JSONL Top-8 C API runner.
- Captured and locally checksum-verified 100 real candidate fixtures (74,813 bytes).
- Added Python schema validation, stable document-first splitting, streaming XML.bz2-to-JSONL.zst extraction, Rime-backed example generation, miss accounting, Recall@8 accounting, duplicate/leakage validation, and contested-key statistics.
- Added conservative gate implementations in Python and C++, Mock/NativeMLP backend contracts, a filter-style librime prototype that reorders genuine candidate objects, and a Core ML backend boundary.
- Added Linear, MLP, and approximately 2M/4M/8M tiny encoders; CUDA FP16 listwise training, baseline-protection loss, confidence gate, metrics, safetensors best/last saves, and verified FP16 export.
- Tests: remote CTest 1/1 passed; remote pytest 13 passed; calibrated model-shape suite 5 passed; Mac native runtime test compiled and passed. The librime filter/module source compiled to object files against the locked headers (upstream warning noise only).
- Git: branch `codex/tinyrime-v0`, ten milestone commits including this final report; final clean status was verified after committing.

## Dataset

- Research dataset: **NOT BUILT** because both AutoDL Wikipedia access and Mac API fallback failed.
- Mechanical smoke only: 80 train / 20 val, documents split and validated, all labels deliberately preserve Rime Top-1. This is not an accuracy benchmark and is not on the Mac.
- Candidate Recall@1/3/5/8: **NOT YET MEASURED** on independently sourced gold data.
- Contested research subset: **NOT BUILT**.
- Reproducibility fixture: 100 public/non-personal pinyin requests and real rime-ice Top-8 outputs, 74,813 bytes.

## Experiments

The scientific first sweep was not run. Full smoke metadata is in `reports/remote-smoke/`; its 12 transferred files total 8,659 bytes and were independently checked against the remote manifest. See `reports/first_sweep.md` for the non-result table and explicit answers.

No optimizer state, Wikipedia content, complete corpus/dataset, candidate cache, remote build cache, raw complete logs, or smoke/model weight was copied to the Mac.

## Deployment

- macOS synthetic NativeMLP runtime microbenchmark: P50 0.0367ms, P95 0.0417ms, P99 0.0456ms, max 0.1139ms over 10,000 iterations. This is not trained-model or end-to-end Squirrel latency.
- Tiny ~8M FP16 export smoke size: 14,865,276 bytes, verified remotely; not retained locally because it is not a valid deployment candidate.
- Core ML conversion: **NOT RUN** (correctly gated on real offline gain).
- Core ML CPU/NE/ALL latency: **NOT YET MEASURED**.
- Incremental RAM, idle CPU, and end-to-end candidate latency: **NOT YET MEASURED**.
- Squirrel system integration: **NOT COMPLETE**; this remains a research prototype.

## Remaining risks

- No gold candidate-recall or contextual-ranking evidence exists yet.
- Wanxiang/octagram has not been built or measured on the same split.
- The locked librime revision requires Boost >=1.77; Ubuntu 22.04 supplied 1.74, so Gate 1 disabled external dynamic-plugin loading and used a static core build.
- Public C API candidates omit C++ quality/type, so headless fixtures contain `null` for those fields; the C++ filter can access them.
- The filter currently supplies raw composition input as one pinyin token; production integration needs the schema's syllable segmentation.
- Model calibration, real promotion precision, deadline behavior under Squirrel load, and model/backend lifecycle require real data and integration tests.
- Training-data provenance must be reviewed before any weight is described as redistributable.

## Next single best step

Stage one pinned Chinese Wikipedia `pages-articles` shard directly on AutoDL, build and validate the true document-split 100k/10k/10k Rime candidate dataset, and measure Recall@1/3/5/8 before starting the first sweep.

## Remote shutdown

Remote shutdown requested at: **2026-08-27 22:06:02 CST**. The SSH connection closed immediately, as expected; the instance was not contacted again.
