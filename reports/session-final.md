# Session final report

## Environment

- Mac: macOS 26.6.2 (25G83), arm64 Darwin 25.6.0, Apple clang 21.0.0.
- AutoDL: Ubuntu 22.04.5, 12 allocated vCPU, 43GB allocated RAM, RTX 3090 24GB, driver 580.76.05.
- Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8 available.
- Remote project after the real sweep: 3.4GB on the 50GB data disk with 47GB free. The >=15GB guard was never approached.
- Mac repository plus selected ignored deployment artifact is under 20MB. No corpus, complete dataset, candidate cache, Wanxiang model, optimizer state, build cache, or raw training log was copied to Mac.

## Engineering

- The HF JSONL adapter preserves snapshot-qualified document IDs and observed `id/title/tags/text` fields.
- Fixed-seed Algorithm R reservoir sampling scans the complete source rather than stopping at its head.
- Document ID determines train/val/test before token/window generation.
- Jieba 2–4 Han-character tokens form a fixed gold query pool shared by both candidate baselines.
- Candidate scoring supports isolated multi-process librime workers: one read-only prebuilt dictionary and one user-data directory per worker. Profiling discovered and eliminated unsafe shared LevelDB access before valid output was accepted.
- The headless runner injects left context into librime `CommitHistory`, so octagram contextual scoring is real rather than an echoed JSON field.
- Linear, MLP, and approximately 2M/4M/8M encoders train with CUDA FP16, listwise loss, baseline-protection loss, conservative confidence/margin gating, best/last safetensors, and held-out test evaluation.
- Remote pytest: 15 passed. Both base and octagram runners compiled and executed. The selected FP16 safetensors container was independently parsed and checksum-verified on the Mac.

## Dataset

- Repository: `fjcanyue/wikipedia-zh-cn`, HF revision `38a697eb24e84c569ce05cb5f23336bdeb6a94c3`.
- Snapshot/file: `wikipedia-zh-cn-20260501.json`, 2,393,692,848 bytes, 1,489,790 JSONL rows.
- SHA-256: `c8c719a84d402371ffa6b99b57bc9bc524bf66e07d72dfc724e51d0224eaee62`.
- License statement from the dataset card: Chinese Wikipedia under GFDL 1.3 and CC BY-SA 4.0, with possible individual exceptions. Experimental weights remain research-only pending provenance review.
- Full-source scan: 715,511 eligible documents; fixed-seed reservoir: 30,000 documents and 29,230,223 retained text characters, compressed to 34,143,466 bytes.
- Fixed query pool: 250,000 train / 25,000 val / 25,000 test; 8,946 contested pinyin keys.
- Candidate dataset: 100,000 train / 10,000 val / 10,000 test, from 7,163 / 719 / 722 source documents; 2,789 retained contested keys. Leakage and duplicate validation passed.

## Candidate baselines

On all 25,000 fixed test queries:

| Baseline | Recall@1 | Recall@3 | Recall@5 | Recall@8 | QPS (8 workers) |
|---|---:|---:|---:|---:|---:|
| Rime + rime-ice | 73.704% | 85.068% | 87.396% | 88.320% | 7,505 |
| + Wanxiang octagram | 78.148% | 86.396% | 88.112% | 88.808% | 5,378 |

The external Wanxiang model is 420,250,668 bytes, SHA-256 `01ffe37f22607bf8a5cd5d82a3349f6df97744369464aee4577585112d85469d`. It remains only on AutoDL.

## Experiments

On the common 10,000-example Rime Recall@8 test set:

| Model | Top-1 | Contested Top-1 | Wins | Losses | Net wins | Coverage | Promotion precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rime | 84.050% | 57.116% | 0 | 0 | 0 | 0% | N/A |
| Wanxiang | **88.470%** | **69.476%** | 551 | 109 | **442** | 7.230% | 76.210% |
| Linear | 84.050% | 57.116% | 0 | 0 | 0 | 0% | 0% |
| MLP | 84.660% | 57.116% | 61 | 0 | 61 | 0.680% | 89.706% |
| Tiny ~2M | 86.570% | 59.925% | 257 | 5 | 252 | 2.850% | 90.175% |
| Tiny ~4M | 86.450% | 59.176% | 243 | 3 | 240 | 2.610% | **93.103%** |
| Tiny ~8M | **87.140% neural** | **60.674% neural** | 316 | 7 | **309 neural** | 3.530% | 89.518% |

Wanxiang beats Tiny ~8M on Top-1, contested metrics, MRR, and net wins, while the neural models are substantially more conservative. Scaling is non-monotonic: 4M is slightly below 2M, and 8M recovers a meaningful but insufficient gain. The experiment therefore does not justify a 1M expansion.

Full metrics, MRR, Top-3, calibration, parameters, model bytes, wall time, and VRAM are in `reports/first_sweep.md` and `reports/remote-real/`.

## Deployment

- Selected neural candidate: Tiny ~8M best checkpoint, based on highest neural Top-1 and net wins.
- FP16 safetensors: 14,865,276 bytes, 45 F16 tensors, SHA-256 `80da936a3e4616fbbb6172cbb37b208408101aab026cf484cb2f5187d288848a`.
- It meets the 16MB hard file cap but misses the 8MB target. Tiny ~4M is the precision-oriented alternative.
- Core ML conversion and trained-model macOS latency/RAM: **NOT YET MEASURED**. They were not allowed to distract from the requested real data and first-sweep path.
- Squirrel system integration remains incomplete; this is still a research prototype.

## Remaining risks

- The HF dataset is a third-party cleaned derivative; its card states Wikipedia licensing, but release of trained weights still needs a provenance review.
- The observed `tags` field is a string although the card describes an array. The adapter preserves it, and training does not use it.
- Public C API candidates omit C++ quality/type, so those features are null in this pilot.
- The 8M model was trained to repair raw Rime, not the stronger Wanxiang order; the two cannot yet be stacked safely.
- Wanxiang provides the best accuracy but its 400MB model is far outside TinyRime's product resource target.
- macOS Core ML numerical parity, latency, incremental RAM, idle CPU, deadline behavior, and real Squirrel stability remain unmeasured.

## Next single best step

Generate a Wanxiang-ranked 100k candidate dataset from the existing fixed queries and train a 2–4M conservative residual model to imitate only Wanxiang's correct promotions while abstaining on its 109 observed regressions; this directly tests whether TinyRime can approach the 400MB grammar baseline at a small deployment size.

## Remote shutdown

Continuation remote shutdown requested at: **2026-08-27 23:25:18 CST**. SSH closed immediately as expected. The instance was powered off, not destroyed, and was not contacted again; the verified corpus, query pool, candidate dataset, checkpoints, Wanxiang model, and source mirror remain on its data disk.
