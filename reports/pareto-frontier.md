# v0.1 resource-aware comparison

This report separates measured accuracy and file size from unmeasured deployment properties. It does not label a full product Pareto frontier because trained-model latency and incremental RSS were not measured.

| Method | Parameters | Recorded model bytes | Conditional Top-1 | Contested Top-1 | Status |
|---|---:|---:|---:|---:|---|
| Rime | — | — | 83.451% | 72.267% | Candidate engine baseline |
| MLP | 278,274 | 1,113,824 FP32 | 84.031% | 73.411% | Smallest learned baseline |
| Tiny-2M | 1,646,562 | 6,588,248 FP32 | 85.951% | 76.571% | Best neural size/accuracy trade-off below 8MB FP32 |
| Tiny-4M | 3,447,970 | 13,796,496 FP32 | 85.806% | 76.355% | Dominated by Tiny-2M on measured accuracy and size |
| Tiny-8M | 7,430,338 | 29,725,976 FP32; 14,865,276 verified FP16 | 86.621% | 77.598% | Best standalone neural accuracy |
| Wanxiang | — | 420,250,668 grammar | **87.944%** | **80.083%** | Best standalone accuracy |
| Wanxiang → Tiny-8M simple hybrid | uses both | combined | **90.349%** | **83.766%** | Exploratory val-selected router |

Tiny-4M is empirically dominated by Tiny-2M in this release. Tiny-8M adds 0.670 Top-1 points over Tiny-2M at a substantially larger footprint. Wanxiang remains more accurate but its grammar file is about 28.27 times the selected Tiny-8M FP16 weights. The simple hybrid is the accuracy leader but requires both systems and therefore is not a compact standalone deployment.

For safe gain over Wanxiang, the val-selected 95%-precision operating point produces +101 net wins on test at 0.476% coverage and 98.095% realized promotion precision. The unconstrained simple hybrid produces +531 net wins at 4.375% coverage and 72.153% promotion precision.

Deployment latency and incremental RSS are **not measured** for these trained models. `reports/macos-runtime-microbenchmark.json` measures only a synthetic native 256×64 MLP (P95 0.0417ms), so it cannot be attached to any row above. No Core ML or end-to-end Squirrel result is part of v0.1.

The current evidence supports Tiny-2M for minimum learned footprint, Tiny-8M for standalone neural accuracy, Wanxiang for standalone benchmark accuracy, and residual routing as the next research direction. It does not yet identify a production Pareto winner.
