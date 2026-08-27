# First sweep decision report

Status: **COMPLETE on the real 100k/10k/10k pilot**.

The source is `fjcanyue/wikipedia-zh-cn` snapshot 2026-05-01. The full 2,393,692,848-byte JSONL stayed on AutoDL. A fixed-seed Algorithm R pass visited all 1,489,790 documents, selected 30,000 documents, then split by document ID before selecting 2–4 Han-character Jieba tokens. The fixed query pool contains 250k/25k/25k queries from 15,879/1,586/1,597 documents.

## Candidate recall on the fixed, unconditional query pool

These figures use all 25,000 test queries. A miss below Top-8 receives no training example.

| Baseline | Recall@1 | Recall@3 | Recall@5 | Recall@8 | 300k wall time | QPS |
|---|---:|---:|---:|---:|---:|---:|
| Rime + rime-ice | 73.704% | 85.068% | 87.396% | 88.320% | 39.98s | 7,505 |
| Rime + rime-ice + Wanxiang octagram | 78.148% | 86.396% | 88.112% | 88.808% | 55.78s | 5,378 |

Wanxiang therefore raises unconditional test Top-1 by 4.444 percentage points and Recall@8 by 0.488 points. The grammar baseline used the same context and queries, with `CommitHistory` injected before composition.

## Ranking results on the retained Rime Recall@8 test set

The model comparison uses the same 10,000 examples whose gold target is in the original Rime Top-8. Rime, Wanxiang, and every learned model below therefore have the same denominator. `Top-3` and `MRR` reflect the conservative gated output; abstention preserves the complete Rime order.

| Model | Top-1 | Top-3 | MRR | Contested Top-1 | Contested MRR | Wins | Losses | Net wins | Coverage | Promotion precision | Params | FP32 bytes | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rime + rime-ice | 84.050% | 96.380% | 0.90452 | 57.116% | 0.73892 | 0 | 0 | 0 | 0% | N/A | — | — | — |
| Wanxiang octagram | 88.470% | 97.300% | 0.93040 | 69.476% | 0.80924 | 551 | 109 | **442** | 7.230% | 76.210% | — | 420,250,668 | — |
| Linear | 84.050% | 96.380% | 0.90452 | 57.116% | 0.73892 | 0 | 0 | 0 | 0% | 0% | 65,622 | 262,880 | 39.11s |
| MLP | 84.660% | 96.710% | 0.90855 | 57.116% | 0.73892 | 61 | 0 | 61 | 0.680% | 89.706% | 278,274 | 1,113,824 | 37.82s |
| Tiny ~2M | 86.570% | 97.100% | 0.91968 | 59.925% | 0.75601 | 257 | 5 | 252 | 2.850% | 90.175% | 1,646,562 | 6,588,248 | 38.64s |
| Tiny ~4M | 86.450% | 97.070% | 0.91908 | 59.176% | 0.75111 | 243 | 3 | 240 | 2.610% | **93.103%** | 3,447,970 | 13,796,496 | 41.54s |
| Tiny ~8M | **87.140%** | **97.280%** | **0.92329** | **60.674%** | **0.76273** | 316 | 7 | **309** | 3.530% | 89.518% | 7,430,338 | 29,725,976 | 40.95s |

Contested metrics use 1,068 test examples whose pinyin maps to at least two different gold targets within the split. Corrected gate ECE is 0.02465 / 0.00829 / 0.03331 / 0.02003 / 0.00979 for Linear, MLP, 2M, 4M, and 8M respectively.

The selected neural checkpoint is Tiny ~8M because it has the highest neural net wins and Top-1. Verified FP16 export is 14,865,276 bytes, SHA-256 `80da936a3e4616fbbb6172cbb37b208408101aab026cf484cb2f5187d288848a`. It meets the 16MB hard cap but misses the 8MB target. Tiny ~4M is the conservative alternative because it has the best promotion precision and only three losses.

## Decisions

1. Rime Recall@8: **88.320% test** on the unconditional fixed query pool.
2. rime-ice baseline: **73.704% unconditional Top-1**; **84.050%** after conditioning on original Top-8 recall.
3. Wanxiang baseline: **78.148% unconditional Top-1** and **88.470%** on the retained test set; it beats every neural model here.
4. Linear improvement: none at the conservative threshold; it abstains everywhere.
5. MLP improvement: +61 net wins with 89.706% promotion precision.
6. Scaling: 2M materially beats MLP; 4M does not beat 2M; 8M gains 0.69 Top-1 points and +69 net wins over 4M. Scaling is useful but non-monotonic.
7. Best neural net wins: **309 (Tiny ~8M)**. Best overall baseline net wins: **442 (Wanxiang)**.
8. Best neural promotion precision: **93.103% (Tiny ~4M)**. Tiny ~8M is 89.518%; Wanxiang is 76.210%.
9. Is a Tiny Transformer worthwhile: it clearly beats MLP and raw Rime, but **not yet as a standalone replacement for Wanxiang**. The next neural experiment should learn residual corrections on top of Wanxiang, not the weaker raw Rime order.
10. Current bottleneck: the strong traditional grammar baseline already resolves more cases; the neural value proposition must become higher-precision corrections to Wanxiang while retaining a much smaller deployment footprint than its 400MB grammar model.

No 1M expansion is authorized from this result: 4M scaling is flat, and Wanxiang remains stronger than 8M on the same test set.
