# First sweep decision report

Status: **NOT RUN**. A valid 100k Wikipedia-derived, document-split candidate dataset was not available. AutoDL could build and train, but its outbound GitHub and Wikipedia connections failed; a Mac-side Wikipedia API fallback also failed. The saved 80/20 dataset is explicitly a mechanical smoke fixture whose labels are all Rime Top-1. It cannot answer accuracy, recall, scaling, or deployment-worthiness questions.

## Gate and smoke evidence

- Real librime + pinned rime-ice returned non-empty candidate lists for 100/100 deterministic requests.
- Candidate Recall@8 is **NOT YET MEASURED** because these requests do not contain independently sourced gold targets.
- The smoke set validated 80 train / 20 val records, no document overlap, and exists only on AutoDL.
- All five model families completed CUDA FP16 forward/backward, evaluation, and safetensors save/load smoke paths.

| Model | Valid Top-1 | Valid MRR | Contested | Wins | Losses | Net wins | Promotion precision | Parameters | FP32 smoke weight | Smoke train wall time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rime/rime-ice | NOT MEASURED | NOT MEASURED | NOT MEASURED | — | — | — | — | — | — | — | — |
| Wanxiang/octagram | NOT RUN | NOT RUN | NOT RUN | — | — | — | — | — | — | — | — |
| Linear | NOT VALID | NOT VALID | NONE | 0 | 0 | 0 | N/A | 65,622 | 262,880 B | 1.531 s | 18,938,368 B |
| MLP | NOT VALID | NOT VALID | NONE | 0 | 0 | 0 | N/A | 278,274 | 1,113,824 B | 1.666 s | 22,898,688 B |
| Tiny ~2M | NOT VALID | NOT VALID | NONE | — | — | — | N/A | 1,646,562 | 6,588,248 B | 1.669 s | 61,939,200 B |
| Tiny ~4M | NOT VALID | NOT VALID | NONE | — | — | — | N/A | 3,447,970 | 13,796,496 B | 1.658 s | 118,575,104 B |
| Tiny ~8M | NOT VALID | NOT VALID | NONE | — | — | — | N/A | 7,430,338 | 29,725,976 B | 2.007 s | 203,018,752 B |

The Tiny ~8M smoke checkpoint exported to verified FP16 safetensors at 14,865,276 bytes with SHA-256 `036f0ffa9f3c4739e8f7b0faa5e51dbd5fe2b6cdbd94b88fde91624024766eac`. It was not synchronized to the Mac because smoke weights are not deployment candidates.

## Required decisions

1. Rime Recall@8: **NOT YET MEASURED**.
2. rime-ice baseline: candidate generation is reproducible; accuracy **NOT YET MEASURED**.
3. Wanxiang baseline: **NOT RUN**; octagram plugin and pinned `.gram` release were not staged.
4. Linear improvement: **UNKNOWN**; smoke only.
5. MLP improvement: **UNKNOWN**; smoke only.
6. 2M/4M/8M scaling: **UNKNOWN**; parameter and execution paths only.
7. Best net wins: **UNKNOWN**.
8. Best promotion precision: **UNKNOWN**.
9. Is a Tiny Transformer worthwhile: **NO EVIDENCE YET**.
10. Current bottleneck: obtaining and processing a pinned public Wikipedia article-text source into real Rime Top-8 examples, then measuring candidate recall before training.
