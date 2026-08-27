# TinyRime Context Reranker

TinyRime is a compact, local-only research reranker for the first eight Chinese pinyin candidates produced by Rime. It does not generate text: every accepted output is a permutation of candidates already supplied by the input engine. If the model abstains, times out, fails, or returns an invalid order, the original Rime order is preserved.

This repository is the **v0.1 research release**. It contains the frozen `TinyRime-Context-v1` protocol, evaluation and overlap tooling, aggregate results, a native runtime contract, and tests. It is not a complete macOS input method, and it does not claim to be the best open-source Chinese IME.

## Architecture

```text
Rime Top-8 candidates + left context + pinyin + candidate metadata
                              |
                    compact residual scorer
                              |
              confidence gate / optional abstention
                              |
             stable permutation of the same Top-8
```

The learned variants share deterministic character and pinyin features. Tiny-2M/4M/8M add a small Transformer encoder; MLP is the non-attention baseline. Candidate type is represented categorically in the corrected code path. The v0 checkpoints predate that fix and are evaluated with an explicit `legacy_zero` compatibility mode rather than silently changing their inputs.

## TinyRime-Context-v1

The benchmark uses `fjcanyue/wikipedia-zh-cn`, snapshot 2026-05-01, sampled across the complete 1,489,790-document JSONL with seed `20260827`. Documents are split before query windows are produced. The fixed query pool contains 250,000 train, 25,000 validation, and 25,000 test queries from disjoint document IDs.

The v1 protocol fixes the pilot's first-success truncation:

- candidate generation runs over each complete fixed query split;
- all recallable validation and test examples are retained (22,066 val; 22,080 test);
- 100,000 training examples are selected only after all 221,069 recallable train examples are generated, using a stable SHA-256 priority;
- `contested` is defined once on the full query pool and persisted;
- candidate misses are sampled and diagnosed independently per split;
- candidate type uses a categorical vocabulary; and
- the hash vocabulary is audited. The 8,499 observed character+pinyin tokens require 8,501 exact embedding entries; the 32,768-bucket hash still has 12.47% unique-token collisions, including 26 of the 100 most frequent tokens.

The immutable manifest with source revisions and content hashes is in [`reports/TinyRime-Context-v1/benchmark_manifest.json`](reports/TinyRime-Context-v1/benchmark_manifest.json).

## Results

Candidate recall uses every test query, including misses. Ranking metrics use all 22,080 test examples whose gold target occurs in the original Rime Top-8.

| Candidate generator | Recall@1 | Recall@3 | Recall@5 | Recall@8 | Recall@32 |
|---|---:|---:|---:|---:|---:|
| Rime + rime-ice | 73.704% | 85.068% | 87.396% | 88.320% | 88.868% |
| Rime + Wanxiang octagram | 78.148% | 86.396% | 88.112% | 88.808% | 89.232% |

| Ranker | Top-1 | Top-3 | MRR | Contested Top-1 |
|---|---:|---:|---:|---:|
| Rime | 83.451% | 96.318% | 0.90087 | 72.267% |
| MLP | 84.031% | 96.599% | 0.90461 | 73.411% |
| Tiny-2M | 85.951% | 96.952% | 0.91580 | 76.571% |
| Tiny-4M | 85.806% | 96.925% | 0.91493 | 76.355% |
| Tiny-8M | **86.621%** | **97.160%** | **0.92004** | **77.598%** |
| Wanxiang | **87.944%** | **97.251%** | **0.92738** | **80.083%** |

Tiny-8M is the strongest standalone neural model, but it does **not** beat Wanxiang: 633 test examples are fixed only by Tiny-8M, while 925 are fixed only by Wanxiang (net -292). Their oracle hybrid reaches 90.811% Top-1. A simple threshold selected on validation reaches 90.349% on test, with 697 wins, 166 losses, and +531 net wins over Wanxiang. This is exploratory hybrid evidence, not a final calibrated product rule.

The selected neural checkpoint has 7,430,338 parameters. Its verified FP16 safetensors export is 14,865,276 bytes (SHA-256 `80da936a3e4616fbbb6172cbb37b208408101aab026cf484cb2f5187d288848a`), versus the 420,250,668-byte external Wanxiang grammar. The checkpoint is kept locally for research but is not committed or included in the public release pending a separate downstream-weight provenance review.

Full results and claim boundaries are in [`reports/v0.1-release-report.md`](reports/v0.1-release-report.md). The resource comparison is in [`reports/pareto-frontier.md`](reports/pareto-frontier.md).

## Reproducing

Run the local test suite first:

```bash
python -m pytest
cmake -S plugin -B build/plugin
cmake --build build/plugin
ctest --test-dir build/plugin --output-on-failure
```

The complete corpus, ranking datasets, candidate caches, external Wanxiang model, and original checkpoints are intentionally absent from Git. On a prepared AutoDL data disk with the locked upstream dependencies and source assets described in [`DATA_LICENSES.md`](DATA_LICENSES.md), reproduce the frozen data path and evaluation with:

```bash
bash scripts/remote/build_context_v1.sh ddba2f778f008813514368226f55a0e7a695c48d
bash scripts/remote/evaluate_context_v1.sh
```

Before copying any result back to a Mac, create and inspect the explicit size/SHA-256 manifest. Do not transfer the corpus, complete datasets, caches, checkpoints, raw logs, or the full prediction artifact. The detailed protocol is in [`docs/benchmark-v1-protocol.md`](docs/benchmark-v1-protocol.md), and the disk rules are in [`docs/data-and-disk-policy.md`](docs/data-and-disk-policy.md).

## Safety invariants

- Top-K defaults to 8; reranking cannot introduce candidate text.
- Conservative gating defaults to abstention.
- Scoring is synchronous; a missed deadline discards the result for that composition.
- There is no network, telemetry, polling, background training, or user-history upload path.
- User dictionaries, candidate objects, comments, preedit, and engine learning remain owned by Rime.

## Limitations

- The benchmark is Wikipedia-derived and measures offline context reranking, not real user typing.
- Results come from one training seed and one 100k training subset; v0.1 does not establish multi-seed variance or data-scaling behavior.
- The candidate pool is Rime Top-8. Even the Rime/Wanxiang union reaches only 89.260% recall at 32 candidates on the full query pool.
- The current checkpoint used legacy candidate-type inputs; categorical-type and exact-vocabulary code paths have not yet received a controlled training ablation.
- Trained-model end-to-end latency and incremental RSS are not measured. The existing native microbenchmark is synthetic and must not be treated as product evidence.
- No external IME benchmark, Squirrel integration, Core ML product optimization, or public weight redistribution review is included.

## Next research question

The one highest-value follow-up is a **validation-calibrated residual router that promotes Tiny-8M only when Wanxiang is likely wrong**. The measured 2.867-point oracle gap and strong val-selected hybrid result make routing more promising than increasing model size or dataset scale. v0.1 deliberately stops before starting that experiment.

TinyRime source code is available under the BSD-3-Clause license. External software, data, and model artifacts retain their own licenses; see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) and [`DATA_LICENSES.md`](DATA_LICENSES.md).
