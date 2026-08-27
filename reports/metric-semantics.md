# Metric semantics: 73.704% versus 84.050%

The two numbers are both correct, but they answer different questions and have different denominators.

## 1. Unconditional candidate accuracy/recall

`Rime Recall@1 = 73.704%` uses all 25,000 fixed test queries. `score_queries.py` increments `attempts` before checking whether the gold target appears in candidates, and increments each Recall@K only when the target rank is below K. A gold target below Top-8 therefore counts as wrong at every K.

Preferred names:

- `unconditional_candidate_recall_at_1`, equivalent here to end-to-end candidate Top-1 accuracy;
- `unconditional_candidate_recall_at_3/5/8`.

The same fixed pool gives Rime Recall@8 88.320%, establishing the maximum fraction that a reranker constrained to those Rime Top-8 lists could repair.

## 2. Conditional ranking accuracy

`Rime Top-1 = 84.050%` uses a separate 10,000-example ranking test. In `score_queries.py`, a query is not written when its gold target is absent from Rime Top-8. Successful examples are written until the split reaches its `test-limit=10000`. `RimeRankingDataset` then evaluates rank zero within those retained lists.

Preferred name:

- `conditional_ranking_accuracy_given_gold_in_rime_top8`.

It answers: among retained examples that Rime made rerankable, how often was the original Rime Top-1 already gold? It is expected to exceed unconditional Recall@1 because every Top-8 miss has already been removed.

## Exact data flow

```text
25,000 fixed test queries from 1,597 source documents
  ├─ score every query for unconditional Recall@1/3/5/8
  ├─ discard query from ranking dataset if gold is absent from Rime Top-8
  └─ retain the first 10,000 successful examples in deterministic query order
       → 10,000 ranking examples from 722 source documents
       → conditional Top-1/Top-3/MRR and neural wins/losses
```

The 25k pool itself is not the dump head. It comes from a fixed-seed 30,000-document reservoir over all 1,489,790 source rows. Documents are split by `source_document_id` before token generation. Within each document, up to 16 Jieba tokens are selected by fixed SHA-256 priority. Queries additionally require a 2–4 Han-character target, non-empty left context, valid pinyin conversion, and a globally unique context-target pair.

## Selection bias and scope

The 10k set is intentionally conditioned on Rime Top-8 recall, so it excludes OOV, decoder misses, and other hard candidate-generation failures. It must not be presented as end-to-end IME accuracy.

There is a second, smaller selection concern: the scorer retains the first 10,000 Top-8 successes in a deterministic, broadly randomized query stream; it does not reservoir-sample 10,000 uniformly from all 22,080 successful test queries. The upstream document reservoir and hashed per-document selection reduce dump-order bias, but do not prove the retained-success set is an unbiased sample of every Rime-recall success. All ranking methods share this same 10k denominator, so within-set comparisons are fair; extrapolation to the full query population needs per-query predictions.

It is invalid to estimate end-to-end neural accuracy by multiplying aggregate `Recall@8 × conditional Top-1`: the conditional Top-1 was measured on the retained first 10k successes, not on all 22,080 Top-8 successes.

## Contested semantics

The reported 1,068 contested test examples belong to the retained 10k ranking test. A pinyin key is contested only when it maps to at least two distinct gold targets within that retained split. It is not the contested subset of the unconditional 25k pool. Consequently, contested Wanxiang/Tiny figures in the first-sweep table are conditional ranking metrics too.

All future reports should label both the denominator and conditioning explicitly.
