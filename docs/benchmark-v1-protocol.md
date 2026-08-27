# TinyRime-Context-v1 protocol

`pilot-v0` remains historical evidence and is not overwritten. `TinyRime-Context-v1` fixes its retained-success ordering bias and freezes a separate manifest.

## Data flow

```text
full 2026-05-01 source JSONL
  → Algorithm R, seed 20260827, 30,000 documents
  → document-ID split
  → fixed 250k/25k/25k query pool
  → canonical contested labels over the complete fixed query pool
  → score every query to Top-32
  → unconditional Recall@1/3/5/8/12/16/24/32
  → derive Top-8 ranking sets
       train: stable-hash sample 100k from every recallable train example
       val: every recallable val example
       test: every recallable test example
```

No first-K retained-success limit exists in v1 val/test. Candidate scoring and train sampling are separate commands. The train sample is invariant to scored-query file order.

## Canonical contested label

Pinyin is normalized to lowercase syllables joined by apostrophe. A key is contested when the complete fixed 300k query pool contains at least two distinct gold targets for it. The boolean is persisted in each query, scored query, ranking example, prediction artifact, and analysis output. Dataset loaders never derive it from the current subset.

## Candidate and miss protocol

Both Rime and Wanxiang are scored once to Top-32. Recall at smaller K is derived from the saved target rank. Miss diagnostics are sampled independently per split using stable SHA-256 priority. Categories distinguish decoder-empty, verified rank below display Top-8, and absence from Top-32. Absence from Top-32 is not falsely labeled OOV without a dictionary lookup. Proper-name status is a deterministic `target in source title` proxy.

The headless runner uses librime's internal candidate objects when available, preserving quality and type. The C API null-metadata path remains only a fallback.

## Feature compatibility

- Historical model presets use `legacy_zero` candidate type because pilot-v0 stored every type as null.
- Corrected scalar ablations hash real type to a scalar and are named `*-scalar-type-hash`.
- v1 categorical ablations use the fixed candidate-type vocabulary and an embedding.
- Hash and exact-vocabulary categorical presets retain the same embedding capacity and architecture, so their parameter budgets match.
- Exact vocabulary is built from train only with PAD=0 and UNK=1. Character and pinyin namespaces have disjoint IDs.

## Frozen evidence

`benchmark_manifest.json` records source revision/SHA, reservoir seed and selected-document hash, query code commit, normalization and pinyin rules, query/scored/ranking split content hashes, and all Rime/Wanxiang version hashes. The freeze command refuses to overwrite an existing manifest. Protocol changes require a separately named v2.

## Evaluation

- Unconditional metrics use every fixed query.
- Conditional Top-1/Top-3/MRR use every gold-in-pool ranking example.
- Gate thresholds and Safe Gain operating points are selected on full val only and applied to full test once.
- Full predictions remain on AutoDL. Mac receives aggregate results and at most 400 deterministic qualitative examples.
