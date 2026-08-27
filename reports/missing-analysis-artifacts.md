# Missing Phase 2A analysis artifacts

**PHASE 2A IMPLEMENTATION COMPLETE BUT SAMPLE-LEVEL ANALYSIS BLOCKED BY MISSING PREDICTION ARTIFACTS**

## Local evidence inventory

The Mac contains:

- aggregate candidate profiles for Rime and Wanxiang;
- per-model aggregate config, epoch summaries, and test metrics;
- the selected Tiny-8M FP16 checkpoint;
- no retained 10k test inputs/candidates;
- no per-example Rime, Wanxiang, Linear, MLP, Tiny-2M, Tiny-4M, or Tiny-8M final predictions;
- no per-example confidence, margin, residual score, or final score arrays.

The only JSONL fixture is the earlier 100-request Gate 1 candidate fixture. It has no gold labels or learned/Wanxiang predictions and cannot answer Phase 2A.

Wanxiang's retained-set scorer stored only one audit example per split on AutoDL; only its aggregate profile was synchronized. Aggregate counts cannot reconstruct example identities or joint correctness.

## What cannot be calculated

Exact Wanxiang/Tiny overlap, Tiny-only repairs, Oracle hybrids, simple hybrid gates, context/target/ambiguity groups, contested overlap, reliability bins, and qualitative error samples are all `NOT AVAILABLE` locally.

For Wanxiang versus Tiny-8M, aggregate accuracies only imply very loose Fréchet bounds on 10,000 test examples:

- both correct: 7,561–8,714;
- Wanxiang-only correct: 133–1,286;
- Tiny-only correct: 0–1,153;
- both wrong: 0–1,153;
- Oracle accuracy: 88.47%–100%;
- Oracle gain over Wanxiang: 0–11.53 percentage points.

Tiny-2M and Tiny-4M have the same 0–11.53-point oracle-gain upper bound because each is less accurate than Wanxiang and no joint outcomes were saved. These ranges are too broad to choose residual correction versus distillation. No point estimate may be inferred from them.

## Minimum required remote artifact

One full file is sufficient on AutoDL:

```text
reports/phase2a-predictions.jsonl.zst
```

It should contain exactly the retained validation and test splits: approximately 10,000 + 10,000 records, using schema version 1 in `docs/prediction-artifact-schema.md`. Required methods are Rime, Wanxiang, Linear, MLP, Tiny-2M, Tiny-4M, and Tiny-8M. Tiny predictions include confidence, margin, residual scores, and final scores. Wanxiang confidence/margin remain explicitly unavailable.

Estimated remote artifact size is 10–35MB compressed (roughly 40–80MB raw). The producer and analyzer are implemented as:

```bash
python -m benchmark.offline.export_prediction_artifact ...
python -m benchmark.offline.error_overlap \
  --predictions reports/phase2a-predictions.jsonl.zst \
  --output reports/error-overlap
```

The exact future export, analysis, and manifest commands are documented in `docs/prediction-artifact-schema.md`.

The full 20k prediction artifact should remain remote because it contains more than the Mac's 1,000-record full-text fixture allowance. The minimum synchronized evaluation files are `summary.json`, `summary.md`, two deterministic error samples capped at 200 records each, and their manifest/checksums—expected to be well below 5MB. Corpus, query pool, 100k/10k/10k candidate dataset, full predictions, Wanxiang model, checkpoints, and caches remain remote.

No AutoDL session was started during this phase.
