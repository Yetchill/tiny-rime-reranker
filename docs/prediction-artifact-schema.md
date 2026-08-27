# Phase 2A prediction artifact schema

The overlap analyzer consumes one JSON object per full recallable v1 validation/test example. A `.jsonl.zst` file is supported when `zstandard` is available; plain `.jsonl` is also valid. The artifact contains predictions and small candidate metadata, never corpus text beyond the already bounded 32-character context.

Required top-level fields:

```json
{
  "schema_version": 1,
  "example_id": "stable-24-hex-id",
  "split": "val",
  "source_document_id": "zhwiki:20260501:123",
  "context": "最多三十二个字符",
  "pinyin": ["shi", "shi"],
  "gold": "事实",
  "contested": true,
  "ambiguity_count": 10,
  "candidates": [
    {"text": "事实", "rime_rank": 0, "quality": null, "type": null},
    {"text": "实施", "rime_rank": 1, "quality": null, "type": null},
    {"text": "时势", "rime_rank": null, "quality": null, "type": null}
  ],
  "methods": {
    "rime": {"text": "事实"},
    "wanxiang": {"text": "实施"},
    "tiny_8m": {
      "text": "实施",
      "proposed_text": "实施",
      "changed": true,
      "confidence": 0.94,
      "margin": 0.31,
      "residual_scores": [0.1, 5.3],
      "final_scores": [0.02, 0.33]
    }
  }
}
```

The required methods are `rime`, `wanxiang`, `linear`, `mlp`, `tiny_2m`, `tiny_4m`, and `tiny_8m`. `candidates` is the union of original Rime Top-8 and Wanxiang Top-8; `rime_rank: null` identifies a Wanxiang-only candidate. Every final prediction must be a member of this union. `ambiguity_count` is the union size, and ambiguity bands are its empirical test-set terciles. Wanxiang margin/confidence is `NOT AVAILABLE` because the current public runner exposes neither.

`contested` is copied from the canonical fixed-query-pool label: normalized pinyin maps to at least two distinct gold targets anywhere in that frozen query pool. Validation records tune any simple-hybrid threshold; test records are never passed to the tuning function.

## Remote export command for a future session

This command performs inference/export only. It does not train, download data, or create another candidate dataset.

```bash
cd /root/autodl-tmp/tiny-rime-reranker
PYTHONPATH=source python -m benchmark.offline.export_prediction_artifact \
  --dataset-dir datasets/TinyRime-Context-v1/ranking-k8 \
  --output reports/TinyRime-Context-v1/phase2b-predictions.jsonl.zst \
  --checkpoint linear=experiments/real-first-sweep/linear/best.safetensors \
  --checkpoint mlp=experiments/real-first-sweep/mlp/best.safetensors \
  --checkpoint tiny_2m=experiments/real-first-sweep/tiny-2m/best.safetensors \
  --checkpoint tiny_4m=experiments/real-first-sweep/tiny-4m/best.safetensors \
  --checkpoint tiny_8m=experiments/real-first-sweep/tiny-8m/best.safetensors \
  --wanxiang-runner bin/tinyrime_rime_runner_octagram \
    --shared-data upstream/rime-ice \
    --user-data work/phase2a-export-user \
    --prebuilt-data work/rime-prebuilt-octagram \
    --schema tinyrime_ice_octagram \
    --top-k 8 --skip-maintenance --skip-deploy
```

Run the analyzer on AutoDL while the full 20k artifact is present:

```bash
PYTHONPATH=source python -m benchmark.offline.error_overlap \
  --predictions reports/TinyRime-Context-v1/phase2b-predictions.jsonl.zst \
  --output reports/TinyRime-Context-v1/error-overlap
```

Before AutoDL-to-Mac synchronization:

```bash
python source/scripts/remote/create_sync_manifest.py \
  /root/autodl-tmp/tiny-rime-reranker \
  reports/TinyRime-Context-v1/error-overlap \
  --output work/sync-manifest-phase2a.json
```

The exact v1 row count is determined by full recallable val+test rather than a 10k cap. Expect roughly 80–180MB raw or 20–60MB compressed, but the full artifact stays on AutoDL. Synchronize only `summary.json`, `summary.md`, the two deterministic error samples (at most 200 records each), and the manifest/checksums. This keeps full-text local fixtures below the existing 1,000-record Mac limit. No corpus, query pool, candidate dataset, model checkpoint, cache, or complete prediction artifact is required on the Mac.
