# TinyRime Context Reranker

TinyRime is a research prototype for conservatively reranking the first eight candidates produced by Rime. It never generates text: every output is a stable permutation of the candidates supplied by the engine. Inference is local-only and a backend failure, timeout, low-confidence decision, or invalid output preserves Rime's original order.

This repository is **not yet a usable macOS input method**. The headless baseline, dataset validation, offline models, and runtime contract are developed here before any system-level Squirrel integration is attempted. Performance targets in the design documents are goals until a report marks them as measured.

## Safety invariants

- Top-K defaults to 8 and outputs cannot introduce candidate text.
- Conservative gating defaults to abstention.
- Scoring is synchronous; a missed deadline discards the result for that composition.
- There is no network, telemetry, polling, background training, or user-history upload path.
- User dictionaries, candidate objects, comments, preedit, and engine learning remain owned by Rime.

## Quick checks

```bash
python -m pytest
cmake -S plugin -B build/plugin
cmake --build build/plugin
ctest --test-dir build/plugin --output-on-failure
```

Training is intentionally performed only on the designated remote GPU. See `docs/data-and-disk-policy.md` before transferring any remote output.
