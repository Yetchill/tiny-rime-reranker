# Phase 2A decision

## Decision

```text
NEXT RECOMMENDATION: STOP
```

`STOP` means stop choosing or launching A/B/C until the already-completed models export one minimal val+test prediction artifact. It does not mean abandon TinyRime.

Exact complementarity is not identifiable from local aggregate metrics. Wanxiang is correct on 8,847/10,000 and Tiny-8M on 8,714/10,000, but Tiny-only correctness can legally be anywhere from 0 to 1,153 examples. Therefore neither a Wanxiang residual corrector nor distillation can be selected honestly yet. Data scaling is already unsupported by the aggregate evidence.

## What aggregate evidence does establish

### Tiny-4M promotion precision

Tiny-4M changes 261/10,000 Top-1 decisions versus 353 for Tiny-8M. It records 243 wins and 3 losses, while 8M records 316 wins and 7 losses. Its 93.103% promotion precision is therefore primarily the arithmetic consequence of lower coverage and fewer observed regressions, not evidence of better calibration.

Corrected gate ECE is 0.02003 for 4M and 0.00979 for 8M, so 8M is better on the saved aggregate calibration metric. On validation at epoch 4, promotion precision was 86.99% for 4M and 86.84% for 8M—nearly identical. The larger 93.10% versus 89.52% test gap could be split noise under one seed. Without confidence bins and example identities, claims that 4M has an intrinsically superior gate or that 8M systematically overrules a particular class of correct Top-1 candidates are not justified.

Product-first interpretation: 4M is the conservative candidate because it is smaller, reorders 26% fewer cases than 8M, and has three rather than seven test losses. It cannot be declared the better product model until repeated seeds or sample-level reliability analysis confirms the difference.

### Scaling

All models use seed 20260827, batch 512, learning rate 0.002, four epochs, and the same 100k/10k/10k data. Parameter counts are 1.65M, 3.45M, and 7.43M. Tiny-2M selects epoch 3 by validation net wins; 4M and 8M select epoch 4. Training loss continues downward for all three. Validation net wins peak at 254 / 223 / 326, while test Top-1 is 86.57% / 86.45% / 87.14%.

There is no saved validation loss, no repeated seed, and no run variance. The 2M→4M regression and 4M→8M gain therefore cannot be interpreted as a reliable scaling law. The first sweep does not authorize 300k/1M data scaling.

### Direction triage

- Direction A, Wanxiang residual correction: potentially attractive only if Tiny has a material Tiny-only-correct quadrant and a val-tuned gate realizes some of that oracle gain at high precision.
- Direction B, Wanxiang→Tiny distillation: favored if Tiny correctness is mostly a subset of Wanxiang, because Wanxiang is stronger but its 420MB model violates the product budget.
- Direction C, data scaling: currently last. Non-monotonic single-seed scaling and a stronger traditional baseline give no evidence that more of the same Rime-conditioned data is the immediate bottleneck.

The missing overlap result is exactly what separates A from B.

## Immediate hypothesis and work size

The next action validates one hypothesis only: **does Tiny-8M correctly cover a material fraction of Wanxiang's 1,153 retained-test errors?**

- AutoDL: a brief future restart is required only to rerun inference/export; no training or download.
- Training data required now: zero.
- Prediction rows: existing 10k val + 10k test.
- Model size to train now: none; compare the five existing checkpoints.
- Expected remote runtime: inference/export only, likely minutes given the recorded scorer/training throughput.
- Expected Mac transfer: under 5MB for summary plus at most 400 full error samples; the 10–35MB full prediction artifact remains remote.

## Success/stop criteria after the artifact exists

Let `G_oracle` be exact Oracle(Wanxiang, Tiny-8M) gain over Wanxiang on the 10k test set.

1. If `G_oracle <= 0.3 percentage points` (30 examples), stop Direction A as too limited and prefer Direction B.
2. Direction A is worth one 100k, 2–4M experiment only if a threshold selected exclusively on val achieves on test:
   - promotion precision at least 95%;
   - positive net wins with at most 10 losses per 10k;
   - realized gain at least `max(0.1 percentage points, 25% of G_oracle)`.
3. Direction A fails if the oracle is material but the simple val-tuned gate captures less than 25% of it, promotion precision is below 95%, or losses exceed 10.
4. Direction B becomes the single recommendation if Tiny-only correctness is negligible or gating fails, because the target then is compression of Wanxiang behavior rather than complementing it.
5. Direction C remains stopped until repeated-seed learning curves—not this single sweep—show a stable capacity/data bottleneck.

No Phase 2A oracle or hybrid number is reported now. Producing one from the aggregate table would be fabrication.

## Verification

- 11 standard-library unit/CLI tests passed: overlap quadrants, oracle, win/loss, grouping with empty subsets, deterministic sampling, stable IDs, missing fields, missing confidence, val-only threshold tuning, test application, and analyzer file output.
- The existing C++ runtime test compiled with Apple clang and passed.
- Python modules pass bytecode compilation and `git diff --check`.
- This Mac has no `pytest`, `cmake`, or `ctest` executable. The phase did not install dependencies under the hard limits; the new tests are `unittest.TestCase` tests and remain discoverable by pytest when the normal project test environment is available.
