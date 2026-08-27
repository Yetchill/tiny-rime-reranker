# Phase 2B final status

Phase 2B closes as the scoped TinyRime v0.1 research release. The authoritative numbers, overlap analysis, claim boundaries, resource status, and next research question are in [`v0.1-release-report.md`](v0.1-release-report.md).

The corrected full test set contains 22,080 Rime Top-8-recallable examples. Tiny-8M scores 86.621% conditional Top-1, below Wanxiang's 87.944%. Their oracle reaches 90.811%; the val-selected simple hybrid reaches 90.349%. Tiny-8M has 633 Wanxiang-complementary wins and 925 losses. The selected FP16 Tiny-8M weights are 14,865,276 bytes; trained-model latency and RSS remain unmeasured.

The chosen direction is a high-precision residual router from Wanxiang to Tiny-8M. The revised release scope explicitly defers residual training, multi-seed validation, data scaling, external benchmarks, Core ML optimization, and Squirrel product integration.
