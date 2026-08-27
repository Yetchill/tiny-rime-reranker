# Core ML backend boundary

`CoreMLBackend` will implement the same synchronous `Backend` contract after an experiment produces a deployment-worthy model. It must load once, make no network calls, and return failure on deadline or model errors. Core ML conversion and CPU/Neural Engine benchmarks are intentionally gated on real offline gains; no unmeasured latency or memory claim is made here.
