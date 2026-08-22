# Z-Image official English suite

This is an opt-in standalone protocol. It does not change
`ImageBenchmarkDataset`, the synchronous training evaluator, or their existing
resolution and sample defaults.

Inspect the 12 locked protocols and the 37,188-image complete contract:

```bash
benchmark-image-official protocols --json
```

`preflight` verifies pinned repository/dataset bytes. External checkouts are
passed as repeatable `--source NAME=PATH` arguments. CVTG and bilingual
Qwen-Image-Bench prompts are verified from the package assets.

`export-records` writes deterministic image-level JSONL. The
`official_report` profile preserves official multiplicity and resolution;
`training_monitor` is explicitly non-reportable and uses one image per prompt.
`--coverage-smoke` produces the stratified smoke set while retaining all four
samples for benchmarks whose official protocol requires them.

`--fallback-resolution N` changes only records labeled with a fallback square
resolution. Benchmark-owned policies such as HPSv3's aspect-aware 1024 sizing
and BizGenEval's dynamic-original sizing remain unchanged. The default is 1024
for backward compatibility.

After standalone generation:

```bash
benchmark-image-official collect-generation \
  --records records.jsonl --image-root run
benchmark-image-official materialize-layouts \
  --records records.jsonl --image-root run --layout-root run/layouts
```

Collection rejects missing/duplicate shards, protocol drift, missing images,
and incorrect dimensions. Layout materialization creates the official GenEval,
DPG-Bench, TIIF, OneIG, BizGenEval, T2I-CoReBench, and HPSv3 structures without
lossy conversion. DPG and OneIG 2x2 grids are produced from all four raw PNGs.

BizGenEval's official evaluator requires a closed Gemini API. Callers that
exclude closed APIs must mark it `DEFERRED_CLOSED_API`; they must not replace
its scorer or mix that partial execution with a complete 12-benchmark report.
