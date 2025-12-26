# Polars FastEmbed

A high-performance Polars plugin for embedding DataFrames, implemented in Rust.

## Overview

This repository contains the Rust implementation of **polars-fastembed**, a Polars plugin for fast text embedding and retrieval directly inside Polars expressions.

## Performance Notes

- Benchmarking shows `Xenova/all-MiniLM-L6-v2` is the fastest on CPU, ~8ms per 1k tokens.
  - This is the default model.
- With GPU, `SnowflakeArcticEmbedXS` can achieve ~5ms/1k tok (on both GPU and CPU executors)

### Embed

`hyperfine` timings from running the `embed_demo.py` script with `--warmup 10`

- Runs with the `polars_fastembed` venv, it should be synced with the `cpu` extra only (the `cuda`
  extra will be slightly slower to load)
- This toy example mainly measures the **minimal** time that we can embed in, not a realistic workload

```bash
bash benchmark_embed.sh
```

```py
Benchmark 1: embed
  Time (mean ± σ):     697.1 ms ±  11.6 ms    [User: 3330.8 ms, System: 1321.1 ms]
  Range (min … max):   679.1 ms … 718.7 ms    10 runs
```

### Embed + Retrieve

`hyperfine` timings from running the `demo.py` script with `--warmup 10`

```bash
bash benchmark_embed_and_retrieve.sh
```

```py
Benchmark 1: embed-and-retrieve
  Time (mean ± σ):     752.6 ms ±  12.5 ms    [User: 4648.9 ms, System: 1344.9 ms]
  Range (min … max):   731.5 ms … 773.5 ms    10 runs
````

### Larger embedding

To embed all 708 Python PEPs and retrieve a query "Typed dictionaries and mappings", polars-fastembed takes ~24s (~5s on GPU)

- Run with `time just bench`

```
uv run --frozen benchmark
Model 'SnowflakeArcticEmbedXS' loaded successfully
CUDA is_available: Ok(true)
CUDA provider registered successfully
Using model: SnowflakeArcticEmbedXS
Embedded 708 documents.
Top 5 retrieval results:
shape: (5, 4)
┌─────┬────────────────────────┬─────────────────────────────────┬────────────┐
│ pep ┆ text                   ┆ embedding                       ┆ similarity │
│ --- ┆ ---                    ┆ ---                             ┆ ---        │
│ i64 ┆ str                    ┆ array[f32, 384]                 ┆ f32        │
╞═════╪════════════════════════╪═════════════════════════════════╪════════════╡
│ 705 ┆ PEP: 705               ┆ [-0.008636, 0.071007, … 0.0072… ┆ 0.80542    │
│     ┆ Title: TypedDict: Rea… ┆                                 ┆            │
│ 589 ┆ PEP: 589               ┆ [-0.001627, 0.088968, … 0.0314… ┆ 0.800909   │
│     ┆ Title: TypedDict: Typ… ┆                                 ┆            │
│ 692 ┆ PEP: 692               ┆ [-0.024454, 0.049467, … 0.0257… ┆ 0.792025   │
│     ┆ Title: Using TypedDic… ┆                                 ┆            │
│ 649 ┆ PEP: 649               ┆ [0.014099, 0.06436, … -0.04776… ┆ 0.779885   │
│     ┆ Title: Deferred Evalu… ┆                                 ┆            │
│ 681 ┆ PEP: 681               ┆ [-0.026525, 0.074915, … 0.0083… ┆ 0.777705   │
│     ┆ Title: Data Class Tra… ┆                                 ┆            │
└─────┴────────────────────────┴─────────────────────────────────┴────────────┘

real    0m5.527s
```
