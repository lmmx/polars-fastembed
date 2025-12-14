# Polars FastEmbed

A high-performance Polars plugin for embedding DataFrames, implemented in Rust.

## Overview

This repository contains the Rust implementation of **polars-fastembed**, a Polars plugin for fast text embedding and retrieval directly inside Polars expressions.

## Performance Notes

- Benchmarking shows `Xenova/all-MiniLM-L6-v2` is the fastest, ~8ms per 1k tokens
- No batching or explicit parallelism is used in the Rust embedding code

### Embed

`hyperfine` timings from running the `embed_demo.py` script with `--warmup 10`

```bash
bash benchmark_embed.sh
```

```py
Benchmark 1: embed
  Time (mean ± σ):     591.9 ms ±  15.5 ms    [User: 1779.2 ms, System: 171.1 ms]
  Range (min … max):   573.2 ms … 615.1 ms    10 runs
```

### Embed + Retrieve

`hyperfine` timings from running the `demo.py` script with `--warmup 10`

```bash
bash benchmark_embed_and_retrieve.sh
```

```py
Benchmark 1: embed-and-retrieve
  Time (mean ± σ):     888.8 ms ±  29.6 ms    [User: 4508.7 ms, System: 253.6 ms]
  Range (min … max):   854.5 ms … 946.9 ms    10 runs
````

### Larger embedding

To embed all Python PEPs and retrieve a query, polars-fastembed takes about a minute

- Run with `time just bench`

```
uv run --frozen benchmark
Using model: Xenova/all-MiniLM-L6-v2
Embedded 708 documents.
Top 5 retrieval results:
shape: (5, 4)
┌─────┬────────────────────────┬─────────────────────────────────┬────────────┐
│ pep ┆ text                   ┆ embedding                       ┆ similarity │
│ --- ┆ ---                    ┆ ---                             ┆ ---        │
│ i64 ┆ str                    ┆ array[f32, 384]                 ┆ f32        │
╞═════╪════════════════════════╪═════════════════════════════════╪════════════╡
│ 589 ┆ PEP: 589               ┆ [-0.007806, 0.044277, … 0.0958… ┆ 0.520573   │
│     ┆ Title: TypedDict: Typ… ┆                                 ┆            │
│ 455 ┆ PEP: 455               ┆ [-0.112585, 0.064582, … 0.0891… ┆ 0.509375   │
│     ┆ Title: Adding a key-t… ┆                                 ┆            │
│ 705 ┆ PEP: 705               ┆ [-0.007666, -0.005967, … 0.118… ┆ 0.501325   │
│     ┆ Title: TypedDict: Rea… ┆                                 ┆            │
│ 764 ┆ PEP: 764               ┆ [-0.038508, 0.046109, … 0.0555… ┆ 0.491672   │
│     ┆ Title: Inline typed d… ┆                                 ┆            │
│ 814 ┆ PEP: 814               ┆ [-0.114546, 0.023868, … 0.0395… ┆ 0.486509   │
│     ┆ Title: Add frozendict… ┆                                 ┆            │
└─────┴────────────────────────┴─────────────────────────────────┴────────────┘
```

- Embedding all 708 Python PEPs and retrieval on query "Typed dictionaries and mappings" takes 30s
