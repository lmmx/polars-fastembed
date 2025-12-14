# Polars FastEmbed

A high-performance Polars plugin for embedding DataFrames, implemented in Rust.

## Overview

This repository contains the Rust implementation of **polars-fastembed**, a Polars plugin for fast text embedding and retrieval directly inside Polars expressions.

## Performance Notes

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
Embedded 708 documents.
Top 5 retrieval results:
shape: (5, 4)
┌─────┬────────────────────────┬─────────────────────────────────┬────────────┐
│ pep ┆ text                   ┆ embedding                       ┆ similarity │
│ --- ┆ ---                    ┆ ---                             ┆ ---        │
│ i64 ┆ str                    ┆ array[f32, 384]                 ┆ f32        │
╞═════╪════════════════════════╪═════════════════════════════════╪════════════╡
│ 593 ┆ PEP: 593               ┆ [-0.058784, 0.016706, … 0.0438… ┆ 0.779388   │
│     ┆ Title: Flexible funct… ┆                                 ┆            │
│ 677 ┆ PEP: 677               ┆ [-0.07416, -0.004228, … 0.0494… ┆ 0.774187   │
│     ┆ Title: Callable Type … ┆                                 ┆            │
│ 589 ┆ PEP: 589               ┆ [-0.100245, 0.044359, … 0.0525… ┆ 0.772979   │
│     ┆ Title: TypedDict: Typ… ┆                                 ┆            │
│ 603 ┆ PEP: 603               ┆ [-0.121382, -0.024193, … -0.00… ┆ 0.767088   │
│     ┆ Title: Adding a froze… ┆                                 ┆            │
│ 646 ┆ PEP: 646               ┆ [-0.063982, -0.012509, … 0.032… ┆ 0.76216    │
│     ┆ Title: Variadic Gener… ┆                                 ┆            │
└─────┴────────────────────────┴─────────────────────────────────┴────────────┘

real    0m55.144s
user    17m47.700s
sys     0m1.624s
```
