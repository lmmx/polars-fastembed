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
