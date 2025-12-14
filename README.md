# Polars FastEmbed

A high-performance Polars plugin for embedding DataFrames, implemented in Rust.

## Overview

This repository contains the Rust implementation of **polars-fastembed**, a Polars plugin for fast text embedding and retrieval directly inside Polars expressions.

## Performance Notes

- No batching or explicit parallelism is used in the Rust embedding code

### Embed

```bash
  Time (mean ± σ):     540.4 ms ±   2.8 ms    [User: 1738.9 ms, System: 155.0 ms]
  Range (min … max):   535.5 ms … 545.3 ms    10 runs
````

### Embed + Retrieve

```bash
  Time (mean ± σ):     845.7 ms ±  17.2 ms    [User: 4446.5 ms, System: 238.7 ms]
  Range (min … max):   817.0 ms … 865.8 ms    10 runs
```
