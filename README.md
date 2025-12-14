# Polars FastEmbed

A Polars plugin for embedding DataFrames.

---

This repo contains 2 subdirectories:

- `original` - the Python version, working
- `polars-fastembed` - the Rust polars-fastembed, 1.4x - 2.1x faster than the original

## Benchmark

- Note the benchmark can no longer be run as the original pure Python was deleted
- Note that this is without any batching/parallelism in the Rust embedding code!

#### Embed

```python
$ bash benchmark_both_embed.sh
Benchmark 1: python-plugin
  Time (mean ± σ):      1.165 s ±  0.018 s    [User: 2.945 s, System: 0.179 s]
  Range (min … max):    1.129 s …  1.198 s    10 runs

Benchmark 2: rust-polars-fastembed
  Time (mean ± σ):     540.4 ms ±   2.8 ms    [User: 1738.9 ms, System: 155.0 ms]
  Range (min … max):   535.5 ms … 545.3 ms    10 runs

Summary
  rust-polars-fastembed ran
    2.16 ± 0.04 times faster than python-plugin
```

#### Embed + retrieve

```python
$ bash benchmark_both_embed_and_retrieve.sh
Benchmark 1: python-plugin
  Time (mean ± σ):      1.159 s ±  0.007 s    [User: 3.029 s, System: 0.172 s]
  Range (min … max):    1.149 s …  1.170 s    10 runs

Benchmark 2: rust-polars-fastembed
  Time (mean ± σ):     845.7 ms ±  17.2 ms    [User: 4446.5 ms, System: 238.7 ms]
  Range (min … max):   817.0 ms … 865.8 ms    10 runs

Summary
  rust-polars-fastembed ran
    1.37 ± 0.03 times faster than python-plugin
```
