# Polars FastEmbed

A Polars plugin for embedding DataFrames

---

This repo contains 2 subdirectories:

- `original` - the Python version, working
- `rewrite` - the Rust rewrite, aiming to be faster than the original

## Benchmark

The Rust rewrite looks ~2x faster at a first impression:

```
hyperfine \
  './original/.venv/bin/python original/embed_demo.py' \
  './rewrite/.venv/bin/python rewrite/embed_demo.py' \
  -n python-plugin \
  -n rust-rewrite \
  --warmup 10
Benchmark 1: python-plugin
  Time (mean ± σ):      1.216 s ±  0.017 s    [User: 3.039 s, System: 0.169 s]
  Range (min … max):    1.179 s …  1.239 s    10 runs

Benchmark 2: rust-rewrite
  Time (mean ± σ):     560.5 ms ±   5.3 ms    [User: 1338.6 ms, System: 143.6 ms]
  Range (min … max):   552.6 ms … 572.8 ms    10 runs

Summary
  rust-rewrite ran
    2.17 ± 0.04 times faster than python-plugin
```
