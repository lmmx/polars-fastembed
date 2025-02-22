# Polars FastEmbed

A Polars plugin for embedding DataFrames.

---

This repo contains 2 subdirectories:

- `original` - the Python version, working
- `rewrite` - the Rust rewrite, aiming to be faster than the original

## Benchmark

### Initial rewrite

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

### After rewriting to feature parity

For a proper comparison I needed to include a retrieval method, for which I used numpy
(temporarily). This lost about half of the performance gain (+120% to just +60%).

- This is fixable by doing the retrieval in Rust too.
- Also note that this is without any batching/parallelism in the Rust embedding code!

#### Embed

```
$ bash benchmark_both_embed.sh
Benchmark 1: python-plugin
  Time (mean ± σ):      1.244 s ±  0.022 s    [User: 3.027 s, System: 0.176 s]
  Range (min … max):    1.211 s …  1.274 s    10 runs

Benchmark 2: rust-rewrite
  Time (mean ± σ):     799.7 ms ±   6.2 ms    [User: 2949.4 ms, System: 166.9 ms]
  Range (min … max):   788.5 ms … 808.5 ms    10 runs

Summary
  rust-rewrite ran
    1.56 ± 0.03 times faster than python-plugin
```

#### Embed + retrieve

```
$ bash benchmark_both_embed_and_retrieve.sh
Benchmark 1: python-plugin
  Time (mean ± σ):      1.262 s ±  0.039 s    [User: 3.068 s, System: 0.172 s]
  Range (min … max):    1.223 s …  1.341 s    10 runs

Benchmark 2: rust-rewrite
  Time (mean ± σ):     802.2 ms ±   7.8 ms    [User: 3415.4 ms, System: 170.7 ms]
  Range (min … max):   792.9 ms … 814.6 ms    10 runs

Summary
  rust-rewrite ran
    1.57 ± 0.05 times faster than python-plugin
```
