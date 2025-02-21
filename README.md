# Polars SBERT

A Polars plugin for using embeddings on DataFrames

## Installation

Install the full Polars extra and CUDA 12.4-based PyTorch (`cu124`):
```bash
pip install "polars-sentence-transformers[polars,cu124]"
```

On older CPUs install `polars-lts-cpu` 

```bash
pip install polars-sentence-transformers[polars-lts-cpu]
```

and for CPU-only PyTorch use the `cpu-only` extra:
```bash
pip install polars-sentence-transformers[cpu-only]
```

## Problems

A quick profiling shows that this spends 90% of its time importing the sentence transformers
dependency, making it a serious burden to use. I'm going to investigate alternatives:

- [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
- [fastembed](https://github.com/qdrant/fastembed) //
  [fastembed-rs](https://github.com/Anush008/fastembed-rs)

## Features

- Embed from a DataFrame by specifying the source column(s)
- Re-order/filter rows by semantic similarity to a query
- Efficiently reuse loaded models via a global registry (no repeated model loads)

## Usage

TBD

## Demo

See [demo.py](https://github.com/lmmx/polars-sbert/tree/master/demo.py)

```
Corpus:
shape: (3, 2)
┌───────────────────────────┬───────────────────────────────┐
│ title                     ┆ body                          │
│ ---                       ┆ ---                           │
│ str                       ┆ str                           │
╞═══════════════════════════╪═══════════════════════════════╡
│ Living With Normal Pets   ┆ Pet owner's manual            │
│ My Cat Wants To Murder Me ┆ Scary stories about evil cats │
│ A Guide to Walking Dogs   ┆ Learning to lead              │
└───────────────────────────┴───────────────────────────────┘
Embedded title and body columns with TaylorAI/gte-tiny
Retrieve top k=None rows with query 'horror stories'
shape: (3, 4)
┌───────────────────────────┬────────────────────────────┬────────────────────────────┬────────────┐
│ title                     ┆ body                       ┆ embedding                  ┆ similarity │
│ ---                       ┆ ---                        ┆ ---                        ┆ ---        │
│ str                       ┆ str                        ┆ list[f32]                  ┆ f64        │
╞═══════════════════════════╪════════════════════════════╪════════════════════════════╪════════════╡
│ My Cat Wants To Murder Me ┆ Scary stories about evil   ┆ [-0.1195, -0.289812, …     ┆ 0.858065   │
│                           ┆ cats                       ┆ 0.14939…                   ┆            │
│ Living With Normal Pets   ┆ Pet owner's manual         ┆ [-0.168101, -0.158491, …   ┆ 0.754353   │
│                           ┆                            ┆ 0.488…                     ┆            │
│ A Guide to Walking Dogs   ┆ Learning to lead           ┆ [-0.388657, -0.37131, …    ┆ 0.727875   │
│                           ┆                            ┆ 0.6422…                    ┆            │
└───────────────────────────┴────────────────────────────┴────────────────────────────┴────────────┘
Currently loaded models: ['TaylorAI/gte-tiny']
Cleared model registry
Currently loaded models: []
```

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

MIT License
