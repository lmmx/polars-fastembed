# Polars FastEmbed

<!-- [![downloads](https://static.pepy.tech/badge/polars-fastembed/month)](https://pepy.tech/project/polars-fastembed) -->
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![PyPI](https://img.shields.io/pypi/v/polars-fastembed.svg)](https://pypi.org/project/polars-fastembed)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/polars-fastembed.svg)](https://pypi.org/project/polars-fastembed)
[![License](https://img.shields.io/pypi/l/polars-fastembed.svg)](https://pypi.python.org/pypi/polars-fastembed)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/lmmx/polars-fastembed/master.svg)](https://results.pre-commit.ci/latest/github/lmmx/polars-fastembed/master)

A Polars plugin for embedding DataFrames

## Installation

```bash
pip install polars-fastembed
```

> The `polars` dependency is required but not included in the package by default.
> It is shipped as an optional extra which can be activated by passing it in square brackets:
> ```bash
> pip install polars-fastembed[polars]          # most users can install regular Polars
> pip install polars-fastembed[polars-lts-cpu]  # for backcompatibility with older CPUs
> ```

## Features

- Embed from a DataFrame by specifying the source column(s)
- Re-order/filter rows by semantic similarity to a query
- Efficiently reuse loaded models via a global registry (no repeated model loads)

## Demo

See [demo.py](https://github.com/lmmx/polars-fastembed/tree/master/rewrite/demo.py)

```py
import polars as pl
from polars_fastembed import register_model

# Create a sample DataFrame
df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "text": [
            "Hello world",
            "Deep Learning is amazing",
            "Polars and FastEmbed are well integrated",
        ],
    }
)

model_id = "Xenova/bge-small-en-v1.5"

# 1) Register a model
#    Optionally specify GPU: providers=["CUDAExecutionProvider"]
#    Or omit it for CPU usage
register_model(model_id, providers=["CPUExecutionProvider"])

# 2) Embed your text
df_emb = df.fastembed.embed(
    columns="text",
    model_name=model_id,
    output_column="embedding",
)

# Inspect embeddings
print(df_emb)

# 3) Perform retrieval
result = df_emb.fastembed.retrieve(
    query="Tell me about deep learning",
    model_name=model_id,
    embedding_column="embedding",
    k=3,
)
print(result)
```

```
shape: (3, 3)
┌─────┬─────────────────────────────────┬─────────────────────────────────┐
│ id  ┆ text                            ┆ embedding                       │
│ --- ┆ ---                             ┆ ---                             │
│ i64 ┆ str                             ┆ array[f32, 384]                 │
╞═════╪═════════════════════════════════╪═════════════════════════════════╡
│ 1   ┆ Hello world                     ┆ [0.015196, -0.022571, … 0.0260… │
│ 2   ┆ Deep Learning is amazing        ┆ [-0.016128, -0.018325, … -0.06… │
│ 3   ┆ Polars and FastEmbed are well … ┆ [-0.086584, 0.026477, … 0.0399… │
└─────┴─────────────────────────────────┴─────────────────────────────────┘
shape: (3, 4)
┌─────┬─────────────────────────────────┬─────────────────────────────────┬────────────┐
│ id  ┆ text                            ┆ embedding                       ┆ similarity │
│ --- ┆ ---                             ┆ ---                             ┆ ---        │
│ i64 ┆ str                             ┆ array[f32, 384]                 ┆ f64        │
╞═════╪═════════════════════════════════╪═════════════════════════════════╪════════════╡
│ 2   ┆ Deep Learning is amazing        ┆ [-0.016128, -0.018325, … -0.06… ┆ 0.825373   │
│ 3   ┆ Polars and FastEmbed are well … ┆ [-0.086584, 0.026477, … 0.0399… ┆ 0.543264   │
│ 1   ┆ Hello world                     ┆ [0.015196, -0.022571, … 0.0260… ┆ 0.52316    │
└─────┴─────────────────────────────────┴─────────────────────────────────┴────────────┘
```

Note:

- This will download a 133 MB model to your working directory under `.fastembed_cache`
- In the original version this was a 384-dimensional array of f64 and here it is a list of f32.
  This will become an array as well in future versions (watch this space).

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

MIT License
