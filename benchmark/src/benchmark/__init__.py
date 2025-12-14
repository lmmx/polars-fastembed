"""
Benchmark Polars FastEmbed using a local PEP corpus.

- Embedding only
- Embedding + retrieval
- Deterministic, no network
- Uses local PEP text files (clone https://github.com/python/peps.git)
"""

from pathlib import Path

import polars as pl
from polars_fastembed import register_model

MODEL_ID = "Xenova/bge-small-en-v1.5"
PEP_DIR = Path(__file__).parents[2] / "benchmark_data" / "peps"
LABEL_COL = "pep"
TEXT_COL = "text"
EMB_COL = "embedding"
QUERY = "Typed dictionaries and mappings"
TOP_K = 5


def load_peps() -> list[dict[str, str]]:
    """Load all PEP text files from the local directory."""
    docs = []
    for path in sorted(PEP_DIR.glob("pep-*.rst")):
        text = path.read_text().strip()
        if text:
            docs.append({LABEL_COL: path.stem.split("-")[1], TEXT_COL: text})
    return docs


def main():
    docs = load_peps()
    df = pl.DataFrame(docs).with_columns(pl.col(LABEL_COL).str.to_integer())

    register_model(MODEL_ID, providers=["CPUExecutionProvider"])

    df_emb = df.fastembed.embed(
        columns=TEXT_COL,
        model_name=MODEL_ID,
        output_column=EMB_COL,
    )
    print(f"Embedded {len(df_emb)} documents.")

    results = df_emb.fastembed.retrieve(
        query=QUERY,
        model_name=MODEL_ID,
        embedding_column=EMB_COL,
        k=TOP_K,
    )
    print(f"Top {TOP_K} retrieval results:")
    print(results)
