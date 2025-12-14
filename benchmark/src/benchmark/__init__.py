"""
Benchmark Polars FastEmbed using a local PEP corpus.

- Embedding only
- Embedding + retrieval
- Deterministic, no network
- Uses local PEP text files (clone https://github.com/python/peps.git)
"""

import argparse
from pathlib import Path

import polars as pl
from polars_fastembed import register_model

DEFAULT_MODEL_ID = "Xenova/all-MiniLM-L6-v2"
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
    parser = argparse.ArgumentParser(description="Benchmark Polars FastEmbed")
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL_ID,
        help=f"Model ID to use (default: {DEFAULT_MODEL_ID})",
    )
    args = parser.parse_args()

    model_id = args.model
    print(f"Using model: {model_id}")

    docs = load_peps()
    df = pl.DataFrame(docs).with_columns(pl.col(LABEL_COL).str.to_integer())

    register_model(model_id, providers=["CPUExecutionProvider"])

    df_emb = df.fastembed.embed(
        columns=TEXT_COL,
        model_name=model_id,
        output_column=EMB_COL,
    )
    print(f"Embedded {len(df_emb)} documents.")

    results = df_emb.fastembed.retrieve(
        query=QUERY,
        model_name=model_id,
        embedding_column=EMB_COL,
        k=TOP_K,
    )
    print(f"Top {TOP_K} retrieval results:")
    print(results)


if __name__ == "__main__":
    main()
