"""
Benchmark Polars FastEmbed using a local PEP corpus.

- Embedding only
- Embedding + retrieval
- Deterministic, no network
- Uses local PEP text files (clone https://github.com/python/peps.git)
"""

import argparse

from polars_fastembed import register_model

from .dataset import load_peps
from .schema import EMB_COL, TEXT_COL

# DEFAULT_MODEL_ID = "SnowflakeArcticEmbedXSQ"
# DEFAULT_MODEL_ID = "Xenova/all-MiniLM-L6-v2"
DEFAULT_MODEL_ID = "SnowflakeArcticEmbedXS"
QUERY = "Typed dictionaries and mappings"
TOP_K = 5


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

    docs_df = load_peps()

    register_model(
        model_id,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        # providers=["CPUExecutionProvider"],
    )

    emb_df = docs_df.fastembed.embed(
        columns=TEXT_COL,
        model_name=model_id,
        output_column=EMB_COL,
    )
    print(f"Embedded {len(emb_df)} documents.")

    results = emb_df.fastembed.retrieve(
        query=QUERY,
        model_name=model_id,
        embedding_column=EMB_COL,
        k=TOP_K,
    )
    print(f"Top {TOP_K} retrieval results:")
    print(results)


if __name__ == "__main__":
    main()
