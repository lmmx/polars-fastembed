# benchmark/src/benchmark/intrinsic_dim.py
"""Estimate intrinsic dimension of embedding spaces."""

import numpy as np
from polars_fastembed import register_model
from sklearn.neighbors import NearestNeighbors

from .dataset import load_peps
from .schema import EMB_COL, TEXT_COL

DEFAULT_MODEL_ID = "Xenova/all-MiniLM-L6-v2"
# DEFAULT_MODEL_ID = "SnowflakeArcticEmbedXS"


def levina_bickel_id(X: np.ndarray, k: int = 10) -> tuple[float, np.ndarray]:
    """
    Estimate intrinsic dimension using Levina-Bickel MLE.

    Args:
        X: (n_samples, n_features) array of embeddings
        k: number of neighbors (paper recommends k in 10-20 range)

    Returns:
        global_id: averaged intrinsic dimension estimate
        local_ids: per-point estimates
    """
    neigh = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    dists, _ = neigh.kneighbors(X)

    # Drop self-distance (column 0)
    dists = dists[:, 1:]

    T_k = dists[:, -1:]  # k-th neighbor distance
    T_j = dists[:, :-1]  # distances to neighbors 1..k-1

    log_ratios = np.log(T_k / (T_j + 1e-10))
    local_ids = (k - 2) / np.sum(log_ratios, axis=1)

    return np.mean(local_ids), local_ids


def embed_and_estimate(texts: list[str], label: str):
    """Embed texts and print ID estimates."""
    import polars as pl

    register_model(DEFAULT_MODEL_ID)

    df = pl.DataFrame({TEXT_COL: texts})
    emb_df = df.fastembed.embed(
        columns=TEXT_COL,
        model_name=DEFAULT_MODEL_ID,
        output_column=EMB_COL,
    )

    X = np.vstack(emb_df[EMB_COL].to_list())
    print(f"\n{label}: {X.shape[0]} docs × {X.shape[1]}D")

    for k in [10, 20]:
        id_est, local_ids = levina_bickel_id(X, k=k)
        print(f"  k={k:2d}: ID={id_est:5.1f} (std={np.std(local_ids):5.1f})")


def estimate_intrinsic_dim():
    """Estimate intrinsic dimension of PEP corpus embeddings."""
    df = load_peps()
    embed_and_estimate(df[TEXT_COL].to_list(), "PEPs")


def estimate_wikipedia_id():
    """Estimate intrinsic dimension of diverse Wikipedia articles."""
    from datasets import load_dataset

    # Grab more, skip the first chunk, sample sparsely
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    texts = []
    for i, x in enumerate(ds):
        if i >= 700000:
            break
        if i % 1000 == 0:  # Take every 1000th article
            texts.append(x["text"])

    embed_and_estimate(texts, "Wikipedia (700 sparse-sampled)")


def estimate_reviews_id():
    """Estimate ID on Amazon reviews via direct parquet load."""
    from datasets import load_dataset

    # Load parquet directly from HF hub
    url = "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/refs%2Fconvert%2Fparquet/en/train/0000.parquet"
    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    texts = [x["text"] for x in ds.take(100)]

    embed_and_estimate(texts, "Amazon reviews (100)")


def estimate_degenerate_id():
    """ID of nearly identical strings (should be ~1-2)."""
    import random

    base = "This is a product review. I bought this item and it works great."
    texts = [base + f" {random.randint(1, 100)}" for _ in range(700)]
    embed_and_estimate(texts, "Near-duplicates (700)")


# Add these to intrinsic_dim.py


def estimate_news_id():
    """News articles - broad topics, journalistic style."""
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="train", streaming=True)
    texts = [x["text"] for x in ds.take(700)]
    embed_and_estimate(texts, "AG News (700)")


def estimate_poems_id():
    """Poetry - very different register from technical docs."""
    from datasets import load_dataset

    ds = load_dataset("merve/poetry", split="train", streaming=True)
    texts = [x["content"] for x in ds.take(700)]
    embed_and_estimate(texts, "Poetry (700)")


def estimate_code_id():
    """Python code - narrow technical domain."""
    from datasets import load_dataset

    ds = load_dataset(
        "codeparrot/codeparrot-clean-train",
        split="train",
        streaming=True,
    )
    texts = [x["content"] for x in ds.take(700)]
    embed_and_estimate(texts, "Python code (700)")


def estimate_legal_id():
    """US Supreme Court opinions - real legal text."""
    from datasets import load_dataset

    ds = load_dataset("ChicagoHAI/CaseSumm", split="train", streaming=True)
    texts = [x["opinion"] for x in ds.take(700)]
    embed_and_estimate(texts, "US Supreme Court opinions (700)")
