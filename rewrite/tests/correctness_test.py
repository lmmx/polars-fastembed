"""
Simple correctness test for polars-fastembed.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from polars_fastembed import embed_text, register_model


def test_embeddings_match_reference():
    """Test that polars-fastembed produces same embeddings as reference FastEmbed.

    Note: polars-fastembed uses the correct (HuggingFace) model IDs, fastembed itself
    uses a misleading one, e.g. BAAI as the username for this model from user Xenova.
    """

    # Load reference data
    fixture_path = Path(__file__).parent / "fixtures" / "reference_embeddings.json"
    with open(fixture_path) as f:
        data = json.load(f)

    documents = data["documents"]
    reference_embeddings = np.array(data["embeddings"])

    # Generate embeddings with polars-fastembed
    register_model("Xenova/bge-small-en-v1.5", providers=["CPUExecutionProvider"])
    df = pl.DataFrame({"text": documents})
    result_df = df.with_columns(embed_text("text").alias("embeddings"))
    your_embeddings = np.array(result_df["embeddings"].to_list())

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.mean(np.sum(a_norm * b_norm, axis=1))

    similarity = cosine_similarity(reference_embeddings, your_embeddings)

    print(f"Cosine similarity: {similarity:.8f}")
    assert similarity > 0.999999, f"Similarity too low: {similarity:.6f}"
