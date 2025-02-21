from __future__ import annotations

import numpy as np
import polars as pl
from fastembed import TextEmbedding
from polars.api import register_dataframe_namespace

# Global dictionary: model_name -> loaded FastEmbed TextEmbedding instance
_FASTEMBED_MODEL_REGISTRY = {}


def register_model(model_name: str, providers: list[str] | None = None) -> None:
    """
    Register/load a FastEmbed TextEmbedding model in the global registry by its name.
    If already present, does nothing.

    Args:
        model_name: A Hugging Face model ID or local path supported by FastEmbed,
                    for example "BAAI/bge-small-en-v1.5" or "intfloat/multilingual-e5-base".
        providers:  Optional list of providers, e.g. ["CPUExecutionProvider"] or ["CUDAExecutionProvider"].
                    If None, defaults to CPU usage.
    """
    if model_name not in _FASTEMBED_MODEL_REGISTRY:
        # This triggers the model download and initialization
        _FASTEMBED_MODEL_REGISTRY[model_name] = TextEmbedding(
            model_name=model_name,
            providers=providers if providers else None,
        )


@register_dataframe_namespace("fastembed")
class FastEmbedPlugin:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def clear_registry(self) -> None:
        """
        Clear the entire global registry of models.
        This frees memory but also removes any loaded models.
        """
        _FASTEMBED_MODEL_REGISTRY.clear()

    def list_models(self) -> list:
        """
        Return a list of model names currently in the global registry.
        """
        return list(_FASTEMBED_MODEL_REGISTRY.keys())

    def embed(
        self,
        columns: str | list[str],
        model_name: str,
        output_column: str = "embedding",
        join_columns: bool = True,
    ) -> pl.DataFrame:
        """
        Embed text from `columns` using the model named `model_name` in the registry.
        - If model_name is not yet registered, automatically loads it.
        - If `join_columns` is True, concatenate the specified columns per row (with space).
        - Appends a new column `output_column` with the embedding (a list[float]) per row.
        Returns a *new* DataFrame with that column appended.
        """
        # Ensure the model is registered
        if model_name not in _FASTEMBED_MODEL_REGISTRY:
            self.register_model(model_name)  # Register it with default CPU provider
        local_model = _FASTEMBED_MODEL_REGISTRY[model_name]

        if isinstance(columns, str):
            columns = [columns]

        # Gather text for each row
        if join_columns:
            # Concatenate specified columns by space
            df_concat = self._df.select(
                pl.concat_str(columns, separator=" ").alias("_text_to_encode")
            )
            texts = df_concat["_text_to_encode"].to_list()
        else:
            # If not joining, just embed the first column for simplicity
            texts = self._df[columns[0]].to_list()

        # FastEmbed returns a generator of numpy arrays
        # We'll convert each array to a Python list of floats
        embeddings = []
        for emb in local_model.embed(texts):
            # emb is a numpy.ndarray (e.g. shape [384] for bge-small)
            embeddings.append(emb.tolist())

        emb_size = len(emb)
        array_dtype = pl.Array(pl.Float64, width=emb_size)

        # Convert embeddings into a Polars Series of type Array[emb_size]
        embedding_series = pl.Series(output_column, embeddings, dtype=array_dtype)
        return self._df.with_columns(embedding_series)

    def retrieve(
        self,
        query: str,
        model_name: str,
        embedding_column: str = "embedding",
        k: int | None = None,
        threshold: float | None = None,
        similarity_metric: str = "cosine",
        add_similarity_column: bool = True,
    ) -> pl.DataFrame:
        """
        Sort/filter rows by similarity to the given `query` using `model_name`.

        - If `model_name` not in registry, automatically loads it.
        - Embeddings must already be in `embedding_column`.
        - If `k` is provided, keep only top-k rows (sorted desc by similarity).
        - If `threshold` is provided, discard rows below this similarity.
        - `similarity_metric` can be "cosine" or "dot".
        - If `add_similarity_column` is True, adds a 'similarity' column in the result.

        Returns a new DataFrame sorted by descending similarity.
        """
        if model_name not in _FASTEMBED_MODEL_REGISTRY:
            self.register_model(model_name)
        local_model = _FASTEMBED_MODEL_REGISTRY[model_name]

        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

        # Embed the query
        query_emb_gen = local_model.embed([query])  # returns a generator
        query_emb = next(query_emb_gen).astype(np.float32)  # shape [dim]

        # Extract row embeddings
        row_embs = self._df[embedding_column].to_list()

        # Compute local similarities
        query_norm = np.linalg.norm(query_emb)
        similarities = []
        if similarity_metric == "cosine":
            for emb_list in row_embs:
                e_arr = np.array(emb_list, dtype=np.float32)
                sim = float(
                    np.dot(e_arr, query_emb) / (np.linalg.norm(e_arr) * query_norm)
                )
                similarities.append(sim)
        elif similarity_metric == "dot":
            for emb_list in row_embs:
                e_arr = np.array(emb_list, dtype=np.float32)
                sim = float(np.dot(e_arr, query_emb))
                similarities.append(sim)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        # Append similarity column (optional)
        result_df = self._df
        if add_similarity_column:
            sim_series = pl.Series("similarity", similarities)
            result_df = result_df.with_columns(sim_series)

        # Filter by threshold if provided
        if threshold is not None:
            result_df = result_df.filter(pl.col("similarity") >= threshold)

        # Sort descending by similarity
        result_df = result_df.sort("similarity", descending=True)

        # Keep top-k
        if k is not None:
            result_df = result_df.head(k)

        return result_df
