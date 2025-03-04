from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from polars.api import register_dataframe_namespace
from polars.plugins import register_plugin_function

from polars_fastembed._polars_fastembed import clear_registry as _clear_registry
from polars_fastembed._polars_fastembed import list_models as _list_models
from polars_fastembed._polars_fastembed import register_model as _register_model

from .utils import parse_into_expr, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Determine the correct plugin path (like your `lib` variable).
if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

__all__ = ["embed_text"]


# --- Re-exported Rust functions so users can import from polars_fastembed directly ---


def register_model(model_name: str, providers: list[str] | None = None) -> None:
    """
    Register/load a model into the global registry by name or HF ID.
    If it's already loaded, this is a no-op.

    Note: providers is not implemented yet (CPU vs. GPU etc).
    """
    # _register_model(model_name, providers)
    _register_model(model_name)


def clear_registry() -> None:
    """Clear the entire global registry of loaded models."""
    _clear_registry()


def list_models() -> list[str]:
    """Return a list of currently loaded model IDs."""
    return _list_models()


# --- End of Rust internal re-exports ---


def plug(expr: IntoExpr, **kwargs) -> pl.Expr:
    """
    Wrap Polars' `register_plugin_function` helper to always
    pass the same `lib` (the directory where _polars_fastembed.so/pyd lives).
    """
    func_name = inspect.stack()[1].function
    into_expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=lib,
        function_name=func_name,
        args=into_expr,
        is_elementwise=True,
        kwargs=kwargs,
    )


def embed_text(expr: IntoExpr, *, model_id: str | None = None) -> pl.Expr:
    """
    Calls the Rust `embed_text` expression from `_polars_fastembed`.
    We pass `model_id` as a kwarg to the Rust side if it was set.
    """
    return plug(expr, **{"model_id": model_id})


# --- Plugin namespace ---


@register_dataframe_namespace("fastembed")
class FastEmbedPlugin:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def embed(
        self,
        columns: str | list[str],
        model_name: str,
        output_column: str = "embedding",
        join_columns: bool = True,
    ) -> pl.DataFrame:
        """
        Mirror the original: embed text from `columns` using `model_name`.
        If `model_name` not in the registry yet, it gets loaded automatically (or call register_model first).
        """
        if isinstance(columns, str):
            columns = [columns]

        # Optionally concat multiple columns
        if join_columns and len(columns) > 1:
            self._df = self._df.with_columns(
                pl.concat_str(columns, separator=" ").alias("_text_to_embed"),
            )
            text_col = "_text_to_embed"
        else:
            text_col = columns[0]

        # Now call the Rust expression
        new_df = self._df.with_columns(
            embed_text(text_col, model_id=model_name).alias(output_column),
        )

        if join_columns and len(columns) > 1:
            new_df = new_df.drop("_text_to_embed")
        return new_df

    def retrieve(
        self,
        query: str,
        model_name: str | None = None,
        embedding_column: str = "embedding",
        k: int | None = None,
        threshold: float | None = None,
        similarity_metric: str = "cosine",
        add_similarity_column: bool = True,
    ) -> pl.DataFrame:
        """
        Sort/filter rows by similarity to the given `query` using `model_name`.
        The embeddings for each row are read from `embedding_column`.

        This method:
         1) Embeds the query via the same Rust plugin (so uses the same model).
         2) For each row in `embedding_column`, calculates similarity to that query.
         3) Sorts (desc) by similarity.
            Optionally filters by `threshold`.
            Optionally keeps top-k rows.
            Optionally adds a "similarity" column.
        """
        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

        # 1) Embed the query in a single-row DF
        q_df = pl.DataFrame({"_q": [query]}).with_columns(
            embed_text("_q", model_id=model_name).alias("_q_emb"),
        )

        # Extract that single embedding as a numpy array
        # This handles both list and array dtype columns
        q_emb = q_df.select("_q_emb").item()

        if q_emb is None:
            raise ValueError("Failed to embed query (got null).")

        # Convert to numpy array if it's not already
        if isinstance(q_emb, list):
            q_emb_arr = np.array(q_emb, dtype=np.float32)
        else:
            # It's already an array
            q_emb_arr = q_emb

        # 2) For each row, compute similarity
        # Need to handle both list and array dtypes
        similarities = []
        q_norm = np.linalg.norm(q_emb_arr)

        # Get the column as a Python object (list or array)
        embs = self._df[embedding_column]

        for emb in embs:
            if emb is None:
                similarities.append(float("nan"))
                continue

            # Convert to numpy array if not already
            if isinstance(emb, list):
                e_arr = np.array(emb, dtype=np.float32)
            else:
                # It's already an array
                e_arr = emb

            if similarity_metric == "cosine":
                sim = float(np.dot(e_arr, q_emb_arr) / (np.linalg.norm(e_arr) * q_norm))
            elif similarity_metric == "dot":
                sim = float(np.dot(e_arr, q_emb_arr))
            else:
                raise ValueError(f"Unknown similarity metric: {similarity_metric}")
            similarities.append(sim)

        # 3) Create a new DF with similarity
        result_df = self._df
        if add_similarity_column:
            result_df = result_df.with_columns(pl.Series("similarity", similarities))

        # 4) Optionally threshold
        if threshold is not None:
            result_df = result_df.filter(pl.col("similarity") >= threshold)

        # 5) Sort desc by similarity
        result_df = result_df.sort("similarity", descending=True)

        # 6) Keep top-k
        if k is not None:
            result_df = result_df.head(k)

        return result_df
