# ruff: noqa: E402
from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import polars_distance as pld
from polars.api import register_dataframe_namespace
from polars.plugins import register_plugin_function

from polars_fastembed._ort_loader import configure_ort

# Set ORT_DYLIB_PATH before importing the Rust extension
configure_ort()

# Now safe to import Rust module
from polars_fastembed._polars_fastembed import (
    clear_registry as _clear_registry,
)
from polars_fastembed._polars_fastembed import (
    list_models as _list_models,
)
from polars_fastembed._polars_fastembed import (
    register_model as _register_model,
)

from .utils import parse_into_expr, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Determine the correct plugin path (like your `lib` variable).
if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

__all__ = ["embed_text", "register_model", "clear_registry", "list_models"]


def register_model(model_name: str, providers: list[str] | None = None) -> None:
    """
    Register/load a model into the global registry by name or HF ID.
    If it's already loaded, this is a no-op.

    Note: providers is not implemented yet (CPU vs. GPU etc).
    """
    _register_model(model_name, providers)


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
        """
        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

        # 1) Embed the query and add to each row
        q_df = pl.DataFrame({"_q": [query]}).with_columns(
            embed_text("_q", model_id=model_name).alias("_q_emb"),
        )

        # Cross join to pair query with all rows
        result_df = self._df.join(q_df.select("_q_emb"), how="cross")

        if similarity_metric == "cosine":
            similarity_expr = 1 - pld.col(embedding_column).dist_arr.cosine("_q_emb")
        elif similarity_metric == "dot":
            similarity_expr = pl.col(embedding_column).dot(pl.col("_q_emb"))
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        if add_similarity_column:
            result_df = result_df.with_columns(similarity_expr.alias("similarity"))

        # Clean up temp column
        result_df = result_df.drop("_q_emb")

        # 3) Filter, sort, and limit
        if threshold is not None:
            result_df = result_df.filter(pl.col("similarity") >= threshold)

        result_df = result_df.sort("similarity", descending=True)

        if k is not None:
            result_df = result_df.head(k)

        return result_df
