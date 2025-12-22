# ruff: noqa: E402
from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
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
    extract_topics as _extract_topics,
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

# Determine the correct plugin path
if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

__all__ = [
    "embed_text",
    "register_model",
    "clear_registry",
    "list_models",
]


def register_model(model_name: str, providers: list[str] | None = None) -> None:
    """Register/load a model into the global registry by name or HF ID."""
    _register_model(model_name, providers)


def clear_registry() -> None:
    """Clear the entire global registry of loaded models."""
    _clear_registry()


def list_models() -> list[str]:
    """Return a list of currently loaded model IDs."""
    return _list_models()


def plug(expr: IntoExpr, **kwargs) -> pl.Expr:
    """Wrap Polars' register_plugin_function helper."""
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
    """Embed text using the specified model."""
    return plug(expr, model_id=model_id)


def s3_fit_transform(
    expr: IntoExpr,
    *,
    n_components: int = 10,
    model_id: str | None = None,
) -> pl.Expr:
    """
    Fit S³ (Semantic Signal Separation) and return document-topic weights.

    Note: This is a corpus-level operation - all documents are used to fit the model.
    """
    return plug(expr, n_components=n_components, model_id=model_id)


# =============================================================================
# DataFrame Namespace
# =============================================================================


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
        """Embed text from columns using the specified model."""
        if isinstance(columns, str):
            columns = [columns]

        if join_columns and len(columns) > 1:
            self._df = self._df.with_columns(
                pl.concat_str(columns, separator=" ").alias("_text_to_embed"),
            )
            text_col = "_text_to_embed"
        else:
            text_col = columns[0]

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
        """Sort/filter rows by similarity to the given query."""
        import polars_distance as pld

        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

        q_df = pl.DataFrame({"_q": [query]}).with_columns(
            embed_text("_q", model_id=model_name).alias("_q_emb"),
        )

        result_df = self._df.join(q_df.select("_q_emb"), how="cross")

        if similarity_metric == "cosine":
            similarity_expr = 1 - pld.col(embedding_column).dist_arr.cosine("_q_emb")
        elif similarity_metric == "dot":
            similarity_expr = pl.col(embedding_column).dot(pl.col("_q_emb"))
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        if add_similarity_column:
            result_df = result_df.with_columns(similarity_expr.alias("similarity"))

        result_df = result_df.drop("_q_emb")

        if threshold is not None:
            result_df = result_df.filter(pl.col("similarity") >= threshold)

        result_df = result_df.sort("similarity", descending=True)

        if k is not None:
            result_df = result_df.head(k)

        return result_df

    def s3_topics(
        self,
        text_column: str,
        n_components: int = 10,
        model_name: str | None = None,
        top_n: int = 10,
    ) -> pl.DataFrame:
        """
        Extract topics using S³ (Semantic Signal Separation).

        Returns the DataFrame with added columns:
        - topic_weights: List of weights for each topic
        - dominant_topic: Index of the highest-weight topic

        Args:
            text_column: Column containing text documents
            n_components: Number of topics to extract
            model_name: Embedding model ID (uses default if None)
            top_n: Number of top terms per topic (for topic descriptions)

        Returns:
            DataFrame with topic_weights and dominant_topic columns
        """
        return self._df.with_columns(
            s3_fit_transform(
                text_column,
                n_components=n_components,
                model_id=model_name,
            ).alias("topic_weights"),
        ).with_columns(
            pl.col("topic_weights")
            .list.eval(pl.element().abs().arg_max())
            .list.first()
            .alias("dominant_topic"),
        )

    def extract_topics(
        self,
        text_column: str,
        n_components: int = 10,
        model_name: str | None = None,
        top_n: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """
        Extract topic descriptions (top terms per topic).

        Args:
            text_column: Column containing text documents
            n_components: Number of topics to extract
            model_name: Embedding model ID (uses default if None)
            top_n: Number of top terms per topic

        Returns:
            List of topics, each topic is a list of (term, importance) tuples
        """
        documents = self._df[text_column].drop_nulls().to_list()
        return _extract_topics(documents, n_components, model_name, top_n)
