# ruff: noqa: E402
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
    "S3Config",
]

DensityType = Literal["tanh", "exp", "cube"]


@dataclass
class S3Config:
    """Configuration for S³ (Semantic Signal Separation) topic modeling.

    This controls the Picard ICA algorithm used for blind source separation.

    Attributes:
        n_components: Number of topics to extract.
        max_iter: Maximum iterations for Picard optimization.
        tol: Convergence tolerance for gradient norm.
        density: Density function - "tanh" (default, super-Gaussian),
            "exp" (heavy tails), or "cube" (sub-Gaussian).
        density_alpha: Scaling parameter for tanh/exp density (default: 1.0).
        ortho: Use orthogonal constraint (Picard-O). Faster but more restrictive.
        extended: Use extended algorithm for mixed sub/super-Gaussian sources.
            Defaults to same as `ortho` if not specified.
        fastica_it: Number of FastICA warm-up iterations before Picard.
        jade_it: Number of JADE warm-up iterations before Picard.
            Cannot be used together with fastica_it.
        m: L-BFGS memory size.
        ls_tries: Maximum line search attempts.
        lambda_min: Minimum eigenvalue for Hessian regularization.
        random_state: Random seed for reproducibility.
        verbose: Print progress information.
    """

    n_components: int = 10
    max_iter: int = 200
    tol: float = 1e-4
    density: DensityType = "tanh"
    density_alpha: float | None = None
    ortho: bool = False
    extended: bool | None = None
    fastica_it: int | None = None
    jade_it: int | None = None
    m: int = 7
    ls_tries: int = 10
    lambda_min: float = 0.01
    random_state: int | None = None
    verbose: bool = False

    def to_kwargs(self) -> dict:
        """Convert to kwargs dict for Rust functions."""
        return {
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "density": self.density,
            "density_alpha": self.density_alpha,
            "ortho": self.ortho,
            "extended": self.extended,
            "fastica_it": self.fastica_it,
            "jade_it": self.jade_it,
            "m": self.m,
            "ls_tries": self.ls_tries,
            "lambda_min": self.lambda_min,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


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
    config: S3Config | None = None,
    **kwargs,
) -> pl.Expr:
    """
    Fit S³ (Semantic Signal Separation) on an embedding column and return document-topic weights.

    Args:
        expr: An embedding column (Array[f32, n])
        config: S3Config object with all parameters, or pass individual kwargs.
        **kwargs: Individual parameters (override config if both provided).
            See S3Config for available options.
    """
    if config is not None:
        merged = config.to_kwargs()
        merged.update(kwargs)
    else:
        # Apply defaults from S3Config
        defaults = S3Config().to_kwargs()
        defaults.update(kwargs)
        merged = defaults

    return plug(expr, **merged)


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
        embedding_column: str = "embedding",
        config: S3Config | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Extract topics using S³ (Semantic Signal Separation) from existing embeddings.

        Returns the DataFrame with added columns:
        - topic_weights: List of weights for each topic
        - dominant_topic: Index of the highest-weight topic

        Args:
            embedding_column: Column containing pre-computed embeddings (Array[f32, n])
            config: S3Config object with Picard ICA parameters.
            **kwargs: Individual parameters (see S3Config). Override config if both provided.
                Common options: n_components, max_iter, tol, density, fastica_it, random_state.

        Returns:
            DataFrame with topic_weights and dominant_topic columns
        """
        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

        return self._df.with_columns(
            s3_fit_transform(
                embedding_column,
                config=config,
                **kwargs,
            ).alias("topic_weights"),
        ).with_columns(
            pl.col("topic_weights")
            .list.eval(pl.element().abs().arg_max())
            .list.first()
            .alias("dominant_topic"),
        )

    def extract_topics(
        self,
        embedding_column: str = "embedding",
        text_column: str | None = None,
        model_name: str | None = None,
        top_n: int = 10,
        config: S3Config | None = None,
        **kwargs,
    ) -> list[list[tuple[str, float]]]:
        """
        Extract topic descriptions (top terms per topic) from existing embeddings.

        Args:
            embedding_column: Column containing pre-computed embeddings (Array[f32, n])
            text_column: Column containing text (for vocabulary extraction)
            model_name: Embedding model ID (needed to embed vocabulary words)
            top_n: Number of top terms per topic
            config: S3Config object with Picard ICA parameters.
            **kwargs: Individual parameters (see S3Config). Override config if both provided.

        Returns:
            List of topics, each topic is a list of (term, importance) tuples
        """
        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

        if text_column is None:
            raise ValueError("text_column is required for vocabulary extraction")

        if text_column not in self._df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        # Merge config with kwargs
        if config is not None:
            params = config.to_kwargs()
            params.update(kwargs)
        else:
            params = S3Config().to_kwargs()
            params.update(kwargs)

        # Get embeddings as list of lists
        embeddings = self._df[embedding_column].drop_nulls().to_list()
        texts = self._df[text_column].drop_nulls().to_list()

        return _extract_topics(
            embeddings=embeddings,
            texts=texts,
            n_components=params["n_components"],
            model_id=model_name,
            top_n=top_n,
            max_iter=params["max_iter"],
            tol=params["tol"],
            density=params["density"],
            density_alpha=params["density_alpha"],
            ortho=params["ortho"],
            extended=params["extended"],
            fastica_it=params["fastica_it"],
            jade_it=params["jade_it"],
            m=params["m"],
            ls_tries=params["ls_tries"],
            lambda_min=params["lambda_min"],
            random_state=params["random_state"],
            verbose=params["verbose"],
        )
