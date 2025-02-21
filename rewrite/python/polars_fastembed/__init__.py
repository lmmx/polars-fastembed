from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

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


def embed_text(expr: IntoExpr, *, model_id: str) -> pl.Expr:
    """
    Calls the Rust `embed_text` expression from `_polars_fastembed`.
    We pass `model_id` as a kwarg to the Rust side.
    """
    return plug(expr, model_id=model_id)
