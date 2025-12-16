"""
Load the dataset of Python PEPs.
"""

from pathlib import Path

import polars as pl

from .schema import LABEL_COL, TEXT_COL

_DATA_DIR = Path(__file__).parents[2] / "benchmark_data"
PEP_DIR = _DATA_DIR / "peps"
PQ_PATH = _DATA_DIR / "peps.parquet"


def load_rst_files() -> list[dict[str, str]]:
    """Load all PEP text files from the local directory."""
    docs = []
    for path in sorted(PEP_DIR.glob("pep-*.rst")):
        text = path.read_text().strip()
        if text:
            docs.append({LABEL_COL: path.stem.split("-")[1], TEXT_COL: text})
    return docs


def load_peps() -> pl.DataFrame:
    """Prepare DataFrame with integer label and text of the PEP content.

    Loads from parquet if available, otherwise from rst files.
    """
    if PQ_PATH.exists():
        return pl.read_parquet(PQ_PATH)
    docs = load_rst_files()
    return pl.DataFrame(docs).with_columns(pl.col(LABEL_COL).str.to_integer())
