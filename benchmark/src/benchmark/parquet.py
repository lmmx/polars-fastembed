"""
Prepare PEPs as a parquet file for benchmarking.
"""

from pathlib import Path

from .dataset import load_peps

PARQUET_PATH = Path(__file__).parents[2] / "benchmark_data" / "peps.parquet"


def prepare_parquet():
    """Load PEPs and write to parquet file."""
    df = load_peps()
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PARQUET_PATH)
    print(f"Wrote {len(df)} PEPs to {PARQUET_PATH}")
