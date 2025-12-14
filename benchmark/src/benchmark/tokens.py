import polars as pl
from transformers import AutoTokenizer

from .dataset import load_peps
from .schema import TEXT_COL

# Load the pretrained MiniLM tokenizer (fast tokenizer by default)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def add_token_counts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a token_count column to a DataFrame of PEP texts.

    Args:
        df (pl.DataFrame): DataFrame with a column named TEXT_COL.

    Returns:
        pl.DataFrame: Same DataFrame with an added 'token_count' column.
    """
    # Use the fast tokenizer interface; encode returns a dict with "input_ids"
    token_counts = [
        len(tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in df[TEXT_COL].to_list()
    ]
    return df.with_columns(pl.Series("token_count", token_counts))


def load_peps_with_tokens() -> pl.DataFrame:
    """
    Load the PEP dataset and compute token counts for each PEP.

    Returns:
        pl.DataFrame: DataFrame with labels, texts, and token_count column.
    """
    df = load_peps()
    return add_token_counts(df)


def count_pep_tokens():
    """Load PEPs and print token counts"""
    df = load_peps_with_tokens()
    tok_counts = df.select([TEXT_COL, "token_count"])
    print(tok_counts)
    print(f"Total token count: {tok_counts.get_column('token_count').sum():,}")
