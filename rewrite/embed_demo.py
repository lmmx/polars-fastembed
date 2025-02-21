import polars as pl
from polars_fastembed import embed_text

df = pl.DataFrame({"sentence": ["Hello world", None, "Polars plugin in Rust!"]})

# call embed_text
df = df.with_columns(embed_text("sentence").alias("embedding"))

print(df)
