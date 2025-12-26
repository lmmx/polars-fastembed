import polars as pl
from polars_fastembed import CUDA_AVAILABLE, embed_text, register_model

df = pl.DataFrame({"sentence": ["Hello world", None, "Polars plugin in Rust!"]})
model_id = "Xenova/all-MiniLM-L6-v2"
# For GPU, prefer the Snowflake Arctic Embed XS model (not quantised)
# model_id = "SnowflakeArcticEmbedXSQ"
register_model(model_id, cuda=False)

# Option 1: Auto-detect (default) - uses GPU if available, else CPU
df = df.with_columns(embed_text("sentence", model_id=model_id).alias("embedding"))

# Option 2: Force CPU only
# register_model(model_id, cuda=False)

# Option 3: Require GPU (fails if unavailable)
# register_model(model_id, cuda=True)

print(f"CUDA available: {CUDA_AVAILABLE}")
print(df)
