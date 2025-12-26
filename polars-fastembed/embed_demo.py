import polars as pl
from polars_fastembed import embed_text

# Create DataFrame
df = pl.DataFrame({"sentence": ["Hello world", None, "Polars plugin in Rust!"]})

# Specify model
model_id = "Xenova/all-MiniLM-L6-v2"

# for GPU, prefer the Snowflake Arctic Embed XS model (not quantised)
# model_id = "SnowflakeArcticEmbedXS"

# 1) Optionally register the model ahead of time (CPU in this case)

# from polars_fastembed import register_model
# register_model(model_id, providers=["CPUExecutionProvider"])
#
# If using GPU it is required to register the model to set the execution providers
#
# register_model(
#     model_id, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
# )

# 2) Embed your text using the registered model
df = df.with_columns(embed_text("sentence", model_id=model_id).alias("embedding"))

print(df)
