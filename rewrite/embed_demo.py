import polars as pl
from polars_fastembed import embed_text  # , register_model

model_id = "Xenova/bge-small-en-v1.5"

# register_model("Xenova/bge-small-en-v1.5", providers=["CPUExecutionProvider"])

df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "text": [
            "Hello world",
            "Deep Learning is amazing",
            "Polars and FastEmbed are well integrated",
        ],
    },
)


# call embed_text
df = df.with_columns(embed_text("text", model_id=model_id).alias("embedding"))

print(df)
