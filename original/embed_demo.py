import polars as pl

from polars_fastembed import register_model

# Create a sample DataFrame
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

model_id = "BAAI/bge-small-en"

# 1) Register a model
#    Optionally specify GPU: providers=["CUDAExecutionProvider"]
#    Or omit it for CPU usage
register_model(model_id, providers=["CPUExecutionProvider"])

# 2) Embed your text
df_emb = df.fastembed.embed(
    columns="text",
    model_name=model_id,
    output_column="embedding",
)

# Inspect embeddings
print(df_emb)
