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

model_id = "Xenova/bge-small-en-v1.5"

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

# 3) Perform retrieval
result = df_emb.fastembed.retrieve(
    query="Tell me about deep learning",
    model_name=model_id,
    embedding_column="embedding",
    k=3,
)
print(result)
