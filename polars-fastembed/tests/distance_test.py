import sys
import polars as pl
from inline_snapshot import snapshot
from polars_fastembed import register_model


def test_distances():
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

    result_dicts = result.drop("embedding").to_dicts()

    expected = snapshot(
        [
            {
                "id": 2,
                "text": "Deep Learning is amazing",
                "similarity": 0.825373113155365,
            },
            {
                "id": 3,
                "text": "Polars and FastEmbed are well integrated",
                "similarity": 0.5432637333869934,
            },
            {
                "id": 1,
                "text": "Hello world",
                "similarity": 0.5231598615646362,
            },
        ],
    )

    if sys.platform == "linux":
        assert result_dicts == expected
    else:
        for result_row, expected_row in zip(result_dicts, expected):
            assert result_row["id"] == expected_row["id"]
            assert result_row["text"] == expected_row["text"]
            assert round(result_row["similarity"], 5) == round(
                expected_row["similarity"],
                5,
            )
