"""Half-document retrieval benchmark on the PEP corpus."""

import polars as pl
from polars_fastembed import register_model

from benchmark.dataset import load_peps
from benchmark.schema import EMB_COL, TEXT_COL

DEFAULT_MODEL_ID = "Xenova/all-MiniLM-L6-v2"


def split_half(text: str):
    """Split a document roughly in half by words."""
    words = text.split()
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def main():
    # Load PEP corpus
    docs_df = load_peps()
    print(f"Loaded {len(docs_df)} PEPs")

    # Split into first and second halves
    first_halves = []
    second_halves = []
    for text in docs_df[TEXT_COL]:
        first, second = split_half(text)
        first_halves.append(first)
        second_halves.append(second)

    df = pl.DataFrame(
        {
            "pep": docs_df["pep"].to_list(),
            "first_half": first_halves,
            "second_half": second_halves,
        },
    )

    # Register model
    register_model(DEFAULT_MODEL_ID)

    # Embed all halves
    df_emb = df.fastembed.embed(
        columns=["first_half", "second_half"],
        model_name=DEFAULT_MODEL_ID,
        output_column=EMB_COL,
    )
    print("Embedded all document halves.")

    # Evaluate retrieval: rank of the correct second half for each first half
    ranks = []
    mismatches = []  # Store cases where rank != 1

    for row in df_emb.iter_rows(named=True):
        first_half_text = row["first_half"]
        second_half_text = row["second_half"]

        retrieved = df_emb.fastembed.retrieve(
            query=first_half_text,
            model_name=DEFAULT_MODEL_ID,
            embedding_column=EMB_COL,
            k=len(df_emb),
        )

        retrieved_texts = retrieved.select("second_half").to_series().to_list()
        retrieved_peps = (
            retrieved.select("pep").to_series().to_list()
        )  # IDs of retrieved PEPs
        rank = retrieved_texts.index(second_half_text) + 1  # 1-based
        ranks.append(rank)

        if rank != 1:
            mismatches.append(
                {
                    "pep": row["pep"],
                    "first_half": first_half_text,
                    "true_second_half": second_half_text,
                    "top_retrieved_second_half": retrieved_texts[0],
                    "top_retrieved_pep": retrieved_peps[
                        0
                    ],  # ID of incorrectly top-ranked PEP
                    "rank": rank,
                },
            )

    # Compute metrics
    ranks_series = pl.Series("rank", ranks)
    top1 = (ranks_series == 1).sum()
    top5 = (ranks_series <= 5).sum()
    top1pct = (ranks_series <= max(1, len(df) // 100)).sum()
    mean_rank = ranks_series.mean()

    print(f"\nHalf-document retrieval results on {len(df)} PEPs:")
    print(f"Top-1: {top1} ({top1 / len(df) * 100:.2f}%)")
    print(f"Top-5: {top5} ({top5 / len(df) * 100:.2f}%)")
    print(f"Top-1%: {top1pct} ({top1pct / len(df) * 100:.2f}%)")
    print(f"Mean rank of correct half: {mean_rank:.2f}")

    # Print mismatches for inspection
    if mismatches:
        mismatches_df = pl.DataFrame(mismatches)
        print("\nCases where the correct second half was NOT ranked 1:")
        print(mismatches_df)
    else:
        print("\nAll first halves retrieved their correct second half as top-1.")


if __name__ == "__main__":
    main()
