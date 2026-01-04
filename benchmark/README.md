Some timings from a small number of rows

The fastest on CPU is `Xenova/all-MiniLM-L6-v2`, so this is the default.

The fastest on GPU is `SnowflakeArcticEmbedXS`

- Note 1: to achieve this performance you must enable both GPU and CPU executors,
  not only the GPU one)
- Note 2: to ensure you don't get the quantised version of the model, which performs worse with GPU,
  pass the code rather than the repo, as the repo contains both quant. and full versions of the model.

Embedding and retrieval of the MiniLM-L6-v2 on all PEPs:

- Total token count: 3,615,903 = 3.6M tokens in 30s = 8ms per 1k tokens

  - Roughly 2x as fast as recently reported 14.7ms per 1k tokens [here](https://www.reddit.com/r/LocalLLaMA/comments/1nrgklt/opensource_embedding_models_which_one_to_use/) or [here](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)

- On GPU with Snowflake Arctic Embed XS, this falls to 5.3ms per 1k tokens (20s total time to run), with all cores still used for a
  significant portion of the computation

### Intrinsic dimension

Also included are some experiments with intrinsic dimension, reproducible with:

```sh
uv run estimate-id
uv run estimate-wiki-id
uv run estimate-revs-id
uv run estimate-dups-id
uv run estimate-news-id
uv run estimate-poems-id
uv run estimate-code-id
uv run estimate-legal-id
```

Respectively these are PEPs, Wikipedia, Amazon reviews, duplicates (a control that should be ~0),
news articles, poems, code (Python), and US Supreme Court legal documents.

## Model Comparison

| Dataset | MiniLM-L6-v2 (k=20) | Snowflake Arctic XS (k=20) | Δ |
|---------|---------------------|---------------------------|-----|
| **Near-duplicates** | 0.2 | 0.2 | — |
| **PEPs** | 16.1 | 19.9 | +3.8 |
| **AG News** | 19.5 | 22.5 | +3.0 |
| **Amazon Reviews** | 22.0 | 19.2* | -2.8 |
| **Poetry** | 21.3 | 24.6 | +3.3 |
| **Python Code** | 20.9 | 25.3 | +4.4 |
| **US Supreme Court** | 23.0 | 28.4 | +5.4 |
| **Wikipedia (sparse)** | 25.0 | 32.8 | +7.8 |

## Observations

1. **Rank order is preserved** — the narrowest (PEPs) and broadest (Wikipedia) corpora stay at their relative positions

2. **Snowflake consistently runs ~3-8 higher** — this suggests Snowflake's embedding space is less compressed / more isotropic than MiniLM, spreading data across more effective dimensions

3. **The gap widens for diverse corpora** — Wikipedia jumps most (+7.8), while narrow PEPs jump least (+3.8). Snowflake seems to "use more dimensions" when content is diverse.

4. **Control still works** — near-duplicates remain at ~0 for both models

## Interpretation

ID is **both** a property of the dataset *and* the model. Different embedding models carve up semantic space differently:
- MiniLM might be more aggressively clustering similar concepts
- Snowflake might preserve finer distinctions, spreading embeddings across more dimensions

This aligns with the research finding that embedding model architecture/training affects ID independently of content.
