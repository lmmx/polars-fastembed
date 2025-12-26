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
