Some timings from a small number of rows

The fastest is `Xenova/all-MiniLM-L6-v2`, so this is the default.

|                     model                    | time_seconds |
| -------------------------------------------- | ------------ |
| Xenova/all-MiniLM-L6-v2                      | 1.41         |
| Xenova/bge-small-zh-v1.5                     | 1.50         |
| Qdrant/all-MiniLM-L6-v2-onnx                 | 1.60         |
| Qdrant/clip-ViT-B-32-text                    | 1.68         |
| Xenova/bge-small-en-v1.5                     | 2.37         |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 | 3.20         |
| intfloat/multilingual-e5-small               | 3.24         |
| Xenova/all-MiniLM-L12-v2                     | 3.41         |
| Xenova/bge-base-en-v1.5                      | 4.46         |
| intfloat/multilingual-e5-base                | 5.52         |
| Alibaba-NLP/gte-base-en-v1.5                 | 5.54         |
| nomic-ai/nomic-embed-text-v1                 | 5.72         |
| jinaai/jina-embeddings-v2-base-code          | 6.51         |
| Xenova/paraphrase-multilingual-mpnet-base-v2 | 7.20         |
| Xenova/bge-large-en-v1.5                     | 13.95        |
| Qdrant/multilingual-e5-large-onnx            | 14.07        |
| lightonai/modernbert-embed-large             | 17.13        |
| Alibaba-NLP/gte-large-en-v1.5                | 17.79        |
| mixedbread-ai/mxbai-embed-large-v1           | 43.78        |
| nomic-ai/nomic-embed-text-v1.5               | 55.29        |

So embedding and retrieval of the MiniLM-L6-v2 on all PEPs

- Total token count: 3,615,903 = 3.6M tokens in 30s = 8ms per 1k tokens

  - Roughly 2x as fast as recently reported 14.7ms per 1k tokens [here](https://www.reddit.com/r/LocalLLaMA/comments/1nrgklt/opensource_embedding_models_which_one_to_use/) or [here](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)

- On GPU, this falls to 5.3ms per 1k tokens (20s total time to run), with all cores still used for a
  significant portion of the computation
