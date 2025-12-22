"""Demo of S³ (Semantic Signal Separation) topic modeling via polars-fastembed."""

import polars as pl
from polars_fastembed import register_model

# Documents with clear thematic clusters
docs = [
    # Technology / AI cluster
    "Machine learning models require large training datasets",
    "Neural networks use backpropagation for gradient optimization",
    "Deep learning models need GPU acceleration for training",
    "Transformers revolutionized natural language processing",
    "Convolutional networks excel at image recognition tasks",
    "Reinforcement learning agents learn through trial and error",
    "Large language models can generate human-like text",
    "Computer vision systems detect objects in images",
    "Speech recognition converts audio to text transcripts",
    "Autonomous vehicles use sensor fusion for navigation",
    # Finance / Business cluster
    "The stock market experienced significant volatility today",
    "Financial regulations affect banking operations globally",
    "Investment portfolios require proper diversification strategy",
    "Bond yields influence mortgage interest rates",
    "Hedge funds employ complex algorithmic trading strategies",
    "Central banks control monetary policy and inflation",
    "Cryptocurrency markets show high price fluctuations",
    "Venture capital funds invest in early stage startups",
    "Corporate earnings reports drive quarterly stock movements",
    "Interest rate decisions impact consumer borrowing costs",
    # Healthcare / Science cluster
    "Clinical trials test new pharmaceutical drug efficacy",
    "Genomic sequencing reveals genetic disease markers",
    "Vaccines stimulate immune system antibody production",
    "Medical imaging uses MRI and CT scan technology",
    "Cancer research focuses on targeted therapy treatments",
    "Epidemiologists track infectious disease outbreaks",
    "Stem cell research offers regenerative medicine possibilities",
    "Neuroscience studies brain function and cognition",
    "Drug discovery pipelines take years of development",
    "Public health initiatives promote preventive care",
]

model_id = "Xenova/bge-small-en-v1.5"

print("Registering model...")
register_model(model_id, providers=["CPUExecutionProvider"])

df = pl.DataFrame({"text": docs})

print(f"\nExtracting 3 topics from {len(docs)} documents...\n")

# =============================================================================
# Method 1: Get topic descriptions (top terms per topic)
# =============================================================================
print("=" * 60)
print("DISCOVERED TOPICS")
print("=" * 60)

topics = df.fastembed.extract_topics(
    text_column="text",
    n_components=3,
    model_name=model_id,
    top_n=8,
)

for i, topic in enumerate(topics):
    terms = [f"{t[0]} ({t[1]:.3f})" for t in topic]
    print(f"\nTopic {i}: {', '.join(terms)}")

# =============================================================================
# Method 2: Get document-topic assignments
# =============================================================================
print("\n" + "=" * 60)
print("DOCUMENT-TOPIC ASSIGNMENTS")
print("=" * 60)

result = df.fastembed.s3_topics(
    text_column="text",
    n_components=3,
    model_name=model_id,
)

# Show documents grouped by dominant topic
print("\nDocuments by dominant topic:\n")
for topic_id in range(3):
    topic_docs = result.filter(pl.col("dominant_topic") == topic_id)
    print(f"--- Topic {topic_id} ({len(topic_docs)} docs) ---")
    for row in topic_docs.iter_rows(named=True):
        weights = row["topic_weights"]
        text_preview = (
            row["text"][:50] + "..." if len(row["text"]) > 50 else row["text"]
        )
        print(
            f"  [{weights[0]:+.2f}, {weights[1]:+.2f}, {weights[2]:+.2f}] {text_preview}",
        )
    print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nDocuments per topic:")
print(result.group_by("dominant_topic").len().sort("dominant_topic"))

print("\nFull result DataFrame:")
print(result)
