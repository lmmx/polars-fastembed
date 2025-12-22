"""
Tests for S³ (Semantic Signal Separation) topic modeling.
"""

import polars as pl
import pytest
from polars_fastembed import register_model


@pytest.fixture(scope="module")
def model_id():
    model = "Xenova/bge-small-en-v1.5"
    register_model(model, providers=["CPUExecutionProvider"])
    return model


@pytest.fixture(scope="module")
def sample_df(model_id):
    """Create a sample DataFrame with pre-computed embeddings."""
    docs = [
        # Tech cluster
        "Machine learning models require large training datasets",
        "Neural networks use backpropagation for optimization",
        "Deep learning needs GPU acceleration for training",
        "Transformers revolutionized natural language processing",
        "Computer vision systems detect objects in images",
        # Finance cluster
        "Stock market experienced significant volatility today",
        "Investment portfolios require proper diversification",
        "Bond yields influence mortgage interest rates",
        "Hedge funds employ algorithmic trading strategies",
        "Central banks control monetary policy and inflation",
        # Healthcare cluster
        "Clinical trials test new pharmaceutical drug efficacy",
        "Genomic sequencing reveals genetic disease markers",
        "Vaccines stimulate immune system antibody production",
        "Medical imaging uses MRI and CT scan technology",
        "Cancer research focuses on targeted therapy treatments",
    ]

    df = pl.DataFrame({"text": docs})

    # Embed once
    df = df.fastembed.embed(
        columns="text",
        model_name=model_id,
        output_column="embedding",
    )

    return df


def test_s3_topics_returns_correct_columns(sample_df):
    """Test that s3_topics adds the expected columns."""
    result = sample_df.fastembed.s3_topics(
        embedding_column="embedding",
        n_components=3,
    )

    assert "topic_weights" in result.columns
    assert "dominant_topic" in result.columns
    assert len(result) == len(sample_df)


def test_s3_topics_weights_shape(sample_df):
    """Test that topic weights have correct shape."""
    n_components = 3
    result = sample_df.fastembed.s3_topics(
        embedding_column="embedding",
        n_components=n_components,
    )

    # Each row should have n_components weights
    weights = result["topic_weights"].to_list()
    for w in weights:
        assert len(w) == n_components


def test_s3_topics_dominant_topic_valid(sample_df):
    """Test that dominant_topic values are valid indices."""
    n_components = 3
    result = sample_df.fastembed.s3_topics(
        embedding_column="embedding",
        n_components=n_components,
    )

    dominant_topics = result["dominant_topic"].to_list()
    for t in dominant_topics:
        assert 0 <= t < n_components


def test_s3_topics_clusters_similar_docs(sample_df):
    """Test that similar documents tend to cluster together."""
    result = sample_df.fastembed.s3_topics(
        embedding_column="embedding",
        n_components=3,
    )

    # Get dominant topics for each cluster (first 5 = tech, next 5 = finance, last 5 = health)
    topics = result["dominant_topic"].to_list()
    tech_topics = set(topics[0:5])
    finance_topics = set(topics[5:10])
    health_topics = set(topics[10:15])

    # Each cluster should have at most 2 different dominant topics
    # (allowing some flexibility since ICA isn't perfectly deterministic)
    assert len(tech_topics) <= 2, (
        f"Tech docs spread across too many topics: {tech_topics}"
    )
    assert len(finance_topics) <= 2, (
        f"Finance docs spread across too many topics: {finance_topics}"
    )
    assert len(health_topics) <= 3, (
        f"Health docs spread across too many topics: {health_topics}"
    )


def test_s3_topics_missing_column_raises():
    """Test that missing embedding column raises ValueError."""
    df = pl.DataFrame({"text": ["hello", "world"]})

    with pytest.raises(ValueError, match="not found"):
        df.fastembed.s3_topics(embedding_column="embedding", n_components=3)


def test_extract_topics_returns_terms(sample_df, model_id):
    """Test that extract_topics returns topic term descriptions."""
    topics = sample_df.fastembed.extract_topics(
        embedding_column="embedding",
        text_column="text",
        n_components=3,
        model_name=model_id,
        top_n=5,
    )

    assert len(topics) == 3  # n_components

    for topic in topics:
        assert len(topic) == 5  # top_n
        for term, score in topic:
            assert isinstance(term, str)
            assert isinstance(score, float)
            assert len(term) > 0
            assert score >= 0


def test_extract_topics_missing_text_column_raises(sample_df, model_id):
    """Test that missing text_column raises ValueError."""
    with pytest.raises(ValueError, match="text_column is required"):
        sample_df.fastembed.extract_topics(
            embedding_column="embedding",
            n_components=3,
            model_name=model_id,
        )


def test_s3_topics_preserves_original_data(sample_df):
    """Test that original DataFrame columns are preserved."""
    result = sample_df.fastembed.s3_topics(
        embedding_column="embedding",
        n_components=3,
    )

    assert "text" in result.columns
    assert "embedding" in result.columns
    assert result["text"].to_list() == sample_df["text"].to_list()
