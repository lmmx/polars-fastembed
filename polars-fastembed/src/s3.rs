use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Predict};
use linfa_ica::fast_ica::{FastIca, GFunc};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::HashMap;

use crate::registry::get_or_load_model;

#[derive(Deserialize)]
pub struct S3Kwargs {
    pub n_components: usize,
}

fn s3_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::List(Box::new(DataType::Float32)),
    ))
}

// ============================================================================
// S³ Model using linfa's FastICA
// ============================================================================

pub struct S3Model {
    pub mean: Array1<f64>,
    pub ica_model: FastIca<f64>,
    pub document_topics: Array2<f64>,
}

impl S3Model {
    pub fn fit(embeddings: Array2<f64>, n_components: usize) -> PolarsResult<Self> {
        let (n_docs, _dim) = embeddings.dim();

        if n_docs <= n_components {
            polars_bail!(ComputeError:
                "Need more than {} documents for {} components, got {}",
                n_components, n_components, n_docs
            );
        }

        // Center the data
        let mean = embeddings.mean_axis(Axis(0)).unwrap();
        let centered = &embeddings - &mean;

        let centered_owned = centered.to_owned();
        let dataset = DatasetBase::from(centered_owned.clone());

        let ica_model = FastIca::params()
            .ncomponents(n_components)
            .gfunc(GFunc::Logcosh(1.0))
            .fit(&dataset)
            .map_err(|e| PolarsError::ComputeError(format!("FastICA failed: {:?}", e).into()))?;

        let document_topics = ica_model.predict(&centered_owned);

        Ok(S3Model {
            mean,
            ica_model,
            document_topics,
        })
    }

    pub fn transform(&self, embeddings: &Array2<f64>) -> Array2<f64> {
        let centered = embeddings - &self.mean;
        self.ica_model.predict(&centered)
    }
}

// ============================================================================
// Helper: Extract embeddings from Array column
// ============================================================================

fn extract_embedding_matrix(series: &Series) -> PolarsResult<(Array2<f64>, Vec<usize>)> {
    let arr = series.array()?;
    let dim = match series.dtype() {
        DataType::Array(_, size) => *size,
        _ => polars_bail!(InvalidOperation: "Expected Array type"),
    };

    let mut valid_indices: Vec<usize> = Vec::new();
    let mut embeddings: Vec<f64> = Vec::new();

    for (i, opt_arr) in arr.into_iter().enumerate() {
        if let Some(inner) = opt_arr {
            let f32_chunked = inner.f32()?;
            let values: Vec<f64> = f32_chunked
                .into_iter()
                .map(|opt_v| opt_v.unwrap_or(0.0) as f64)
                .collect();

            if values.len() == dim {
                embeddings.extend(values);
                valid_indices.push(i);
            }
        }
    }

    if valid_indices.is_empty() {
        polars_bail!(ComputeError: "No valid embeddings found");
    }

    let n_docs = valid_indices.len();
    let embedding_matrix = Array2::from_shape_vec((n_docs, dim), embeddings)
        .map_err(|e| PolarsError::ComputeError(format!("Shape error: {:?}", e).into()))?;

    Ok((embedding_matrix, valid_indices))
}

// ============================================================================
// Polars Expression: Fit topics from embedding column
// ============================================================================

#[polars_expr(output_type_func=s3_output_type)]
pub fn s3_fit_transform(inputs: &[Series], kwargs: S3Kwargs) -> PolarsResult<Series> {
    let s = &inputs[0];

    // Validate input is an Array type
    match s.dtype() {
        DataType::Array(inner, _) if **inner == DataType::Float32 => {},
        _ => polars_bail!(InvalidOperation:
            "s3_fit_transform requires Array[f32, n] column, got {:?}",
            s.dtype()
        ),
    }

    let (embedding_matrix, valid_indices) = extract_embedding_matrix(s)?;
    let model = S3Model::fit(embedding_matrix, kwargs.n_components)?;

    // Build output
    let total_rows = s.len();
    let n_components = model.document_topics.ncols();

    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        s.name().clone(),
        total_rows,
        total_rows * n_components,
        DataType::Float32,
    );

    let mut doc_idx = 0;
    for i in 0..total_rows {
        if doc_idx < valid_indices.len() && valid_indices[doc_idx] == i {
            let weights: Vec<f32> = model
                .document_topics
                .row(doc_idx)
                .iter()
                .map(|&x| x as f32)
                .collect();
            builder.append_slice(&weights);
            doc_idx += 1;
        } else {
            builder.append_null();
        }
    }

    Ok(builder.finish().into_series())
}

// ============================================================================
// Python Function: Extract topics with term labels
// ============================================================================

#[pyfunction]
#[pyo3(signature = (embeddings, texts, n_components, model_id=None, top_n=10))]
pub fn extract_topics(
    embeddings: Vec<Vec<f32>>,
    texts: Vec<String>,
    n_components: usize,
    model_id: Option<String>,
    top_n: usize,
) -> PyResult<Vec<Vec<(String, f32)>>> {
    let n_docs = embeddings.len();
    if n_docs != texts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "embeddings and texts must have same length"
        ));
    }

    if n_docs == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("No documents provided"));
    }

    let dim = embeddings[0].len();

    // Build embedding matrix from provided embeddings
    let mut embedding_matrix = Array2::<f64>::zeros((n_docs, dim));
    for (i, emb) in embeddings.iter().enumerate() {
        for (j, &val) in emb.iter().enumerate() {
            embedding_matrix[[i, j]] = val as f64;
        }
    }

    let model = S3Model::fit(embedding_matrix, n_components)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Build vocabulary from texts
    let mut vocab: HashMap<String, usize> = HashMap::new();
    for doc in &texts {
        for word in doc.split_whitespace() {
            let clean: String = word
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect();
            if clean.len() > 2 {
                let next_idx = vocab.len();
                vocab.entry(clean).or_insert(next_idx);
            }
        }
    }

    let mut vocab_list: Vec<String> = vec![String::new(); vocab.len()];
    for (word, &idx) in &vocab {
        vocab_list[idx] = word.clone();
    }

    if vocab_list.is_empty() {
        return Ok(vec![vec![]; n_components]);
    }

    // Only here do we need the embedder - for vocabulary words
    let embedder = get_or_load_model(&model_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut guard = embedder
        .lock()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Lock poison"))?;

    let vocab_refs: Vec<&str> = vocab_list.iter().map(|s| s.as_str()).collect();
    let word_embeddings_raw = guard
        .embed(vocab_refs, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

    drop(guard);

    let mut word_embeddings = Array2::<f64>::zeros((vocab_list.len(), dim));
    for (i, emb) in word_embeddings_raw.iter().enumerate() {
        for (j, &val) in emb.iter().enumerate() {
            word_embeddings[[i, j]] = val as f64;
        }
    }

    // Project vocabulary onto topic space
    let word_topics = model.transform(&word_embeddings);

    // Extract top terms per topic
    let mut topics = Vec::with_capacity(n_components);
    for topic_idx in 0..n_components {
        let mut word_scores: Vec<(usize, f32)> = word_topics
            .column(topic_idx)
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score.abs() as f32))
            .collect();

        word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_terms: Vec<(String, f32)> = word_scores
            .into_iter()
            .take(top_n)
            .map(|(idx, score)| (vocab_list[idx].clone(), score))
            .collect();

        topics.push(top_terms);
    }

    Ok(topics)
}
