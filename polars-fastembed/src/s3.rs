use ndarray::{Array1, Array2, Axis};
use picard::{Picard, PicardConfig, PicardResult, DensityType};
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
// S³ Model using Picard ICA
// ============================================================================

pub struct S3Model {
    mean: Array1<f64>,
    picard_result: PicardResult,
    // Cache the full unmixing matrix for transforms
    full_unmixing: Array2<f64>,
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

        // Compute mean for later transforms
        let mean = embeddings.mean_axis(Axis(0)).unwrap();

        // Picard expects (n_features x n_samples)
        // embeddings is (n_docs x dim), so we need (dim x n_docs)
        // reversed_axes reinterprets layout without copying
        let x = embeddings.reversed_axes();

        let config = PicardConfig::builder()
            .n_components(n_components)
            .max_iter(200)
            .tol(1e-4)
            .density(DensityType::tanh())
            .ortho(false)
            .random_state(0)
            .build();

        let picard_result = Picard::fit_with_config(&x, &config)
            .map_err(|e| PolarsError::ComputeError(format!("Picard ICA failed: {e}").into()))?;

        // Cache full unmixing matrix for efficient transforms
        let full_unmixing = picard_result.full_unmixing();

        Ok(S3Model {
            mean,
            picard_result,
            full_unmixing,
        })
    }

    /// Number of components (topics)
    #[inline]
    pub fn n_components(&self) -> usize {
        self.picard_result.sources.nrows()
    }

    /// Get topic weights for a single document by index.
    /// Collects into Vec to avoid lifetime issues with column view.
    #[inline]
    pub fn document_weights(&self, doc_idx: usize) -> Vec<f32> {
        // sources is (n_components x n_samples), column gives weights for one doc
        self.picard_result
            .sources
            .column(doc_idx)
            .iter()
            .map(|&w| w as f32)
            .collect()
    }

    /// Transform new embeddings, yielding weights per sample.
    pub fn transform_sample_weights<'a>(
        &'a self,
        embeddings: &'a Array2<f64>,
    ) -> impl Iterator<Item = Vec<f32>> + 'a {
        let n_components = self.n_components();

        embeddings.rows().into_iter().map(move |sample| {
            let mut weights = Vec::with_capacity(n_components);

            for comp_idx in 0..n_components {
                let unmixing_row = self.full_unmixing.row(comp_idx);
                let weight: f64 = unmixing_row
                    .iter()
                    .zip(sample.iter())
                    .zip(self.mean.iter())
                    .map(|((&u, &s), &m)| u * (s - m))
                    .sum();
                weights.push(weight as f32);
            }

            weights
        })
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

    let n_total = arr.len();
    let mut valid_indices = Vec::with_capacity(n_total);
    let mut embeddings = Vec::with_capacity(n_total * dim);

    for (i, opt_arr) in arr.into_iter().enumerate() {
        if let Some(inner) = opt_arr {
            let f32_chunked = inner.f32()?;

            // Check if all values are present and correct length
            if f32_chunked.len() == dim && f32_chunked.null_count() == 0 {
                embeddings.extend(f32_chunked.into_no_null_iter().map(|v| v as f64));
                valid_indices.push(i);
            }
        }
    }

    if valid_indices.is_empty() {
        polars_bail!(ComputeError: "No valid embeddings found");
    }

    let n_docs = valid_indices.len();
    let embedding_matrix = Array2::from_shape_vec((n_docs, dim), embeddings)
        .map_err(|e| PolarsError::ComputeError(format!("Shape error: {e}").into()))?;

    Ok((embedding_matrix, valid_indices))
}

// ============================================================================
// Polars Expression: Fit topics from embedding column
// ============================================================================

#[polars_expr(output_type_func=s3_output_type)]
pub fn s3_fit_transform(inputs: &[Series], kwargs: S3Kwargs) -> PolarsResult<Series> {
    let s = &inputs[0];

    match s.dtype() {
        DataType::Array(inner, _) if **inner == DataType::Float32 => {}
        _ => polars_bail!(InvalidOperation:
            "s3_fit_transform requires Array[f32, n] column, got {:?}",
            s.dtype()
        ),
    }

    let (embedding_matrix, valid_indices) = extract_embedding_matrix(s)?;
    let model = S3Model::fit(embedding_matrix, kwargs.n_components)?;

    eprintln!("Picard converged: {}, n_iter: {}, final_grad: {:.4e}",
        model.picard_result.converged,
        model.picard_result.n_iterations,
        model.picard_result.gradient_norm);

    let total_rows = s.len();
    let n_components = model.n_components();

    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        s.name().clone(),
        total_rows,
        total_rows * n_components,
        DataType::Float32,
    );

    // Use a cursor into valid_indices for O(n) traversal
    let mut valid_cursor = 0;

    for row_idx in 0..total_rows {
        if valid_cursor < valid_indices.len() && valid_indices[valid_cursor] == row_idx {
            let weights = model.document_weights(valid_cursor);
            builder.append_slice(&weights);
            valid_cursor += 1;
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
            "embeddings and texts must have same length",
        ));
    }

    if n_docs == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No documents provided",
        ));
    }

    let dim = embeddings[0].len();

    // Build embedding matrix (n_docs x dim) directly from flat iterator
    let flat_data: Vec<f64> = embeddings
        .iter()
        .flat_map(|emb| emb.iter().map(|&v| v as f64))
        .collect();

    let embedding_matrix = Array2::from_shape_vec((n_docs, dim), flat_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

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

            if clean.len() > 2 && !vocab.contains_key(&clean) {
                let idx = vocab.len();
                vocab.insert(clean, idx);
            }
        }
    }

    if vocab.is_empty() {
        return Ok(vec![vec![]; n_components]);
    }

    // Build vocab list ordered by index
    let mut vocab_list = vec![String::new(); vocab.len()];
    for (word, &idx) in &vocab {
        vocab_list[idx].clone_from(word);
    }

    // Embed vocabulary words
    let embedder = get_or_load_model(&model_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let word_embeddings_raw = {
        let mut guard = embedder
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Lock poison"))?;

        let vocab_refs: Vec<&str> = vocab_list.iter().map(String::as_str).collect();

        guard
            .embed(vocab_refs, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?
    };

    // Build word embeddings matrix
    let word_flat: Vec<f64> = word_embeddings_raw
        .iter()
        .flat_map(|emb| emb.iter().map(|&v| v as f64))
        .collect();

    let word_embeddings = Array2::from_shape_vec((vocab_list.len(), dim), word_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))?;

    // Collect word scores per topic
    let mut topic_scores: Vec<Vec<(usize, f32)>> =
        vec![Vec::with_capacity(vocab_list.len()); n_components];

    for (word_idx, weights) in model.transform_sample_weights(&word_embeddings).enumerate() {
        for (topic_idx, &weight) in weights.iter().enumerate() {
            topic_scores[topic_idx].push((word_idx, weight.abs()));
        }
    }

    // Sort and take top_n for each topic
    let topics: Vec<Vec<(String, f32)>> = topic_scores
        .iter_mut()
        .map(|scores| {
            scores.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            scores
                .iter()
                .take(top_n)
                .map(|&(word_idx, score)| (vocab_list[word_idx].clone(), score))
                .collect()
        })
        .collect();

    Ok(topics)
}
