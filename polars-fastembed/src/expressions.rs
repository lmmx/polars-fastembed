use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use crate::registry::TextEmbeddingExt;
use serde::Deserialize;

use crate::registry::get_or_load_model;

#[derive(Deserialize)]
pub struct EmbedTextKwargs {
    /// The name/id of the model to load from Hugging Face or local ONNX
    #[serde(default)]
    pub model_id: Option<String>,
}

fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    // Get the embedder to retrieve the dimension
    let embedder = get_or_load_model(&None)?;

    // Use the extension trait to get the dimension
    use crate::registry::TextEmbeddingExt;
    let dim = embedder.get_dimension();

    // Return a fixed-size array type with the retrieved dimension
    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::Array(Box::new(DataType::Float32), dim)
    ))
}

/// Polars expression that reads a String column, embeds each row with fastembed-rs,
/// and returns a fixed-size Array(Float32, dim). We bail if the column is not String.
#[polars_expr(output_type_func=list_idx_dtype)]
pub fn embed_text(inputs: &[Series], kwargs: EmbedTextKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];

    if s.dtype() != &DataType::String {
        polars_bail!(InvalidOperation:
            format!("Data type {:?} not supported. Must be a String column.", s.dtype())
        );
    }

    let embedder = get_or_load_model(&kwargs.model_id)?;
    let dim = embedder.get_dimension();  // Locks, gets dim, unlocks

    let mut embedder_guard = embedder.lock()
        .map_err(|_| PolarsError::ComputeError("Lock poison".into()))?;

    let ca = s.str()?;

    let mut row_embeddings = Vec::with_capacity(ca.len());
    for opt_str in ca.into_iter() {
        if let Some(text) = opt_str {
            match embedder_guard.embed(vec![text], None) {
                Ok(mut results) => {
                    if let Some(embedding) = results.pop() {
                        if embedding.len() == dim {
                            row_embeddings.push(Some(embedding));
                        } else {
                            polars_bail!(ComputeError:
                                format!("Embedding dimension mismatch: expected {}, got {}", dim, embedding.len())
                            );
                        }
                    } else {
                        row_embeddings.push(None);
                    }
                },
                Err(_err) => row_embeddings.push(None),
            }
        } else {
            row_embeddings.push(None);
        }
    }

    drop(embedder_guard);  // Explicitly release the lock

    use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;

    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        s.name().clone(),
        row_embeddings.len(),
        row_embeddings.len() * dim,
        DataType::Float32,
    );

    for opt_vec in row_embeddings {
        match opt_vec {
            Some(v) => {
                if v.len() != dim {
                    polars_bail!(ComputeError:
                        format!("Embedding dimension mismatch: expected {}, got {}", dim, v.len())
                    );
                }
                builder.append_slice(&v);
            },
            None => builder.append_null(),
        }
    }

    let list_series = builder.finish().into_series();
    let array_series = list_series.cast(&DataType::Array(Box::new(DataType::Float32), dim))?;

    Ok(array_series)
}
