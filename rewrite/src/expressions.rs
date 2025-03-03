use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
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
    let dim = embedder.get_dimension()?;

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
    // 1) Grab the input Series
    let s = &inputs[0];

    // 2) Check it's a String column
    if s.dtype() != &DataType::String {
        polars_bail!(InvalidOperation:
            format!("Data type {:?} not supported. Must be a String column.", s.dtype())
        );
    }

    // Look up or load the requested model (or the "default" if None)
    let embedder = get_or_load_model(&kwargs.model_id)?;

    // Use the extension trait to get the embedding dimension
    use crate::registry::TextEmbeddingExt;
    let dim = embedder.get_dimension()?;

    let ca = s.str()?; // Polars string chunked array

    // Embed row-by-row (TODO: batch for performance)
    let mut row_embeddings = Vec::with_capacity(ca.len());
    for opt_str in ca.into_iter() {
        if let Some(text) = opt_str {
            match embedder.embed([text].to_vec(), None) {
                Ok(mut results) => {
                    if let Some(embedding) = results.pop() {
                        // Verify the dimension matches what we expect
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
            // null row
            row_embeddings.push(None);
        }
    }

    // Convert Vec<Option<Vec<f32>>> to a Polars Array(Float32, dim) column
    use polars::chunked_array::builder::ArrayBuilder;

    // Create an array builder with the fixed dimension
    let mut builder = ArrayBuilder::new(
        s.name(),
        ca.len(),
        DataType::Float32,
        dim,
    );

    for opt_embedding in row_embeddings {
        match opt_embedding {
            Some(vec) => builder.append_values(&vec),
            None => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}
