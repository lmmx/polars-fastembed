#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel, ModelInfo};

#[derive(Deserialize)]
pub struct EmbedTextKwargs {
    /// The name/id of the model to load from Hugging Face or local ONNX
    pub model_id: String,
    // If you want more fields (e.g. batch_size), add them here:
    // pub batch_size: Option<usize>,
}

fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name.clone().as_str(),
        DataType::List(Box::new(DataType::Float32))
    ))
}

/// Finds which enum variant corresponds to the given model-code string.
fn from_model_code(code: &str) -> Option<EmbeddingModel> {
    // Retrieve the vector of `ModelInfo<EmbeddingModel>` for all known text-embedding models
    let all_models: Vec<ModelInfo<EmbeddingModel>> = TextEmbedding::list_supported_models();

    // Find the model whose `.model_code` matches `code`, then return its `.model` (the enum)
    all_models
        .into_iter()
        .find(|info| info.model_code == code)
        .map(|info| info.model)
}

/// Polars expression that reads a String column, embeds each row with fastembed-rs,
/// and returns a List(Float32). We bail if the column is not String.
#[polars_expr(output_type_func=list_idx_dtype)]
pub fn embed_text(inputs: &[Series], kwargs: EmbedTextKwargs) -> PolarsResult<Series> {
    // 1) Grab the input Series
    let s = &inputs[0];

    let chosen_model = from_model_code(&kwargs.model_id).ok_or_else(|| {
        PolarsError::ComputeError(
            format!("Unsupported or unknown model code: {}", kwargs.model_id).into()
        )
    })?;

    // 2) Check if it's a String column
    match s.dtype() {
        DataType::String => {
            let ca = s.str()?; // Polars string chunked array

            // 3) Initialize the fastembed text model
            let embedder = TextEmbedding::try_new(
                InitOptions::new(chosen_model)
                    .with_show_download_progress(false)
            ).map_err(|e| PolarsError::ComputeError(
                format!("Failed to load model {}: {e}", kwargs.model_id).into()
            ))?;

            // 4) Embed row-by-row (or you can do batching for performance)
            let mut row_embeddings = Vec::with_capacity(ca.len());

            for opt_str in ca.into_iter() {
                if let Some(text) = opt_str {
                    match embedder.embed(&[text], None) {
                        Ok(mut results) => {
                            if let Some(emb) = results.pop() {
                                row_embeddings.push(Some(emb));
                            } else {
                                // no embedding returned
                                row_embeddings.push(None);
                            }
                        },
                        Err(_err) => {
                            // embedding failed
                            row_embeddings.push(None);
                        }
                    }
                } else {
                    // null row
                    row_embeddings.push(None);
                }
            }

            // 5) Convert Vec<Option<Vec<f32>>> to a Polars List(Float32) column
            let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
                "embedding",
                row_embeddings.len(),
                0,
            );

            for opt_vec in row_embeddings {
                match opt_vec {
                    Some(v) => builder.append_values(&v),
                    None => builder.append_null(),
                }
            }

            Ok(builder.finish().into_series())
        },
        dtype => polars_bail!(InvalidOperation:
            format!("Data type {dtype} not supported. Must be a String column.")
        ),
    }
}
