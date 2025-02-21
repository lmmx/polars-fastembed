use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use fastembed::{TextEmbedding, InitOptions};

use crate::model_suggestions::from_model_code;

#[derive(Deserialize)]
pub struct EmbedTextKwargs {
    /// The name/id of the model to load from Hugging Face or local ONNX
    #[serde(default)]
    pub model_id: Option<String>,
}

fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::List(Box::new(DataType::Float32))
    ))
}

/// Polars expression that reads a String column, embeds each row with fastembed-rs,
/// and returns a List(Float32). We bail if the column is not String.
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
    let ca = s.str()?; // Polars string chunked array

    // 3) Initialize the fastembed text model
    let embedder = match &kwargs.model_id {
        // If user explicitly passed a model_id, do your from_model_code logic:
        Some(model_id) => {
            let chosen_model = from_model_code(model_id)?;
            TextEmbedding::try_new(InitOptions::new(chosen_model).with_show_download_progress(false))
        }
        // If user did NOT pass one at all, rely on fastembed-rs's default
        None => TextEmbedding::try_new(InitOptions::default().with_show_download_progress(false)),
    }
    .map_err(|e| PolarsError::ComputeError(
        format!("Failed to load model: {e}").into()
    ))?;

    // 4) Embed row-by-row (or do batching for performance)
    let mut row_embeddings = Vec::with_capacity(ca.len());
    for opt_str in ca.into_iter() {
        if let Some(text) = opt_str {
            match embedder.embed([text].to_vec(), None) {
                Ok(mut results) => row_embeddings.push(results.pop()),
                Err(_err) => row_embeddings.push(None),
            }
        } else {
            // null row
            row_embeddings.push(None);
        }
    }

    // 5) Convert Vec<Option<Vec<f32>>> to a Polars List(Float32) column
    use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;

    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        "embedding".into(),
        row_embeddings.len(),
        0,
        DataType::Float32,
    );

    for opt_vec in row_embeddings {
        match opt_vec {
            Some(v) => builder.append_slice(&v),
            None => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}
