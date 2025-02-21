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

    // Look up or load the requested model (or the "default" if None)
    let embedder = get_or_load_model(&kwargs.model_id)?;

    let ca = s.str()?; // Polars string chunked array

    // Embed row-by-row (TODO: batch for performance)
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
