use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// use fastembed::{ExecutionProviderDispatch, InitOptions, TextEmbedding};
use fastembed::{InitOptions, TextEmbedding};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use polars::prelude::{PolarsError, PolarsResult};

// use ort::execution_providers::{
//     CPUExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch,
// };

use crate::model_suggestions::from_model_code;

/// Global registry of loaded models (model_name -> loaded `TextEmbedding`).
static MODEL_REGISTRY: Lazy<RwLock<HashMap<String, Arc<TextEmbedding>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

// Extension trait to add dimension-related methods to TextEmbedding
pub trait TextEmbeddingExt {
    fn get_dimension(&self) -> usize;
}

impl TextEmbeddingExt for TextEmbedding {
    fn get_dimension(&self) -> usize {
        // Run a test embedding to determine the dimension
        let test_text = "dimension_test";
        match self.embed(vec![test_text], None) {
            Ok(embeddings) if !embeddings.is_empty() => embeddings[0].len(),
            _ => panic!("Failed to determine embedding dimension"),
        }
    }
}

// /// Parse e.g. ["CPUExecutionProvider"] => vec![ExecutionProviderDispatch::CPU]
// fn parse_providers(provider_names: &[String]) -> Result<Vec<ExecutionProviderDispatch>, String> {
//     let mut parsed = Vec::with_capacity(provider_names.len());
//     for provider_str in provider_names {
//         match provider_str.as_str() {
//             "CPUExecutionProvider" => {
//                 // Wrap the default CPU provider in a dispatch
//                 let dispatch = ExecutionProviderDispatch::new(CPUExecutionProvider::default());
//                 parsed.push(dispatch);
//             }
//             "CUDAExecutionProvider" => {
//                 // Similarly, for CUDA (make sure your crate has the `cuda` feature enabled)
//                 let dispatch = ExecutionProviderDispatch::new(CUDAExecutionProvider::default());
//                 parsed.push(dispatch);
//             }
//             // You can handle more, e.g. "TensorRTExecutionProvider", "CoreMLExecutionProvider", etc...
//             other => {
//                 return Err(format!(
//                     "Unrecognized execution provider '{other}'. \
//                      Must be one of: CPUExecutionProvider, CUDAExecutionProvider, ..."
//                 ));
//             }
//         }
//     }
//     Ok(parsed)
// }


// /// Register a model (by huggingface ID or local path) with optional providers.
// /// If it's already loaded, does nothing.
// ///
// /// Example providers: ["CPUExecutionProvider"], ["CUDAExecutionProvider"], etc.
// #[pyfunction(signature = (model_name, providers=None))]
// pub fn register_model(model_name: String, providers: Option<Vec<String>>) -> PyResult<()> {
/// Register a model (by huggingface ID or local path). If it's already loaded, does nothing.
#[pyfunction]
pub fn register_model(model_name: String) -> PyResult<()> {
    let mut map = MODEL_REGISTRY
        .write()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poison"))?;

    // Already loaded?
    if map.contains_key(&model_name) {
        return Ok(());
    }

    // from_model_code either returns a known EmbeddingModel or error with suggestions
    let embedding_model = from_model_code(&model_name).map_err(|polars_err| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(polars_err.to_string())
    })?;

    // Build the init options
    let init = InitOptions::new(embedding_model);
    // let mut init = InitOptions::new(embedding_model); // only needs mut if changing providers

    // if let Some(provider_list) = providers {
    //     // parse the strings -> Vec<ExecutionProviderDispatch>
    //     let dispatches = parse_providers(&provider_list)
    //         .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err))?;
    //     // pass to fastembed
    //     init = init.with_execution_providers(dispatches);
    // }

    // Actually load the model
    let embedder = TextEmbedding::try_new(init)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load model '{model_name}': {e}")))?;

    map.insert(model_name, Arc::new(embedder));
    Ok(())
}

/// Clear the entire model registry (free memory).
#[pyfunction]
pub fn clear_registry() -> PyResult<()> {
    let mut map = MODEL_REGISTRY
        .write()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poison"))?;
    map.clear();
    Ok(())
}

/// Return a list of currently registered model names.
#[pyfunction]
pub fn list_models() -> PyResult<Vec<String>> {
    let map = MODEL_REGISTRY
        .read()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poison"))?;
    Ok(map.keys().cloned().collect())
}

/// Return an Arc<TextEmbedding> from the registry. If None, load the default fastembed model.
/// If Some(...) is not found in the registry, we load it via from_model_code(...) or error.
pub fn get_or_load_model(model_name: &Option<String>) -> PolarsResult<Arc<TextEmbedding>> {
    // If no model name is provided, use fastembed's default
    if model_name.is_none() {
        let embedder = TextEmbedding::try_new(InitOptions::default()).map_err(|e| {
            PolarsError::ComputeError(format!("Failed to load default model: {e}").into())
        })?;
        return Ok(Arc::new(embedder));
    }
    let name = model_name.as_ref().unwrap();

    // Lock the registry
    let mut map = MODEL_REGISTRY
        .write()
        .map_err(|_| PolarsError::ComputeError("Lock poison".into()))?;

    // Already loaded?
    if let Some(arc_embedder) = map.get(name) {
        return Ok(arc_embedder.clone());
    }

    // Not loaded => try to load it now
    let embedding_model = from_model_code(name).map_err(|e| {
        PolarsError::ComputeError(format!("While loading {name}: {e}").into())
    })?;

    let init = InitOptions::new(embedding_model).with_show_download_progress(false);
    let embedder = TextEmbedding::try_new(init)
        .map_err(|e| PolarsError::ComputeError(format!("Failed to load {name}: {e}").into()))?;
    let arc_embedder = Arc::new(embedder);
    map.insert(name.clone(), arc_embedder.clone());
    Ok(arc_embedder)
}
