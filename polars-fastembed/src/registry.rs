use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};

#[cfg(feature = "ort-dynamic")]
use ort::execution_providers::{ExecutionProviderDispatch, CPUExecutionProvider, CUDAExecutionProvider};
use ort::execution_providers::ExecutionProvider;
use fastembed::{InitOptions, TextEmbedding};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use polars::prelude::{PolarsError, PolarsResult};

use crate::model_suggestions::from_model_code;

#[cfg(feature = "ort-dynamic")]
fn default_providers() -> Vec<ExecutionProviderDispatch> {
    let cuda = CUDAExecutionProvider::default();
    eprintln!("CUDA is_available: {:?}", cuda.is_available());

    match ort::session::Session::builder() {
        Ok(mut builder) => {
            match cuda.register(&mut builder) {
                Ok(_) => eprintln!("CUDA provider registered successfully"),
                Err(e) => eprintln!("CUDA provider registration FAILED: {:?}", e),
            }
        }
        Err(e) => eprintln!("Session builder failed: {:?}", e),
    }

    vec![
        cuda.into(),
        CPUExecutionProvider::default().into(),
    ]
}

#[cfg(not(feature = "ort-dynamic"))]
fn default_providers() -> Vec<fastembed::ExecutionProviderDispatch> {
    // fastembed has its own type for this
    vec![]
}

/// Global registry of loaded models (model_name -> loaded `TextEmbedding`).
static MODEL_REGISTRY: Lazy<RwLock<HashMap<String, Arc<Mutex<TextEmbedding>>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Lazily-initialized default model for when no model_id is specified.
/// This avoids repeated model loads when calling embed_text without an explicit model.
static DEFAULT_MODEL: Lazy<Arc<Mutex<TextEmbedding>>> = Lazy::new(|| {
    let init = InitOptions::default()
        .with_show_download_progress(false)
        .with_execution_providers(default_providers());
    Arc::new(Mutex::new(
        TextEmbedding::try_new(init)
            .expect("Failed to load default embedding model"),
    ))
});

/// Extension trait to add dimension-related methods to TextEmbedding.
pub trait TextEmbeddingExt {
    fn get_dimension(&self) -> usize;
}

impl TextEmbeddingExt for Mutex<TextEmbedding> {
    fn get_dimension(&self) -> usize {
        let mut embedder = self.lock().unwrap();
        let test_text = "dimension_test";
        match embedder.embed(vec![test_text], None) {
            Ok(embeddings) if !embeddings.is_empty() => embeddings[0].len(),
            _ => panic!("Failed to determine embedding dimension"),
        }
    }
}

/// Parse e.g. ["CPUExecutionProvider"] => vec![ExecutionProviderDispatch]
#[cfg(feature = "ort-dynamic")]
fn parse_providers(provider_names: &[String]) -> Result<Vec<ExecutionProviderDispatch>, String> {
    let mut parsed = Vec::with_capacity(provider_names.len());
    for provider_str in provider_names {
        let dispatch: ExecutionProviderDispatch = match provider_str.as_str() {
            "CPUExecutionProvider" => CPUExecutionProvider::default().into(),
            "CUDAExecutionProvider" => CUDAExecutionProvider::default().into(),
            // Add more as needed...
            other => {
                return Err(format!(
                    "Unrecognized execution provider '{other}'. \
                     Must be one of: CPUExecutionProvider, CUDAExecutionProvider, ..."
                ));
            }
        };
        parsed.push(dispatch);
    }
    Ok(parsed)
}

/// Register a model (by huggingface ID or local path). If it's already loaded, does nothing.
///
/// Example providers: ["CPUExecutionProvider"], ["CUDAExecutionProvider"], etc.
#[pyfunction]
#[pyo3(signature = (model_name, providers=None))]
pub fn register_model(
    model_name: String,
    #[cfg_attr(not(feature = "ort-dynamic"), allow(unused_variables))]
    providers: Option<Vec<String>>,
) -> PyResult<()> {
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

    #[cfg_attr(not(feature = "ort-dynamic"), warn(unused_mut))]
    let mut init = InitOptions::new(embedding_model);

    #[cfg(feature = "ort-dynamic")]
    {
        let providers = match providers {
            Some(list) => {
                parse_providers(&list).map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err))?
            },
            None => default_providers(),
        };
        init = init.with_execution_providers(providers);
    }

    let embedder = TextEmbedding::try_new(init)
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load model '{model_name}': {e}")))?;

    eprintln!("Model '{}' loaded successfully", model_name);

    map.insert(model_name, Arc::new(Mutex::new(embedder)));
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

/// Return an Arc<TextEmbedding> from the registry.
/// - If `model_name` is None, returns the cached default model.
/// - If `model_name` is Some but not in the registry, loads and caches it.
pub fn get_or_load_model(model_name: &Option<String>) -> PolarsResult<Arc<Mutex<TextEmbedding>>> {
    // If no model name is provided, return the cached default
    if model_name.is_none() {
        return Ok(DEFAULT_MODEL.clone());
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
    let arc_embedder = Arc::new(Mutex::new(embedder));
    map.insert(name.clone(), arc_embedder.clone());
    Ok(arc_embedder)
}
