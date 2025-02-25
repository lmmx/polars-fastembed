// See discussion 162
// https://github.com/PyO3/maturin-action/discussions/162#discussioncomment-7978369
#[cfg(feature = "openssl-vendored")]
use openssl_probe;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod expressions;
mod model_suggestions;
mod registry;
#[cfg(feature = "ort-dynamic")]
mod ort_helpers;

#[cfg(feature = "ort-dynamic")]
use dirs;

// See discussion 162
#[cfg(feature = "openssl-vendored")]
fn probe_ssl_certs() {
    openssl_probe::init_ssl_cert_env_vars();
}

#[cfg(feature = "ort-dynamic")]
fn init_ort() {
    // Set up the ONNX Runtime dynamic library
    if let Err(e) = ort_helpers::setup_ort_dylib() {
        eprintln!("Warning: Failed to set up ONNX Runtime dynamic library: {}", e);
        // Don't panic as this might not be fatal - user could have set ORT_DYLIB_PATH manually
    }
}

// --- Start of no-op zone ---

#[cfg(not(feature = "openssl-vendored"))]
fn probe_ssl_certs() {}

#[cfg(not(feature = "ort-dynamic"))]
fn init_ort() {}

// --- End of no-op zone ---

#[pymodule]
fn _polars_fastembed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // See discussion 162
    probe_ssl_certs();

    // Initialize ONNX Runtime if needed
    init_ort();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(registry::register_model, m)?)?;
    m.add_function(wrap_pyfunction!(registry::clear_registry, m)?)?;
    m.add_function(wrap_pyfunction!(registry::list_models, m)?)?;

    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
