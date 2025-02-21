use pyo3_polars::PolarsAllocator;
use pyo3::prelude::*;

mod expressions;
mod model_suggestions;
mod registry;

#[pymodule]
fn _polars_fastembed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(registry::register_model, m)?)?;
    m.add_function(wrap_pyfunction!(registry::clear_registry, m)?)?;
    m.add_function(wrap_pyfunction!(registry::list_models, m)?)?;

    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
