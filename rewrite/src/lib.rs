use pyo3_polars::PolarsAllocator;
use pyo3::prelude::*;

mod expressions;

#[pymodule]
fn _polars_fastembed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
