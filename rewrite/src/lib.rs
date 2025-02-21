use pyo3::prelude::*;

#[pymodule]
fn _polars_fastembed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
