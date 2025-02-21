use pyo3::prelude::*;
use polars::prelude::*;

mod expressions;
use expressions::embed_text;

#[pymodule]
fn _polars_fastembed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(embed_text, m)?)?;
    Ok(())
}
