//! Environment probe only; this is not the application's trading engine.
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn sum_squares(py: Python<'_>, values: Vec<f64>) -> f64 {
    py.detach(|| values.par_iter().map(|x| x * x).sum())
}

#[pymodule]
fn gen_trade_env_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_squares, m)?)?;
    Ok(())
}
