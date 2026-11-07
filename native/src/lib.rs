//! Causal directional-change extraction and batched trade execution; both
//! match their Python references exactly.
mod execution;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type DCColumns = (
    Vec<i8>,
    Vec<bool>,
    Vec<f64>,
    Vec<i64>,
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Vec<i64>,
    Vec<bool>,
);

fn profile(prices: Vec<f64>, threshold: f64) -> DCColumns {
    let n = prices.len();
    let mut direction = vec![0_i8; n];
    let mut confirmation = vec![false; n];
    let mut anchor_prices = vec![f64::NAN; n];
    let mut anchor_indices = vec![-1_i64; n];
    let mut confirmation_indices = vec![-1_i64; n];
    let mut confirmation_prices = vec![f64::NAN; n];
    let mut extreme_prices = prices.clone();
    let mut extreme_indices = (0..n as i64).collect::<Vec<_>>();
    let mut overshoots = vec![false; n];
    if n != 0 {
        let mut state = 0_i8;
        let (mut high, mut low) = (prices[0], prices[0]);
        let (mut high_index, mut low_index) = (0_i64, 0_i64);
        let (mut anchor_price, mut confirmation_price) = (f64::NAN, f64::NAN);
        let (mut anchor_index, mut confirmation_index) = (-1_i64, -1_i64);
        let (mut extreme_price, mut extreme_index) = (prices[0], 0_i64);
        let mut has_overshoot = false;
        for (i, price) in prices.into_iter().enumerate() {
            let index = i as i64;
            let mut confirmed = false;
            if state == 0 {
                if price > high {
                    (high, high_index) = (price, index);
                }
                if price < low {
                    (low, low_index) = (price, index);
                }
                if price > low && price >= low * (1.0 + threshold) {
                    (state, anchor_price, anchor_index) = (1, low, low_index);
                    confirmed = true;
                } else if price < high && price <= high * (1.0 - threshold) {
                    (state, anchor_price, anchor_index) = (-1, high, high_index);
                    confirmed = true;
                }
            } else if state == 1 {
                if price < extreme_price && price <= extreme_price * (1.0 - threshold) {
                    (state, anchor_price, anchor_index) = (-1, extreme_price, extreme_index);
                    confirmed = true;
                } else if price > extreme_price {
                    (extreme_price, extreme_index, has_overshoot) = (price, index, true);
                }
            } else if price > extreme_price && price >= extreme_price * (1.0 + threshold) {
                (state, anchor_price, anchor_index) = (1, extreme_price, extreme_index);
                confirmed = true;
            } else if price < extreme_price {
                (extreme_price, extreme_index, has_overshoot) = (price, index, true);
            }
            if confirmed {
                confirmation_price = price;
                confirmation_index = index;
                (extreme_price, extreme_index, has_overshoot) = (price, index, false);
            }
            direction[i] = state;
            confirmation[i] = confirmed;
            if state != 0 {
                anchor_prices[i] = anchor_price;
                anchor_indices[i] = anchor_index;
                confirmation_prices[i] = confirmation_price;
                confirmation_indices[i] = confirmation_index;
                extreme_prices[i] = extreme_price;
                extreme_indices[i] = extreme_index;
                overshoots[i] = has_overshoot;
            }
        }
    }
    (
        direction,
        confirmation,
        anchor_prices,
        anchor_indices,
        confirmation_indices,
        confirmation_prices,
        extreme_prices,
        extreme_indices,
        overshoots,
    )
}

#[pyfunction]
fn dc_arrays(py: Python<'_>, prices: Vec<f64>, threshold: f64) -> PyResult<DCColumns> {
    if !threshold.is_finite() || !(0.0 < threshold && threshold < 1.0) {
        return Err(PyValueError::new_err(
            "threshold must be finite and between zero and one",
        ));
    }
    if prices
        .iter()
        .any(|price| !price.is_finite() || *price <= 0.0)
    {
        return Err(PyValueError::new_err(
            "prices must be finite and strictly positive",
        ));
    }
    Ok(py.detach(|| profile(prices, threshold)))
}

#[pymodule]
fn gen_trade_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dc_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(execution::execute_batch_arrays, m)?)?;
    Ok(())
}
