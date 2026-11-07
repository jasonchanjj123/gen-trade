//! Batched account-currency stock/FX execution.
//!
//! Matches `gen_trade.market_execution`'s Python reference exactly: every
//! sleeve performs the same floating-point operations in the same order.
//! Strategies are independent, so each runs through all bars on its own with
//! its state in registers. The reference's one cross-strategy condition
//! (financing is skipped when no strategy holds a position) only skips adding
//! an exact zero to flat strategies, so skipping it per strategy is identical.

use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const YEAR_NS: f64 = 365.25 * 86400.0 * 1e9;
const SIGNAL: u8 = 0;
const MARGIN_CALL: u8 = 3;
const HOLDING_DEADLINE: u8 = 4;
const SAMPLE_END: u8 = 5;
const QUOTE_ROWS: usize = 10;
const CLOCK_ROWS: usize = 3;
const TRADE_INTS: usize = 5;
const TRADE_FLOATS: usize = 11;

/// Costs and contract terms, read from a Python dict with these keys.
#[derive(FromPyObject, Clone, Copy)]
#[pyo3(from_item_all)]
pub(crate) struct Costs {
    slip: f64,
    fee_rate: f64,
    minimum_commission: f64,
    commission_per_unit: f64,
    leverage: f64,
    quantity_step: f64,
    financing_long: f64,
    financing_short: f64,
    debit_interest_rate: f64,
    maintenance_margin_fraction: f64,
    holding_ns: Option<i64>,
    stock: bool,
    allow_short: bool,
}

#[derive(Clone, Copy)]
struct Flags {
    record: bool,
    keep_equity: bool,
    record_exposure: bool,
}

struct Market<'a> {
    bid_open: &'a [f64],
    ask_open: &'a [f64],
    bid_close: &'a [f64],
    ask_close: &'a [f64],
    conv_bid_open: &'a [f64],
    conv_ask_open: &'a [f64],
    conv_bid_close: &'a [f64],
    conv_ask_close: &'a [f64],
    split: &'a [f64],
    dividend: &'a [f64],
    time_ns: &'a [i64],
    known_ns: &'a [i64],
    open_month: &'a [i64],
    final_month: usize,
    months: usize,
    rollover: Option<(&'a [f64], &'a [f64])>,
    allowed: Option<&'a [bool]>,
}

/// numpy.maximum: NaN in either argument propagates.
#[inline]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() { a } else { b }
}

/// numpy.minimum: NaN in either argument propagates.
#[inline]
fn np_minimum(a: f64, b: f64) -> f64 {
    if a <= b || a.is_nan() { a } else { b }
}

#[inline]
fn convert(amount: f64, bid: f64, ask: f64) -> f64 {
    amount * if amount >= 0.0 { bid } else { ask }
}

#[inline]
fn notional(price: f64, units: f64, conv_bid: f64, conv_ask: f64) -> f64 {
    units.abs() * price * ((conv_bid + conv_ask) / 2.0)
}

#[inline]
fn commission(costs: &Costs, units: f64, value: f64) -> f64 {
    if units != 0.0 {
        np_maximum(
            costs.minimum_commission,
            costs.commission_per_unit * units.abs(),
        ) + costs.fee_rate * value
    } else {
        0.0
    }
}

/// numpy.sign for the values a position can take.
#[inline]
fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        value
    }
}

struct TradeRecord {
    column: usize,
    entry_index: i64,
    exit_index: usize,
    final_close: bool,
    reason: u8,
    original_fill: f64,
    fill: f64,
    trade_return: f64,
    original_quantity: f64,
    exit_quantity: f64,
    pnl: f64,
    fees: f64,
    financing: f64,
    dividend_income: f64,
    unit_net_profit: f64,
    unit_entry_cost: f64,
}

/// One strategy's account; field names follow the Python reference.
struct Sleeve {
    cash: f64,
    quantity: f64,
    entry_fill: f64,
    entry_notional: f64,
    entry_capital: f64,
    original_quantity: f64,
    original_fill: f64,
    principal_flow: f64,
    entry_index: i64,
    entry_time: i64,
    position_fees: f64,
    position_carry: f64,
    position_income: f64,
    pending: u8,
    total_fees: f64,
    total_carry: f64,
    total_dividends: f64,
    count: i64,
    return_sum: f64,
    mean: f64,
    m2: f64,
    log_growth: f64,
    unit_profit: f64,
    first_unit_capital: f64,
}

impl Sleeve {
    fn new(cash: f64) -> Self {
        Sleeve {
            cash,
            quantity: 0.0,
            entry_fill: 0.0,
            entry_notional: 0.0,
            entry_capital: 1.0,
            original_quantity: 1.0,
            original_fill: 0.0,
            principal_flow: 0.0,
            entry_index: 0,
            entry_time: 0,
            position_fees: 0.0,
            position_carry: 0.0,
            position_income: 0.0,
            pending: 0,
            total_fees: 0.0,
            total_carry: 0.0,
            total_dividends: 0.0,
            count: 0,
            return_sum: 0.0,
            mean: 0.0,
            m2: 0.0,
            log_growth: 0.0,
            unit_profit: 0.0,
            first_unit_capital: 0.0,
        }
    }

    fn equity(&self, costs: &Costs, bid: f64, ask: f64, conv_bid: f64, conv_ask: f64) -> f64 {
        let mark = if self.quantity >= 0.0 { bid } else { ask };
        if costs.stock {
            self.cash + convert(self.quantity * mark, conv_bid, conv_ask)
        } else {
            self.cash + convert(self.quantity * (mark - self.entry_fill), conv_bid, conv_ask)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accrue(
        &mut self,
        costs: &Costs,
        rollover: bool,
        duration: i64,
        bid: f64,
        ask: f64,
        conv_bid: f64,
        conv_ask: f64,
    ) {
        if duration <= 0 || self.quantity == 0.0 {
            return;
        }
        let duration = duration as f64;
        let value = notional((bid + ask) / 2.0, self.quantity, conv_bid, conv_ask);
        let rate = if self.quantity > 0.0 {
            costs.financing_long
        } else {
            costs.financing_short
        };
        let mut charge = value * rate * duration / YEAR_NS;
        if rollover {
            charge = 0.0;
        }
        charge += np_maximum(-self.cash, 0.0) * costs.debit_interest_rate * duration / YEAR_NS;
        self.cash -= charge;
        self.position_carry += charge;
        self.total_carry += charge;
    }

    #[allow(clippy::too_many_arguments)]
    fn close(
        &mut self,
        costs: &Costs,
        quotes: (f64, f64, f64, f64),
        column: usize,
        index: usize,
        final_close: bool,
        reason: u8,
        month_sum: &mut f64,
        month_count: &mut i64,
        trades: Option<&mut Vec<TradeRecord>>,
    ) {
        let (bid, ask, conv_bid, conv_ask) = quotes;
        let quantity = self.quantity;
        let fill = if quantity > 0.0 {
            bid * (1.0 - costs.slip)
        } else {
            ask * (1.0 + costs.slip)
        };
        let value = notional(fill, quantity, conv_bid, conv_ask);
        let exit_fee = commission(costs, quantity, value);
        let pnl;
        if costs.stock {
            let proceeds = convert(quantity * fill, conv_bid, conv_ask);
            pnl = self.principal_flow + proceeds + self.position_income
                - self.position_carry
                - self.position_fees
                - exit_fee;
            self.cash += proceeds - exit_fee;
        } else {
            let gross = convert(quantity * (fill - self.entry_fill), conv_bid, conv_ask);
            pnl =
                gross + self.position_income - self.position_carry - self.position_fees - exit_fee;
            self.cash += gross - exit_fee;
        }
        self.total_fees += exit_fee;
        let trade_return = pnl / self.entry_capital;
        let first = self.count == 0;
        self.count += 1;
        let delta = trade_return - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (trade_return - self.mean);
        self.return_sum += trade_return;
        self.log_growth += np_maximum(trade_return, -1.0 + 1e-12).ln_1p();
        *month_sum += trade_return;
        *month_count += 1;
        self.unit_profit += pnl / self.original_quantity;
        if first {
            self.first_unit_capital = self.entry_capital / self.original_quantity;
        }
        if let Some(trades) = trades {
            trades.push(TradeRecord {
                column,
                entry_index: self.entry_index,
                exit_index: index,
                final_close,
                reason,
                original_fill: self.original_fill,
                fill,
                trade_return,
                original_quantity: self.original_quantity,
                exit_quantity: quantity.abs(),
                pnl,
                fees: self.position_fees + exit_fee,
                financing: self.position_carry,
                dividend_income: self.position_income,
                unit_net_profit: pnl / self.original_quantity,
                unit_entry_cost: self.entry_capital / self.original_quantity,
            });
        }
        self.quantity = 0.0;
        self.pending = 0;
    }
}

struct Batch {
    terminal: Vec<f64>,
    counts: Vec<i64>,
    return_sum: Vec<f64>,
    std: Vec<f64>,
    paper: Vec<f64>,
    monthly: Vec<f64>,
    monthly_counts: Vec<i64>,
    fees: Vec<f64>,
    financing: Vec<f64>,
    dividends: Vec<f64>,
    log_growth: Vec<f64>,
    equity: Option<Vec<f64>>,
    held: Option<Vec<bool>>,
    trades: Vec<TradeRecord>,
}

#[allow(clippy::too_many_arguments)]
fn simulate(
    market: &Market,
    costs: &Costs,
    flags: Flags,
    signals: &[i8],
    width: usize,
    column: usize,
    cash: f64,
    take_profit: f64,
    stop_loss: f64,
    month_sums: &mut [f64],
    month_counts: &mut [i64],
    equity: &mut [f64],
    held: &mut [bool],
    trades: &mut Vec<TradeRecord>,
) -> Sleeve {
    let n = market.time_ns.len();
    let rollover = market.rollover.is_some();
    let mut s = Sleeve::new(cash);
    if flags.keep_equity {
        equity[column] = s.cash;
    }
    for i in 0..n {
        if i > 0 {
            s.accrue(
                costs,
                rollover,
                market.time_ns[i] - market.known_ns[i - 1],
                market.bid_close[i - 1],
                market.ask_close[i - 1],
                market.conv_bid_close[i - 1],
                market.conv_ask_close[i - 1],
            );
        }
        let (conv_bid_open, conv_ask_open) = (market.conv_bid_open[i], market.conv_ask_open[i]);
        if costs.stock {
            s.quantity *= market.split[i];
            s.entry_fill /= market.split[i];
            let income = convert(
                s.quantity * market.dividend[i],
                conv_bid_open,
                conv_ask_open,
            );
            s.cash += income;
            s.position_income += income;
            s.total_dividends += income;
        }
        let signal = if i > 0 && market.known_ns[i - 1] <= market.time_ns[i] {
            signals[(i - 1) * width + column]
        } else {
            0
        };
        let holding = s.quantity != 0.0;
        let deadline = holding
            && costs
                .holding_ns
                .is_some_and(|limit| market.time_ns[i] >= s.entry_time.wrapping_add(limit));
        let exits = holding
            && (s.pending != 0
                || deadline
                || (s.quantity > 0.0 && signal == -1)
                || (s.quantity < 0.0 && signal == 1));
        if exits {
            let reason = if deadline {
                HOLDING_DEADLINE
            } else if s.pending != 0 {
                s.pending
            } else {
                SIGNAL
            };
            let month = market.open_month[i] as usize;
            s.close(
                costs,
                (
                    market.bid_open[i],
                    market.ask_open[i],
                    conv_bid_open,
                    conv_ask_open,
                ),
                column,
                i,
                false,
                reason,
                &mut month_sums[month],
                &mut month_counts[month],
                flags.record.then_some(&mut *trades),
            );
        }
        if market.allowed.is_none_or(|allowed| allowed[i]) {
            let direction: i8 = if signal == 1 {
                1
            } else if signal == -1 && costs.allow_short {
                -1
            } else {
                0
            };
            if s.quantity == 0.0 && !exits && direction != 0 && s.cash > 0.0 {
                let fill = if direction > 0 {
                    market.ask_open[i] * (1.0 + costs.slip)
                } else {
                    market.bid_open[i] * (1.0 - costs.slip)
                };
                let unit_value = if costs.stock {
                    fill * if direction > 0 {
                        conv_ask_open
                    } else {
                        conv_bid_open
                    }
                } else {
                    fill * ((conv_bid_open + conv_ask_open) / 2.0)
                };
                let denominator = unit_value / costs.leverage + costs.fee_rate * unit_value;
                let mut maximum = np_minimum(
                    np_maximum(s.cash - costs.minimum_commission, 0.0) / denominator,
                    s.cash / (denominator + costs.commission_per_unit),
                );
                if costs.quantity_step != 0.0 {
                    maximum = (maximum / costs.quantity_step + 1e-12).floor() * costs.quantity_step;
                }
                let new = f64::from(direction) * np_maximum(maximum, 0.0);
                if new != 0.0 {
                    let fee = commission(
                        costs,
                        new,
                        notional(fill, new, conv_bid_open, conv_ask_open),
                    );
                    s.quantity = new;
                    s.entry_fill = fill;
                    s.original_quantity = new.abs();
                    s.original_fill = fill;
                    s.entry_notional = new.abs() * unit_value;
                    s.entry_capital = s.entry_notional / costs.leverage + fee;
                    s.position_fees = fee;
                    s.position_carry = 0.0;
                    s.position_income = 0.0;
                    s.entry_index = i as i64;
                    s.entry_time = market.time_ns[i];
                    if costs.stock {
                        let flow = convert(-new * fill, conv_bid_open, conv_ask_open);
                        s.cash += flow;
                        s.principal_flow = flow;
                    }
                    s.cash -= fee;
                    s.total_fees += fee;
                }
            }
        }
        s.accrue(
            costs,
            rollover,
            market.known_ns[i] - market.time_ns[i],
            market.bid_open[i],
            market.ask_open[i],
            conv_bid_open,
            conv_ask_open,
        );
        let (conv_bid_close, conv_ask_close) = (market.conv_bid_close[i], market.conv_ask_close[i]);
        if let Some((long, short)) = market.rollover {
            let payment = s.quantity.abs() * if s.quantity >= 0.0 { long[i] } else { short[i] };
            let credit = convert(payment, conv_bid_close, conv_ask_close);
            s.cash += credit;
            s.position_carry -= credit;
            s.total_carry -= credit;
        }
        let close_quote = if s.quantity >= 0.0 {
            market.bid_close[i]
        } else {
            market.ask_close[i]
        };
        let movement = if s.entry_fill > 0.0 {
            (close_quote - s.entry_fill) / s.entry_fill
        } else {
            0.0
        } * sign(s.quantity);
        let holding = s.quantity != 0.0;
        s.pending = 0;
        if holding && take_profit.is_finite() && movement >= take_profit {
            s.pending = 1;
        }
        if holding && stop_loss.is_finite() && movement <= -stop_loss {
            s.pending = 2;
        }
        let last_equity = s.equity(
            costs,
            market.bid_close[i],
            market.ask_close[i],
            conv_bid_close,
            conv_ask_close,
        );
        let margin_applicable = !costs.stock || costs.leverage > 1.0 || s.quantity < 0.0;
        if holding
            && margin_applicable
            && last_equity <= s.entry_notional / costs.leverage * costs.maintenance_margin_fraction
        {
            s.pending = MARGIN_CALL;
        }
        if flags.keep_equity {
            equity[(i + 1) * width + column] = last_equity;
        }
        if flags.record_exposure {
            held[i * width + column] = s.quantity != 0.0;
        }
    }
    if s.quantity != 0.0 {
        let last = n - 1;
        s.close(
            costs,
            (
                market.bid_close[last],
                market.ask_close[last],
                market.conv_bid_close[last],
                market.conv_ask_close[last],
            ),
            column,
            last,
            true,
            SAMPLE_END,
            &mut month_sums[market.final_month],
            &mut month_counts[market.final_month],
            flags.record.then_some(&mut *trades),
        );
    }
    if flags.keep_equity {
        equity[n * width + column] = s.cash;
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn run(
    market: &Market,
    costs: &Costs,
    flags: Flags,
    signals: &[i8],
    width: usize,
    initial_cash: &[f64],
    take_profit: &[f64],
    stop_loss: &[f64],
) -> Batch {
    let n = market.time_ns.len();
    let months = market.months;
    let mut batch = Batch {
        terminal: Vec::with_capacity(width),
        counts: Vec::with_capacity(width),
        return_sum: Vec::with_capacity(width),
        std: Vec::with_capacity(width),
        paper: Vec::with_capacity(width),
        monthly: vec![0.0; width * months],
        monthly_counts: vec![0; width * months],
        fees: Vec::with_capacity(width),
        financing: Vec::with_capacity(width),
        dividends: Vec::with_capacity(width),
        log_growth: Vec::with_capacity(width),
        equity: flags.keep_equity.then(|| vec![0.0; (n + 1) * width]),
        held: flags.record_exposure.then(|| vec![false; n * width]),
        trades: Vec::new(),
    };
    let mut no_equity: [f64; 0] = [];
    let mut no_held: [bool; 0] = [];
    for column in 0..width {
        let row = column * months..(column + 1) * months;
        let s = simulate(
            market,
            costs,
            flags,
            signals,
            width,
            column,
            initial_cash[column],
            take_profit[column],
            stop_loss[column],
            &mut batch.monthly[row.clone()],
            &mut batch.monthly_counts[row.clone()],
            batch.equity.as_deref_mut().unwrap_or(&mut no_equity),
            batch.held.as_deref_mut().unwrap_or(&mut no_held),
            &mut batch.trades,
        );
        for (sum, &count) in batch.monthly[row.clone()]
            .iter_mut()
            .zip(&batch.monthly_counts[row])
        {
            *sum = if count > 0 { *sum / count as f64 } else { 0.0 };
        }
        let variance = if s.count > 0 {
            s.m2 / s.count as f64
        } else {
            0.0
        };
        batch.terminal.push(s.cash);
        batch.counts.push(s.count);
        batch.return_sum.push(s.return_sum);
        batch.std.push(np_maximum(variance, 0.0).sqrt());
        batch.paper.push(if s.first_unit_capital > 0.0 {
            100.0 * s.unit_profit / s.first_unit_capital
        } else {
            0.0
        });
        batch.fees.push(s.total_fees);
        batch.financing.push(s.total_carry);
        batch.dividends.push(s.total_dividends);
        batch.log_growth.push(s.log_growth);
    }
    // The Python ledger lists trades by exit bar, opens before the final close,
    // then by column.
    batch
        .trades
        .sort_by_key(|t| (t.exit_index, t.final_close, t.column));
    batch
}

fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}

/// Run the batch; see `gen_trade.market_execution._native_execute_batch`.
///
/// `quotes` rows: bid/ask open, bid/ask close, conversion bid/ask open,
/// conversion bid/ask close, split ratio, dividend. `clock` rows: bar open
/// time, availability time (ns), open-month index. `signals` is bars by
/// strategies.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (quotes, clock, months, final_month, rollover, allowed, signals, initial_cash,
                    take_profit, stop_loss, costs, record, record_equity, record_exposure))]
pub(crate) fn execute_batch_arrays<'py>(
    py: Python<'py>,
    quotes: PyReadonlyArray2<'py, f64>,
    clock: PyReadonlyArray2<'py, i64>,
    months: usize,
    final_month: usize,
    rollover: Option<PyReadonlyArray2<'py, f64>>,
    allowed: Option<PyReadonlyArray1<'py, bool>>,
    signals: PyReadonlyArray2<'py, i8>,
    initial_cash: PyReadonlyArray1<'py, f64>,
    take_profit: PyReadonlyArray1<'py, f64>,
    stop_loss: PyReadonlyArray1<'py, f64>,
    costs: Costs,
    record: bool,
    record_equity: bool,
    record_exposure: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let n = clock.shape()[1];
    let width = signals.shape()[1];
    if n == 0 || quotes.shape() != [QUOTE_ROWS, n] || clock.shape() != [CLOCK_ROWS, n] {
        return Err(invalid(
            "execution needs aligned, nonempty quote and clock rows",
        ));
    }
    if signals.shape()[0] != n
        || [&initial_cash, &take_profit, &stop_loss]
            .iter()
            .any(|values| values.len() != width)
    {
        return Err(invalid(
            "execution needs an aligned n-by-strategy decision matrix",
        ));
    }
    let quote_values = quotes.as_slice()?;
    let clock_values = clock.as_slice()?;
    let open_month = &clock_values[2 * n..];
    if final_month >= months
        || open_month
            .iter()
            .any(|&month| month < 0 || month as usize >= months)
    {
        return Err(invalid("month indices must lie inside the month range"));
    }
    let rollover_values = match &rollover {
        Some(values) if values.shape() == [2, n] => Some(values.as_slice()?),
        Some(_) => {
            return Err(invalid(
                "rollover needs long and short rows aligned with bars",
            ));
        }
        None => None,
    };
    let allowed_values = match &allowed {
        Some(values) if values.len() == n => Some(values.as_slice()?),
        Some(_) => return Err(invalid("entry-hour mask must align with bars")),
        None => None,
    };
    let row = |k: usize| &quote_values[k * n..(k + 1) * n];
    let market = Market {
        bid_open: row(0),
        ask_open: row(1),
        bid_close: row(2),
        ask_close: row(3),
        conv_bid_open: row(4),
        conv_ask_open: row(5),
        conv_bid_close: row(6),
        conv_ask_close: row(7),
        split: row(8),
        dividend: row(9),
        time_ns: &clock_values[..n],
        known_ns: &clock_values[n..2 * n],
        open_month,
        final_month,
        months,
        rollover: rollover_values.map(|values| (&values[..n], &values[n..])),
        allowed: allowed_values,
    };
    let flags = Flags {
        record,
        keep_equity: record || record_equity,
        record_exposure,
    };
    let signal_values = signals.as_slice()?;
    let (cash, tp, sl) = (
        initial_cash.as_slice()?,
        take_profit.as_slice()?,
        stop_loss.as_slice()?,
    );
    let batch = py.detach(|| run(&market, &costs, flags, signal_values, width, cash, tp, sl));

    let out = PyDict::new(py);
    out.set_item("terminal_equity", batch.terminal.into_pyarray(py))?;
    out.set_item("trade_counts", batch.counts.into_pyarray(py))?;
    out.set_item("trade_return_sum", batch.return_sum.into_pyarray(py))?;
    out.set_item("trade_return_std", batch.std.into_pyarray(py))?;
    out.set_item("paper_total_return", batch.paper.into_pyarray(py))?;
    out.set_item(
        "monthly_returns",
        batch.monthly.into_pyarray(py).reshape([width, months])?,
    )?;
    out.set_item(
        "monthly_counts",
        batch
            .monthly_counts
            .into_pyarray(py)
            .reshape([width, months])?,
    )?;
    out.set_item("fees", batch.fees.into_pyarray(py))?;
    out.set_item("financing", batch.financing.into_pyarray(py))?;
    out.set_item("dividends", batch.dividends.into_pyarray(py))?;
    out.set_item("trade_log_growth", batch.log_growth.into_pyarray(py))?;
    match batch.equity {
        Some(values) => out.set_item("equity", values.into_pyarray(py).reshape([n + 1, width])?)?,
        None => out.set_item("equity", py.None())?,
    }
    match batch.held {
        Some(values) => out.set_item("held", values.into_pyarray(py).reshape([n, width])?)?,
        None => out.set_item("held", py.None())?,
    }
    if record {
        let count = batch.trades.len();
        let mut ints = Vec::with_capacity(count * TRADE_INTS);
        let mut floats = Vec::with_capacity(count * TRADE_FLOATS);
        for t in &batch.trades {
            ints.extend_from_slice(&[
                t.column as i64,
                t.entry_index,
                t.exit_index as i64,
                i64::from(t.final_close),
                i64::from(t.reason),
            ]);
            floats.extend_from_slice(&[
                t.original_fill,
                t.fill,
                t.trade_return,
                t.original_quantity,
                t.exit_quantity,
                t.pnl,
                t.fees,
                t.financing,
                t.dividend_income,
                t.unit_net_profit,
                t.unit_entry_cost,
            ]);
        }
        out.set_item(
            "trade_ints",
            ints.into_pyarray(py).reshape([count, TRADE_INTS])?,
        )?;
        out.set_item(
            "trade_floats",
            floats.into_pyarray(py).reshape([count, TRADE_FLOATS])?,
        )?;
    }
    Ok(out)
}
