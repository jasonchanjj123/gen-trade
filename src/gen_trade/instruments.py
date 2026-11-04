"""Explicit stock/FX contract and cost assumptions, all amounts in account currency."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    asset_class: str
    base_currency: str = 'USD'
    quote_currency: str = 'USD'
    account_currency: str = 'USD'
    quantity_step: float = 0.0
    leverage: float = 1.0
    allow_short: bool = False
    spread_bps: float = 0.0
    commission_bps: float = 0.0
    commission_per_unit: float = 0.0
    minimum_commission: float = 0.0
    financing_long: float = 0.0
    financing_short: float = 0.0
    debit_interest_rate: float = 0.0
    maintenance_margin_fraction: float = 0.5
    financing_source: str = 'declared annualized scenario; not historical broker swap rates'
    minimum_spread: float = 0.0

    def __post_init__(self):
        if not self.symbol or self.asset_class not in ('stock','fx'):
            raise ValueError('Instrument needs a symbol and stock/fx asset class')
        if any(len(c)!=3 or not c.isalpha() for c in (self.base_currency,self.quote_currency,self.account_currency)):
            raise ValueError('Currencies must be three-letter codes')
        for name in ('quantity_step','spread_bps','commission_bps','commission_per_unit','minimum_commission',
                     'financing_long','financing_short','debit_interest_rate','minimum_spread'):
            value = getattr(self,name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not np.isfinite(self.leverage) or self.leverage < 1:
            raise ValueError('leverage must be finite and >=1')
        if not 0 < self.maintenance_margin_fraction <= 1:
            raise ValueError('maintenance_margin_fraction must be in (0,1]')
        if self.spread_bps >= 10000 or self.commission_bps >= 10000:
            raise ValueError('Spread and commission bps must be below 10000')

    @property
    def scale_invariant(self):
        return self.quantity_step == 0 and self.minimum_commission == 0


FX_FINANCING_STRESS_FLOOR = 0.02


def benchmark_spec(symbol: str, *, stress=False, fx_financing: float = .02, stock_minimum_spread: float = 0.0) -> InstrumentSpec:
    """Benchmark contract/cost scenario.

    Original protocol: ``fx_financing=0.02`` (2%/yr debit on long and short FX
    notional, doubled under stress) and a 2 bp proportional stock spread.
    Audited protocol (audit items 6 and 13): ``fx_financing=0`` in the primary
    scenario, with the original 2% debit retained as the stress scenario,
    because a symmetric debit ignores the sign of the actual carry; and
    ``stock_minimum_spread=0.01`` (one cent full spread, doubled under stress),
    which binds for low-priced shares where 2 bp is below one tick.
    """
    multiplier = 2 if stress else 1
    if '/' not in symbol:
        return InstrumentSpec(symbol,'stock',spread_bps=2*multiplier,commission_per_unit=.005*multiplier,
                              financing_short=.03,debit_interest_rate=.05,
                              minimum_spread=stock_minimum_spread*multiplier)
    base, quote = symbol.split('/')
    financing = fx_financing*multiplier if fx_financing > 0 else (FX_FINANCING_STRESS_FLOOR if stress else 0.0)
    source = ('declared annualized scenario; not historical broker swap rates' if fx_financing > 0 else
              'primary: zero carry (sign of historical carry not modeled); stress: 2%/yr debit on held notional')
    return InstrumentSpec(symbol,'fx',base_currency=base,quote_currency=quote,
                          commission_bps=.35*multiplier,financing_long=financing,
                          financing_short=financing,financing_source=source)
