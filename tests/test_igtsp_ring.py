import itertools
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gen_trade.backtest import ExecutionConfig, PortfolioSignals, run_backtest, run_portfolio_backtest
from gen_trade.methods.igtsp_ring import (
    DEFAULT_BANK, FIDELITY, INDICATORS, PAPER_SETTINGS, SOURCE_PDF_SHA256,
    GroupChromosome, IGTSPRing, _Island, crossover, decode_bits, elite_selection,
    invert, mutate, portfolio_fitness, random_chromosome, ring_migrate, strategy_signals,
)
from gen_trade.methods.ring_fitness import PreparedMarket, monthly_trade_profiles
from gen_trade.runner import synthetic_markets


def chromosome(labels=(0, 0, 1), weights=((0, 1), (1, 1)), tp=(1, 0), sl=(1, 1)):
    return GroupChromosome(labels, weights, tp, sl)


def test_figure4_binary_tpsl_example_and_capital_allocation_without_cash_gene():
    c = chromosome((0, 0, 0, 1, 1), ((0, 1, 1, 1), (0, 0, 1, 1)))
    tp, sl = c.thresholds(.1, .1)
    assert tp == pytest.approx(.1 * 2/3)
    assert sl == .1
    np.testing.assert_allclose(c.group_weights(), [.7, .3])
    np.testing.assert_allclose(c.allocations(), [.7/3, .7/3, .7/3, .15, .15])
    assert decode_bits((1, 0)) == 2
    zero = chromosome(weights=((0, 0), (0, 0)), tp=(0, 0), sl=(0, 0))
    np.testing.assert_allclose(zero.group_weights(), [.5, .5])
    assert zero.thresholds() == (0., 0.)


@pytest.mark.parametrize('labels,weights,tp,sl', [
    ((0, 0, 0), ((0, 1), (1, 1)), (1, 0), (1, 1)),
    ((0, 1, 2), ((0, 1), (1, 1)), (1, 0), (1, 1)),
    ((0, 0, 1), ((0, 2), (1, 1)), (1, 0), (1, 1)),
    ((0, 0, 1), ((1,), (1, 1)), (1, 0), (1, 1)),
    ((0, 0, 1), ((0, 1), (1, 1)), (), (1, 1)),
    ((True, 0, 1), ((0, 1), (1, 1)), (1, 0), (1, 1)),
])
def test_chromosome_enforces_nonempty_partition_and_binary_parts(labels, weights, tp, sl):
    with pytest.raises(ValueError):
        GroupChromosome(labels, weights, tp, sl)


def test_all_operators_preserve_four_part_constraints_and_change_each_part():
    rng = np.random.default_rng(71)
    first = random_chromosome(rng, 34, 17)
    changed = set()
    for _ in range(150):
        second = random_chromosome(rng, 34, 17)
        for child in (crossover(first, second, rng), mutate(first, rng), invert(first, rng)):
            assert sorted(sum(child.groups, ())) == list(range(34))
            assert set(child.assignments) == set(range(17))
            assert len(child.take_profit_bits) == len(child.stop_loss_bits) == 8
            assert all(len(bits) == 8 and set(bits) <= {0, 1} for bits in child.weight_bits)
            assert child.allocations().sum() == pytest.approx(1)
            for part in ('assignments', 'weight_bits', 'take_profit_bits', 'stop_loss_bits'):
                if getattr(child, part) != getattr(first, part):
                    changed.add(part)
    assert changed == {'assignments', 'weight_bits', 'take_profit_bits', 'stop_loss_bits'}


def test_one_point_crossover_on_binary_parts_and_inversion_on_whole_groups():
    first = chromosome((0, 0, 1, 1), ((0, 0), (0, 0)), (0, 0), (0, 0))
    second = chromosome((0, 1, 0, 1), ((1, 1), (1, 1)), (1, 1), (1, 1))
    child = crossover(first, second, np.random.default_rng(5))
    for bits in (child.take_profit_bits + child.stop_loss_bits, sum(child.weight_bits, ())):
        assert bits[0] == 0 and bits[-1] == 1
        assert list(bits) == sorted(bits)  # precisely one 0 -> 1 splice
    inverted = invert(first, np.random.default_rng(3))
    assert inverted.groups == first.groups[::-1]
    assert inverted.weight_bits == first.weight_bits
    assert inverted.take_profit_bits == first.take_profit_bits


def test_eq6_to_16_hand_calculation_uses_component_risk_and_raw_entropy():
    c = chromosome()
    paths = np.array([[1, 1.2, .9], [1, 1.1, 1.2], [1, 1.3, 1.]])
    result = portfolio_fitness(c, paths, np.array([1, 1, 1]))
    gb = -(2/3 * np.log(2/3) + 1/3 * np.log(1/3))
    assert result['paper_portfolio_return'] == pytest.approx(.0125)
    assert result['paper_portfolio_risk'] == pytest.approx(.2625)
    assert result['group_balance'] == pytest.approx(gb)
    assert result['fitness'] == pytest.approx(.0125/.2625 + gb)
    # Mixture drawdown differs from Eq.13's weighted component drawdowns.
    mixed = c.allocations() @ paths
    assert np.max(np.maximum.accumulate(mixed) - mixed) == pytest.approx(.25)
    assert result['paper_portfolio_risk'] != pytest.approx(.25)


def test_closed_form_is_exact_average_of_all_one_member_per_group_portfolios():
    rng = np.random.default_rng(12)
    c = random_chromosome(rng, 8, 3)
    paths = np.column_stack((np.ones(8), np.exp(np.cumsum(rng.normal(.01,.1,(8,5)),axis=1))))
    risks = np.max(np.maximum.accumulate(paths,axis=1)-paths,axis=1)
    returns, losses = [], []
    weights = c.group_weights()
    for portfolio in itertools.product(*c.groups):
        returns.append(sum(weights[g]*(paths[i,-1]-1) for g,i in enumerate(portfolio)))
        losses.append(sum(weights[g]*risks[i] for g,i in enumerate(portfolio)))
    result = portfolio_fitness(c,paths,np.ones(8,dtype=int))
    assert result['paper_portfolio_return'] == pytest.approx(np.mean(returns))
    assert result['paper_portfolio_risk'] == pytest.approx(np.mean(losses))


def test_fitness_degeneracy_is_finite_explicit_and_no_trade_candidates_cannot_win():
    c = chromosome()
    positive = np.array([[1.,1.1], [1.,1.2], [1.,1.3]])
    result = portfolio_fitness(c, positive, np.ones(3,dtype=int))
    assert result['eligible'] and result['risk_denominator_floored'] and result['ratio_capped']
    assert np.isfinite(result['fitness'])
    assert portfolio_fitness(c, positive, np.zeros(3))['eligible'] is False
    assert portfolio_fitness(c, np.ones((3,2)), np.ones(3))['eligible'] is False
    with pytest.raises(ValueError):
        portfolio_fitness(c, positive * 100, np.ones(3))


def bars_at(timestamps, prices):
    t = pd.to_datetime(timestamps, utc=True)
    values = np.asarray(prices,dtype=float)
    return pd.DataFrame({'timestamp':t, 'available_at':t+pd.Timedelta(hours=1),
                         'open':values, 'high':values*1.01, 'low':values*.99,
                         'close':values, 'volume':100.})


def test_eq9_to_11_monthly_trade_average_then_compounding_and_empty_month():
    bars = bars_at(['2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05',
                    '2024-02-01','2024-03-01','2024-03-02','2024-03-03'],
                   [100,100,110,100,130,100,100,90,90])
    signals = np.array([1,-1,1,-1,0,1,-1,0,0],dtype=np.int8)[:,None]
    profile = monthly_trade_profiles(PreparedMarket.from_bars(bars),signals,ExecutionConfig(fee=0))
    assert profile.months == ('2024-01','2024-02','2024-03')
    np.testing.assert_array_equal(profile.monthly_trade_counts,[[2,0,1]])
    np.testing.assert_allclose(profile.mean_returns,[[.2,0,-.1]],atol=1e-14)
    np.testing.assert_allclose(profile.wealth,[[1,1.2,1.2,1.08]])
    assert profile.wealth[0,-1] != pytest.approx(1.1*1.3*.9)


@pytest.mark.parametrize('config', [
    ExecutionConfig(fee=0), ExecutionConfig(fee=.003,slippage_bps=7),
    ExecutionConfig(fee=.001,take_profit=0,stop_loss=0),
    ExecutionConfig(fee=.002,take_profit=.025,stop_loss=.015),
    ExecutionConfig(fee=.003,take_profit=.1,stop_loss=.03,max_holding_days=2,entry_hours=(9,)),
    ExecutionConfig(fee=.003,entry_hours=(12,)),
])
def test_batch_profiles_match_shared_next_open_simulator_including_costs_and_exits(config):
    bars = synthetic_markets(120,seed=7)['BTC/USDT']
    signals = np.random.default_rng(6).choice([-1,0,1],size=(120,5)).astype(np.int8)
    profile = monthly_trade_profiles(PreparedMarket.from_bars(bars),signals,config)
    for i in range(5):
        result = run_backtest(bars,signals[:,i],config)
        sums, counts = np.zeros(len(profile.months)), np.zeros(len(profile.months),dtype=int)
        for trade in result.trades:
            month = profile.months.index(trade.exit_time.strftime('%Y-%m'))
            sums[month] += trade.return_fraction
            counts[month] += 1
        means = np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
        np.testing.assert_array_equal(profile.monthly_trade_counts[i],counts)
        np.testing.assert_allclose(profile.mean_returns[i],means,atol=1e-14)
        np.testing.assert_allclose(profile.wealth[i],np.r_[1,np.cumprod(1+means)],atol=1e-14)
        assert profile.trade_counts[i] == len(result.trades)


def test_terminal_liquidation_is_attributed_to_availability_month():
    bars = bars_at(['2024-01-30 09:00','2024-01-31 09:00'],[100,110])
    bars.loc[1,'available_at'] = pd.Timestamp('2024-02-01',tz='UTC')
    profile = monthly_trade_profiles(PreparedMarket.from_bars(bars),np.array([[1],[0]]),ExecutionConfig(fee=.001))
    assert profile.months == ('2024-01','2024-02')
    np.testing.assert_array_equal(profile.monthly_trade_counts,[[0,1]])


def test_portfolio_accounting_charges_costs_and_carries_evolved_exit_policy():
    bars = synthetic_markets(100)['BTC/USDT'].iloc[:25]
    base = ExecutionConfig(fee=.003,slippage_bps=4)
    a = np.ones(25,dtype=np.int8)
    b = np.where(np.arange(25)%4==0,1,np.where(np.arange(25)%4==2,-1,0))
    plan = PortfolioSignals(('a','b'),np.column_stack((a,b)),(.2,.8),take_profit=.02,stop_loss=.01)
    result = run_portfolio_backtest(bars,plan,base)
    config = replace(base,take_profit=.02,stop_loss=.01)
    expected = .2*run_backtest(bars,a,config).equity + .8*run_backtest(bars,b,config).equity
    np.testing.assert_allclose(result.equity,expected)
    assert result.config.stop_loss == .01 and result.config.take_profit == .02
    assert {t.strategy_id for t in result.trades} == {'a','b'}
    assert all(t.allocation_fraction == (.2 if t.strategy_id=='a' else .8) for t in result.trades)
    with pytest.raises(ValueError,match='sum to one'):
        PortfolioSignals(('a','b'),plan.signals,(.8,.8))


def test_21_indicator_cartesian_bank_is_causal_and_missing_volume_does_not_backfill():
    bars = synthetic_markets(120)['BTC/USDT']
    assert len(INDICATORS) == 21 and len(DEFAULT_BANK) == 42*42
    signals = strategy_signals(bars)
    assert signals.shape == (120,len(DEFAULT_BANK))
    for size in (1,19,60,91):
        np.testing.assert_array_equal(signals[:size],strategy_signals(bars.iloc[:size]))
    changed = bars.copy()
    changed.loc[80:,['open','high','low','close']] *= 7
    np.testing.assert_array_equal(signals[:80],strategy_signals(changed)[:80])
    missing = bars.copy()
    missing['volume'] = np.nan
    assert strategy_signals(missing).shape == signals.shape


def test_elite_selection_and_paper_algorithm1_sequential_best_for_best_exchange():
    rng = np.random.default_rng(19)
    values = [random_chromosome(rng,8,3) for _ in range(6)]
    islands = [_Island(values[i:i+2],rng) for i in range(0,6,2)]
    all_scores = {c:{'fitness':float(i+1)} for i,c in enumerate(values)}
    for island in islands:
        island.cache.update(all_scores)
    assert elite_selection(islands[0],.5) == [values[1]]
    before = sorted(sum([i.population for i in islands],[]),key=lambda c:all_scores[c]['fitness'])
    events = ring_migrate(islands,count=1)
    # Swap best 2/4, then best 3/6, then best 5/4 (wrap). No cloning or worst replacement.
    assert [(e['left_island'],e['right_island']) for e in events] == [(0,1),(1,2),(2,0)]
    assert {all_scores[c]['fitness'] for c in islands[0].population} == {1.,5.}
    assert {all_scores[c]['fitness'] for c in islands[1].population} == {2.,6.}
    assert {all_scores[c]['fitness'] for c in islands[2].population} == {3.,4.}
    after = sorted(sum([i.population for i in islands],[]),key=lambda c:all_scores[c]['fitness'])
    assert before == after


def test_ring_reproducibility_frozen_bank_and_tpsl_and_total_population():
    bars = synthetic_markets(120)['BTC/USDT']
    options = dict(population_size=17,generations=3,seed=13,island_count=4,migration_interval=1)
    first, second = IGTSPRing(**options).fit(bars.iloc[:80]), IGTSPRing(**options).fit(bars.iloc[:80])
    assert first.parameters() == second.parameters()
    assert first.diagnostics == second.diagnostics
    assert sum(first.diagnostics['island_population_sizes']) == 17
    assert len(first.diagnostics['migration_events']) == 12
    prediction = first.predict(bars)
    np.testing.assert_array_equal(prediction.signals[:70],first.predict(bars.iloc[:70]).signals)
    saved = first.parameters()
    changed = bars.copy()
    changed.loc[90:,['open','high','low','close']] *= 5
    first.predict(changed)
    first.evaluate_paper(changed,start=80)
    assert first.parameters() == saved
    assert saved['paper_fitness_verified'] is True and saved['fidelity'] == FIDELITY
    assert saved['published_numerical_results_reproduced'] is False
    assert first.diagnostics['bank_selection']['training_only'] is True
    assert len(saved['bank']) == 34
    global_best = [generation['best_fitness'] for generation in first.diagnostics['global_best_history']]
    assert all(b >= a for a,b in zip(global_best,global_best[1:]))
    json.dumps(first.parameters(),allow_nan=False)
    json.dumps(first.diagnostics,allow_nan=False)


def test_serial_and_process_islands_have_identical_equations_and_search_results():
    bars = synthetic_markets(100)['BTC/USDT'].iloc[:65]
    options = dict(population_size=12,generations=2,seed=27,island_count=3,migration_interval=1)
    serial = IGTSPRing(**options).fit(bars)
    parallel = IGTSPRing(**options,island_workers=2).fit(bars)
    assert serial.diagnostics == parallel.diagnostics
    assert serial.parameters()['chromosome'] == parallel.parameters()['chromosome']


def test_table5_defaults_and_full_text_provenance():
    model = IGTSPRing()
    for key, value in PAPER_SETTINGS.items():
        actual = model._settings[key] if key in model._settings else getattr(model,key)
        assert actual == value
    assert SOURCE_PDF_SHA256 == '7ccf3ab38a5609ffc5b89e72521587ecb6d9b7b42bdf18b2fa299765b8b31d42'


@pytest.mark.parametrize('settings',[{'island_count':1},{'population_size':7},{'groups':35},
    {'migrants':40},{'migration_interval':0},{'migration_rate':1.1},{'mutation_probability':1.1},
    {'island_workers':0},{'selection_rate':0},{'strategy_count':16},{'take_profit_bits':0}])
def test_configuration_validation(settings):
    with pytest.raises(ValueError):
        IGTSPRing(**settings)


def test_unfitted_no_trade_and_unsupported_features_fail_clearly():
    bars = synthetic_markets(100)['BTC/USDT']
    model = IGTSPRing(population_size=8,generations=0,island_count=4)
    with pytest.raises(RuntimeError):
        model.predict(bars)
    with pytest.raises(ValueError,match='external'):
        model.fit(bars,{'x':np.ones(100)})
    constant = bars.copy()
    constant[['open','high','low','close']] = 100.
    with pytest.raises(ValueError,match='two completed'):
        model.fit(constant)
    with pytest.raises(RuntimeError):
        model.parameters()
