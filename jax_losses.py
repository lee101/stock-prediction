from __future__ import annotations

from typing import Any

from flax import struct
import jax
import jax.numpy as jnp


DEFAULT_MAKER_FEE_RATE = 0.001
HOURLY_PERIODS_PER_YEAR = 24 * 365
_EPS = 1e-8


@struct.dataclass
class HourlySimulationResult:
    pnl: jax.Array
    returns: jax.Array
    portfolio_values: jax.Array
    cash: jax.Array
    inventory: jax.Array
    buy_fill_probability: jax.Array
    sell_fill_probability: jax.Array
    executed_buys: jax.Array
    executed_sells: jax.Array
    inventory_path: jax.Array


def _to_array(value: jax.Array | float, reference: jax.Array) -> jax.Array:
    return jnp.asarray(value, dtype=reference.dtype)


def _broadcast_batch_control(
    value: jax.Array | float | bool,
    reference: jax.Array,
    batch_shape: tuple[int, ...],
) -> jax.Array:
    array = _to_array(value, reference)
    if array.ndim == 0:
        return jnp.broadcast_to(array, batch_shape)
    if array.shape == reference.shape:
        array = array[..., 0]
    return jnp.broadcast_to(array, batch_shape)


def _safe_denominator(value: jax.Array) -> jax.Array:
    return jnp.clip(value, min=_EPS)


def _check_shapes(reference: jax.Array, *others: jax.Array) -> None:
    for tensor in others:
        if reference.shape != tensor.shape:
            raise ValueError("All tensors must share the same shape for simulation")


def approx_buy_fill_probability(
    buy_price: jax.Array,
    low_price: jax.Array,
    close_price: jax.Array,
    *,
    temperature: float = 5e-4,
) -> jax.Array:
    _check_shapes(buy_price, low_price, close_price)
    scale = jnp.clip(jnp.abs(close_price), min=1e-4)
    score = (buy_price - low_price) / _safe_denominator(scale * _to_array(temperature, close_price))
    return jax.nn.sigmoid(score)


def approx_sell_fill_probability(
    sell_price: jax.Array,
    high_price: jax.Array,
    close_price: jax.Array,
    *,
    temperature: float = 5e-4,
) -> jax.Array:
    _check_shapes(sell_price, high_price, close_price)
    scale = jnp.clip(jnp.abs(close_price), min=1e-4)
    score = (high_price - sell_price) / _safe_denominator(scale * _to_array(temperature, close_price))
    return jax.nn.sigmoid(score)


def _prepare_lagged(
    *,
    highs: jax.Array,
    lows: jax.Array,
    closes: jax.Array,
    opens: jax.Array | None,
    buy_prices: jax.Array,
    sell_prices: jax.Array,
    trade_intensity: jax.Array,
    buy_trade_intensity: jax.Array,
    sell_trade_intensity: jax.Array,
    max_leverage: jax.Array | float,
    decision_lag_bars: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    original_steps = closes.shape[-1]
    if decision_lag_bars <= 0:
        leverage_array = _to_array(max_leverage, closes)
        if leverage_array.ndim == 0:
            leverage_array = jnp.broadcast_to(leverage_array, closes.shape)
        return (
            highs,
            lows,
            closes,
            opens,
            buy_prices,
            sell_prices,
            trade_intensity,
            buy_trade_intensity,
            sell_trade_intensity,
            leverage_array,
        )

    lag = int(decision_lag_bars)
    highs = highs[..., lag:]
    lows = lows[..., lag:]
    closes = closes[..., lag:]
    if opens is not None:
        opens = opens[..., lag:]
    buy_prices = buy_prices[..., :-lag]
    sell_prices = sell_prices[..., :-lag]
    trade_intensity = trade_intensity[..., :-lag]
    buy_trade_intensity = buy_trade_intensity[..., :-lag]
    sell_trade_intensity = sell_trade_intensity[..., :-lag]

    leverage_array = _to_array(max_leverage, closes)
    if leverage_array.ndim == 0:
        leverage_array = jnp.broadcast_to(leverage_array, closes.shape)
    elif leverage_array.shape[-1] == original_steps:
        leverage_array = leverage_array[..., lag:]
    else:
        leverage_array = jnp.broadcast_to(leverage_array, closes.shape)

    return (
        highs,
        lows,
        closes,
        opens,
        buy_prices,
        sell_prices,
        trade_intensity,
        buy_trade_intensity,
        sell_trade_intensity,
        leverage_array,
    )


def _simulate_core(
    *,
    highs: jax.Array,
    lows: jax.Array,
    closes: jax.Array,
    opens: jax.Array | None,
    buy_prices: jax.Array,
    sell_prices: jax.Array,
    trade_intensity: jax.Array,
    buy_trade_intensity: jax.Array,
    sell_trade_intensity: jax.Array,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    temperature: float = 5e-4,
    max_leverage: float | jax.Array = 1.0,
    can_short: bool | float | jax.Array = False,
    can_long: bool | float | jax.Array = True,
    decision_lag_bars: int = 0,
    market_order_entry: bool = False,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
    probabilistic: bool,
) -> HourlySimulationResult:
    _check_shapes(highs, lows, closes, buy_prices, sell_prices, trade_intensity)
    _check_shapes(highs, buy_trade_intensity, sell_trade_intensity)
    if opens is not None:
        _check_shapes(highs, opens)
    if closes.ndim == 0:
        raise ValueError("Input tensors must include a time dimension")

    (
        highs,
        lows,
        closes,
        opens,
        buy_prices,
        sell_prices,
        trade_intensity,
        buy_trade_intensity,
        sell_trade_intensity,
        leverage_array,
    ) = _prepare_lagged(
        highs=highs,
        lows=lows,
        closes=closes,
        opens=opens,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=trade_intensity,
        buy_trade_intensity=buy_trade_intensity,
        sell_trade_intensity=sell_trade_intensity,
        max_leverage=max_leverage,
        decision_lag_bars=decision_lag_bars,
    )

    batch_shape = closes.shape[:-1]
    fee = jnp.asarray(maker_fee, dtype=closes.dtype)
    fee_buy = 1.0 + fee
    fee_sell = 1.0 - fee
    margin_cost_per_step = float(margin_annual_rate) / HOURLY_PERIODS_PER_YEAR

    can_short_array = _broadcast_batch_control(can_short, closes, batch_shape)
    can_long_array = _broadcast_batch_control(can_long, closes, batch_shape)
    cash0 = jnp.full(batch_shape, initial_cash, dtype=closes.dtype)
    inventory0 = jnp.full(batch_shape, initial_inventory, dtype=closes.dtype)
    prev_value0 = cash0 + inventory0 * closes[..., 0]

    if opens is None:
        opens = closes

    def _step(carry: tuple[jax.Array, jax.Array, jax.Array], step_inputs: tuple[jax.Array, ...]):
        cash, inventory, prev_value = carry
        close, high, low, open_price, b_price, s_price, b_intensity, s_intensity, step_limit = step_inputs

        b_price = jnp.clip(b_price, min=_EPS)
        s_price = jnp.clip(s_price, min=_EPS)
        b_price = jnp.where(market_order_entry, jnp.clip(open_price, min=_EPS), b_price)

        step_limit = jnp.clip(step_limit, min=_EPS)
        b_intensity = jnp.minimum(jnp.clip(b_intensity, min=0.0), step_limit)
        s_intensity = jnp.minimum(jnp.clip(s_intensity, min=0.0), step_limit)
        b_frac_limit = b_intensity / jnp.clip(step_limit, min=_EPS)
        s_frac_limit = s_intensity / jnp.clip(step_limit, min=_EPS)

        buy_threshold = jnp.where(fill_buffer_pct > 0.0, b_price * (1.0 - fill_buffer_pct), b_price)
        sell_threshold = jnp.where(fill_buffer_pct > 0.0, s_price * (1.0 + fill_buffer_pct), s_price)
        if probabilistic:
            buy_fill = jnp.where(
                market_order_entry,
                jnp.ones_like(b_price),
                approx_buy_fill_probability(buy_threshold, low, close, temperature=temperature),
            )
            sell_fill = approx_sell_fill_probability(sell_threshold, high, close, temperature=temperature)
        else:
            buy_fill = jnp.where(
                market_order_entry,
                (b_intensity > 0).astype(closes.dtype),
                ((low <= buy_threshold) & (b_intensity > 0)).astype(closes.dtype),
            )
            sell_fill = ((high >= sell_threshold) & (s_intensity > 0)).astype(closes.dtype)

        equity = cash + inventory * close
        max_buy_cash = jnp.where(
            b_price > 0,
            cash / _safe_denominator(b_price * fee_buy),
            0.0,
        )
        target_notional = step_limit * jnp.clip(equity, min=_EPS)
        current_notional = inventory * b_price
        leveraged_capacity = jnp.where(
            b_price > 0,
            jnp.clip(target_notional - current_notional, min=0.0) / _safe_denominator(b_price * fee_buy),
            0.0,
        )
        buy_capacity = jnp.where(step_limit <= 1.0 + 1e-6, jnp.clip(max_buy_cash, min=0.0), leveraged_capacity)
        buy_qty = b_frac_limit * buy_capacity * buy_fill

        cover_only_cap = jnp.clip(-inventory, min=0.0)
        buy_qty = jnp.where(
            can_long_array > 0.5,
            buy_qty,
            jnp.minimum(buy_qty, cover_only_cap),
        )

        long_to_close = jnp.clip(inventory, min=0.0)
        max_short_qty = jnp.where(
            s_price > 0,
            (step_limit * jnp.clip(equity, min=_EPS)) / _safe_denominator(s_price * fee_buy),
            0.0,
        )
        current_short_qty = jnp.clip(-inventory, min=0.0)
        short_open_cap = jnp.clip(max_short_qty - current_short_qty, min=0.0)
        sell_capacity = long_to_close + jnp.where(can_short_array > 0.5, short_open_cap, 0.0)
        sell_qty = s_frac_limit * sell_capacity * sell_fill

        executed_buys = buy_qty
        executed_sells = sell_qty

        cash = cash - executed_buys * b_price * fee_buy + executed_sells * s_price * fee_sell
        inventory = inventory + executed_buys - executed_sells

        if margin_cost_per_step > 0.0:
            pos_value = jnp.abs(inventory * close)
            eq = cash + inventory * close
            margin_used = jnp.clip(pos_value - jnp.clip(eq, min=0.0), min=0.0)
            cash = cash - margin_used * margin_cost_per_step

        portfolio_value = cash + inventory * close
        pnl = portfolio_value - prev_value
        returns = pnl / _safe_denominator(prev_value)

        outputs = (
            pnl,
            returns,
            portfolio_value,
            buy_fill,
            sell_fill,
            executed_buys,
            executed_sells,
            inventory,
        )
        return (cash, inventory, portfolio_value), outputs

    scan_inputs = (
        jnp.moveaxis(closes, -1, 0),
        jnp.moveaxis(highs, -1, 0),
        jnp.moveaxis(lows, -1, 0),
        jnp.moveaxis(opens, -1, 0),
        jnp.moveaxis(buy_prices, -1, 0),
        jnp.moveaxis(sell_prices, -1, 0),
        jnp.moveaxis(buy_trade_intensity, -1, 0),
        jnp.moveaxis(sell_trade_intensity, -1, 0),
        jnp.moveaxis(leverage_array, -1, 0),
    )
    (_, final_inventory, final_value), outputs = jax.lax.scan(_step, (cash0, inventory0, prev_value0), scan_inputs)
    pnl, returns, values, buy_fill, sell_fill, exec_buys, exec_sells, inventory_path = outputs

    return HourlySimulationResult(
        pnl=jnp.moveaxis(pnl, 0, -1),
        returns=jnp.moveaxis(returns, 0, -1),
        portfolio_values=jnp.moveaxis(values, 0, -1),
        cash=final_value - final_inventory * closes[..., -1],
        inventory=final_inventory,
        buy_fill_probability=jnp.moveaxis(buy_fill, 0, -1),
        sell_fill_probability=jnp.moveaxis(sell_fill, 0, -1),
        executed_buys=jnp.moveaxis(exec_buys, 0, -1),
        executed_sells=jnp.moveaxis(exec_sells, 0, -1),
        inventory_path=jnp.moveaxis(inventory_path, 0, -1),
    )


def simulate_hourly_trades(
    *,
    highs: jax.Array,
    lows: jax.Array,
    closes: jax.Array,
    opens: jax.Array | None = None,
    buy_prices: jax.Array,
    sell_prices: jax.Array,
    trade_intensity: jax.Array,
    buy_trade_intensity: jax.Array | None = None,
    sell_trade_intensity: jax.Array | None = None,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    temperature: float = 5e-4,
    max_leverage: float | jax.Array = 1.0,
    can_short: bool | float | jax.Array = False,
    can_long: bool | float | jax.Array = True,
    decision_lag_bars: int = 0,
    market_order_entry: bool = False,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
) -> HourlySimulationResult:
    buy_trade_intensity = trade_intensity if buy_trade_intensity is None else buy_trade_intensity
    sell_trade_intensity = trade_intensity if sell_trade_intensity is None else sell_trade_intensity
    return _simulate_core(
        highs=highs,
        lows=lows,
        closes=closes,
        opens=opens,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=trade_intensity,
        buy_trade_intensity=buy_trade_intensity,
        sell_trade_intensity=sell_trade_intensity,
        maker_fee=maker_fee,
        initial_cash=initial_cash,
        initial_inventory=initial_inventory,
        temperature=temperature,
        max_leverage=max_leverage,
        can_short=can_short,
        can_long=can_long,
        decision_lag_bars=decision_lag_bars,
        market_order_entry=market_order_entry,
        fill_buffer_pct=fill_buffer_pct,
        margin_annual_rate=margin_annual_rate,
        probabilistic=True,
    )


def simulate_hourly_trades_binary(
    *,
    highs: jax.Array,
    lows: jax.Array,
    closes: jax.Array,
    opens: jax.Array | None = None,
    buy_prices: jax.Array,
    sell_prices: jax.Array,
    trade_intensity: jax.Array,
    buy_trade_intensity: jax.Array | None = None,
    sell_trade_intensity: jax.Array | None = None,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    initial_cash: float = 1.0,
    initial_inventory: float = 0.0,
    max_leverage: float | jax.Array = 1.0,
    can_short: bool | float | jax.Array = False,
    can_long: bool | float | jax.Array = True,
    decision_lag_bars: int = 0,
    market_order_entry: bool = False,
    fill_buffer_pct: float = 0.0,
    margin_annual_rate: float = 0.0,
) -> HourlySimulationResult:
    buy_trade_intensity = trade_intensity if buy_trade_intensity is None else buy_trade_intensity
    sell_trade_intensity = trade_intensity if sell_trade_intensity is None else sell_trade_intensity
    return _simulate_core(
        highs=highs,
        lows=lows,
        closes=closes,
        opens=opens,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=trade_intensity,
        buy_trade_intensity=buy_trade_intensity,
        sell_trade_intensity=sell_trade_intensity,
        maker_fee=maker_fee,
        initial_cash=initial_cash,
        initial_inventory=initial_inventory,
        temperature=5e-4,
        max_leverage=max_leverage,
        can_short=can_short,
        can_long=can_long,
        decision_lag_bars=decision_lag_bars,
        market_order_entry=market_order_entry,
        fill_buffer_pct=fill_buffer_pct,
        margin_annual_rate=margin_annual_rate,
        probabilistic=False,
    )


def compute_hourly_objective(
    hourly_returns: jax.Array,
    *,
    periods_per_year: float | jax.Array = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
    smoothness_penalty: float | jax.Array = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    hourly_returns = jnp.asarray(hourly_returns)
    mean_return = hourly_returns.mean(axis=-1)
    downside_sq = jnp.square(jnp.clip(-hourly_returns, min=0.0))
    downside_std = jnp.sqrt(downside_sq.mean(axis=-1) + _EPS)
    periods = _to_array(periods_per_year, mean_return)
    sortino = mean_return / jnp.clip(downside_std, min=_EPS)
    sortino = sortino * jnp.sqrt(jnp.clip(periods, min=_EPS))
    annual_return = mean_return * periods
    score = sortino + return_weight * annual_return
    if hourly_returns.shape[-1] > 1:
        smoothness_weight = _to_array(smoothness_penalty, mean_return)
        returns_diff = hourly_returns[..., 1:] - hourly_returns[..., :-1]
        score = score - returns_diff.std(axis=-1) * smoothness_weight
    return score, sortino, annual_return


def combined_sortino_pnl_loss(
    hourly_returns: jax.Array,
    *,
    target_sign: float = 1.0,
    periods_per_year: float | jax.Array = HOURLY_PERIODS_PER_YEAR,
    return_weight: float = 0.05,
    smoothness_penalty: float | jax.Array = 0.0,
) -> jax.Array:
    score, _, _ = compute_hourly_objective(
        hourly_returns,
        periods_per_year=periods_per_year,
        return_weight=return_weight,
        smoothness_penalty=smoothness_penalty,
    )
    loss = -target_sign * score.mean()
    if hourly_returns.shape[-1] > 1:
        smoothness_weight = _to_array(smoothness_penalty, score)
        diffs = hourly_returns[..., 1:] - hourly_returns[..., :-1]
        loss = loss + smoothness_weight * jnp.abs(diffs).mean()
    return loss
