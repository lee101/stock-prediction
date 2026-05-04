#!/usr/bin/env python3
"""Generate a local Plotly HTML workspace for Binance backtest traces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, read_mktd, simulate_daily_policy
from scripts.sweep_binance33_rules import RuleConfig, _default_configs, _make_policy, _slice_window


DEFAULT_RULE = "reversal5_short_bottom_rb1_thr0_btcmom20gt-0.05"
DEFAULT_DATA_PATH = Path("pufferlib_market/data/binance33_daily_val.bin")
DEFAULT_OUTPUT = Path("analysis/binance_backtest_space.html")
DEFAULT_TRACE_OUTPUT = Path("analysis/binance_backtest_space_trace.json")


def _float_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        out.append(float(value) if np.isfinite(value) else None)
    return out


def _find_rule(name: str) -> RuleConfig:
    for config in _default_configs():
        if config.name == name:
            return config
    raise ValueError(f"Unknown rule config {name!r}; run scripts/sweep_binance33_rules.py to list configs.")


def _resolve_window_start(data: MktdData, eval_days: int, requested: str) -> int:
    max_start = int(data.num_timesteps) - int(eval_days) - 1
    if max_start < 0:
        raise ValueError(f"data has {data.num_timesteps} bars; cannot make {eval_days}d + 1-bar window")
    value = str(requested).strip().lower()
    if value in {"latest", "last", "auto"}:
        return max_start
    start = int(value)
    if start < 0 or start > max_start:
        raise ValueError(f"window_start must be in [0, {max_start}], got {start}")
    return start


def _date_labels(
    *,
    data: MktdData,
    window_start: int,
    window_len: int,
    date_start: str,
    date_end: str,
) -> list[str]:
    if not date_start or not date_end:
        return [f"bar_{idx:04d}" for idx in range(window_start, window_start + window_len)]
    full_index = pd.date_range(
        pd.to_datetime(date_start, utc=True).floor("D"),
        pd.to_datetime(date_end, utc=True).floor("D"),
        freq="D",
        tz="UTC",
    )
    if len(full_index) != int(data.num_timesteps):
        return [f"bar_{idx:04d}" for idx in range(window_start, window_start + window_len)]
    labels = full_index[window_start : window_start + window_len]
    return [ts.strftime("%Y-%m-%d") for ts in labels]


def _bars_payload(data: MktdData, labels: list[str]) -> dict[str, dict[str, Any]]:
    bars: dict[str, dict[str, Any]] = {}
    for idx, symbol in enumerate(data.symbols):
        px = data.prices[:, idx, :]
        tradable = (
            np.asarray(data.tradable[:, idx], dtype=bool)
            if data.tradable is not None
            else np.ones((data.num_timesteps,), dtype=bool)
        )
        bars[str(symbol).upper()] = {
            "x": labels,
            "open": _float_list(px[:, 0]),
            "high": _float_list(px[:, 1]),
            "low": _float_list(px[:, 2]),
            "close": _float_list(px[:, 3]),
            "volume": _float_list(px[:, 4]),
            "tradable": [bool(x) for x in tradable],
        }
    return bars


def _position_payload(result, symbols: list[str], total_bars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append({"bar_index": 0, "symbol": "", "side": "flat", "qty": 0.0, "entry_price": None})
    history = result.position_history or []
    for idx in range(1, total_bars):
        pos = history[idx - 1] if idx - 1 < len(history) else None
        if pos is None:
            rows.append({"bar_index": idx, "symbol": "", "side": "flat", "qty": 0.0, "entry_price": None})
        else:
            rows.append(
                {
                    "bar_index": idx,
                    "symbol": symbols[int(pos.sym)],
                    "side": "short" if bool(pos.is_short) else "long",
                    "qty": float(pos.qty),
                    "entry_price": float(pos.entry_price),
                }
            )
    return rows


def _positions_by_bar_payload(result, symbols: list[str], total_bars: int) -> list[list[dict[str, Any]]]:
    positions = _position_payload(result, symbols, total_bars)
    rows: list[list[dict[str, Any]]] = []
    for rec in positions:
        if rec.get("side") == "flat" or not rec.get("symbol"):
            rows.append([])
        else:
            rows.append(
                [
                    {
                        "bar_index": int(rec["bar_index"]),
                        "symbol": str(rec["symbol"]).upper(),
                        "side": str(rec["side"]),
                        "qty": float(rec["qty"]),
                        "entry_price": rec["entry_price"],
                        "weight": None,
                    }
                ]
            )
    return rows


def _trade_payload(result, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in result.trade_events or []:
        rec = asdict(event)
        bar_index = int(rec["bar_index"])
        rec["x"] = labels[bar_index] if 0 <= bar_index < len(labels) else f"bar_{bar_index:04d}"
        rows.append(rec)
    return rows


def _decode_action(action: int, symbols: list[str]) -> dict[str, Any]:
    action = int(action)
    num_symbols = len(symbols)
    if action <= 0:
        return {"symbol": "", "side": "flat", "symbol_index": -1}
    idx = action - 1
    if 0 <= idx < num_symbols:
        return {"symbol": symbols[idx], "side": "long", "symbol_index": idx}
    short_idx = idx - num_symbols
    if 0 <= short_idx < num_symbols:
        return {"symbol": symbols[short_idx], "side": "short", "symbol_index": short_idx}
    return {"symbol": "", "side": "unknown", "symbol_index": -1}


def _action_payload(result, labels: list[str], symbols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, action in enumerate(np.asarray(result.actions, dtype=np.int64).reshape(-1)):
        decoded = _decode_action(int(action), symbols)
        rows.append({"bar_index": int(idx), "x": labels[idx], "action": int(action), **decoded})
    return rows


def _default_symbol(result, symbols: list[str], requested: str = "") -> str:
    requested = str(requested or "").upper()
    if requested:
        return requested

    history = result.position_history or []
    for pos in reversed(history):
        if pos is not None and 0 <= int(pos.sym) < len(symbols):
            return symbols[int(pos.sym)]

    counts: dict[str, int] = {}
    for pos in history:
        if pos is not None and 0 <= int(pos.sym) < len(symbols):
            symbol = symbols[int(pos.sym)]
            counts[symbol] = counts.get(symbol, 0) + 1
    for event in result.trade_events or []:
        symbol = str(event.symbol).upper()
        if symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
    if counts:
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    return symbols[0] if symbols else ""


def build_trace(args: argparse.Namespace) -> dict[str, Any]:
    data = read_mktd(args.data_path)
    eval_days = int(args.eval_days)
    window_start = _resolve_window_start(data, eval_days, str(args.window_start))
    window = _slice_window(data, window_start, eval_days)
    rule = _find_rule(str(args.rule_config))
    policy = _make_policy(window, rule, decision_lag=int(args.decision_lag))
    result = simulate_daily_policy(
        window,
        policy,
        max_steps=eval_days,
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.max_leverage),
        periods_per_year=float(args.periods_per_year),
        enable_drawdown_profit_early_exit=False,
        enable_metric_threshold_early_exit=False,
    )

    labels = _date_labels(
        data=data,
        window_start=window_start,
        window_len=eval_days + 1,
        date_start=str(args.date_start),
        date_end=str(args.date_end),
    )
    symbols = [str(symbol).upper() for symbol in window.symbols]
    equity = result.equity_curve if result.equity_curve is not None else np.asarray([], dtype=np.float64)

    return {
        "meta": {
            "data_path": str(args.data_path),
            "rule_config": rule.name,
            "window_start": int(window_start),
            "eval_days": int(eval_days),
            "decision_lag": int(args.decision_lag),
            "fee_rate": float(args.fee_rate),
            "slippage_bps": float(args.slippage_bps),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "max_leverage": float(args.max_leverage),
            "periods_per_year": float(args.periods_per_year),
            "total_return_pct": float(result.total_return * 100.0),
            "sortino": float(result.sortino),
            "max_drawdown_pct": float(result.max_drawdown * 100.0),
            "num_trades": int(result.num_trades),
            "win_rate_pct": float(result.win_rate * 100.0),
            "default_symbol": _default_symbol(result, symbols, str(args.default_symbol)),
        },
        "symbols": symbols,
        "labels": labels,
        "bars": _bars_payload(window, labels),
        "trades": _trade_payload(result, labels),
        "positions": _position_payload(result, symbols, len(labels)),
        "positions_by_bar": _positions_by_bar_payload(result, symbols, len(labels)),
        "actions": _action_payload(result, labels, symbols),
        "equity": {"x": labels[: len(equity)], "value": _float_list(equity)},
    }


def _html_document(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Backtest Space</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #617080;
      --line: #d8dee8;
      --panel: #f6f8fb;
      --accent: #0f766e;
      --buy: #15803d;
      --sell: #b91c1c;
      --short: #6d28d9;
      --cover: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #ffffff;
      color: var(--ink);
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 10px 14px 8px;
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
    }}
    h1 {{
      font-size: 18px;
      line-height: 1.2;
      margin: 0 10px 0 0;
      font-weight: 650;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      min-width: 0;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    select, input, button {{
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    select {{ min-width: 112px; padding: 0 8px; }}
    input[type="number"] {{ width: 72px; padding: 0 8px; }}
    input[type="range"] {{ width: min(420px, 42vw); }}
    button {{
      padding: 0 11px;
      cursor: pointer;
      background: var(--panel);
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .metrics {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
      padding: 8px 14px;
      font-size: 12px;
      color: var(--muted);
    }}
    .metrics strong {{ color: var(--ink); font-weight: 650; }}
    .workspace {{
      width: 100vw;
      min-height: calc(100vh - 104px);
    }}
    #chart {{
      width: 100vw;
      height: calc(100vh - 236px);
      min-height: 620px;
    }}
    .table-wrap {{
      border-top: 1px solid var(--line);
      padding: 8px 14px 12px;
      overflow-x: auto;
      background: #fff;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 760px;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #eef2f7;
      padding: 5px 8px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: #fafbfc;
    }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .side-short {{ color: var(--short); font-weight: 650; }}
    .side-long {{ color: var(--buy); font-weight: 650; }}
    .empty {{ color: var(--muted); }}
    @media (max-width: 760px) {{
      header {{ align-items: flex-start; }}
      h1 {{ width: 100%; }}
      input[type="range"] {{ width: 82vw; }}
      #chart {{ height: calc(100vh - 306px); min-height: 720px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Backtest Space</h1>
    <div class="controls">
      <label>Mode <select id="mode"><option value="multi">Multi</option><option value="single">Single</option></select></label>
      <label>Symbol <select id="symbol"></select></label>
      <label>Panels <input id="panels" type="number" min="1" max="10" step="1" value="10"></label>
      <label>Bars <input id="window" type="number" min="20" max="400" step="10" value="140"></label>
      <button id="play" class="primary" type="button">Play</button>
      <label>Frame <input id="frame" type="range" min="0" max="0" value="0"></label>
      <span id="frameLabel"></span>
    </div>
  </header>
  <section class="metrics" id="metrics"></section>
  <main class="workspace">
    <div id="chart"></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Slot</th>
            <th>Symbol</th>
            <th>Position</th>
            <th class="num">Weight</th>
            <th class="num">Qty</th>
            <th class="num">Entry</th>
            <th class="num">Mark</th>
            <th class="num">Unrealized</th>
            <th>Current Action</th>
            <th>Last Trade</th>
          </tr>
        </thead>
        <tbody id="positionRows"></tbody>
      </table>
    </div>
  </main>
  <script>
    const payload = {payload_json};
    const symbols = payload.symbols;
    const labels = payload.labels;
    const trades = payload.trades || [];
    const positions = payload.positions || [];
    const positionsByBar = payload.positions_by_bar || [];
    const actions = payload.actions || [];
    const symbolSelect = document.getElementById('symbol');
    const modeSelect = document.getElementById('mode');
    const panelsInput = document.getElementById('panels');
    const windowInput = document.getElementById('window');
    const frameInput = document.getElementById('frame');
    const playButton = document.getElementById('play');
    const frameLabel = document.getElementById('frameLabel');
    const positionRows = document.getElementById('positionRows');
    let timer = null;
    let slotSymbols = [];

    function fmtPct(value) {{
      const number = Number(value || 0);
      return `${{number >= 0 ? '+' : ''}}${{number.toFixed(2)}}%`;
    }}

    function fmtNum(value, digits = 6) {{
      const number = Number(value);
      if (!Number.isFinite(number)) return '';
      if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, {{maximumFractionDigits: 2}});
      if (Math.abs(number) >= 1) return number.toFixed(4);
      return number.toPrecision(digits);
    }}

    function metricHtml() {{
      const m = payload.meta;
      return [
        `<span><strong>${{m.rule_config}}</strong></span>`,
        `<span>return <strong>${{fmtPct(m.total_return_pct)}}</strong></span>`,
        `<span>max DD <strong>${{fmtPct(-Math.abs(m.max_drawdown_pct))}}</strong></span>`,
        `<span>trades <strong>${{m.num_trades}}</strong></span>`,
        `<span>lag <strong>${{m.decision_lag}}</strong></span>`,
        `<span>slip/fill <strong>${{m.slippage_bps}}/${{m.fill_buffer_bps}} bps</strong></span>`,
        `<span>leverage <strong>${{m.max_leverage}}x</strong></span>`
      ].join('');
    }}

    function initControls() {{
      for (const symbol of symbols) {{
        const option = document.createElement('option');
        option.value = symbol;
        option.textContent = symbol;
        symbolSelect.appendChild(option);
      }}
      const preferred = payload.meta.default_symbol && symbols.includes(payload.meta.default_symbol)
        ? payload.meta.default_symbol
        : symbols[0];
      symbolSelect.value = preferred;
      modeSelect.value = 'multi';
      frameInput.min = 0;
      frameInput.max = Math.max(0, labels.length - 1);
      frameInput.value = Math.max(0, labels.length - 1);
      document.getElementById('metrics').innerHTML = metricHtml();
    }}

    function eventStyle(side) {{
      if (side === 'buy') return {{color: '#15803d', symbol: 'triangle-up', name: 'buy'}};
      if (side === 'sell') return {{color: '#b91c1c', symbol: 'triangle-down', name: 'sell'}};
      if (side === 'short_sell') return {{color: '#6d28d9', symbol: 'triangle-down', name: 'short sell'}};
      return {{color: '#b45309', symbol: 'triangle-up', name: 'buy cover'}};
    }}

    function axisName(prefix, index) {{
      return index === 1 ? prefix : `${{prefix}}${{index}}`;
    }}

    function positionsAt(index) {{
      if (positionsByBar.length) return positionsByBar[Math.max(0, Math.min(index, positionsByBar.length - 1))] || [];
      const rec = positions[Math.max(0, Math.min(index, positions.length - 1))] || {{}};
      if (!rec.symbol || rec.side === 'flat') return [];
      return [rec];
    }}

    function actionAt(index) {{
      return actions[Math.max(0, Math.min(index, actions.length - 1))] || {{}};
    }}

    function markPrice(symbol, index) {{
      const bars = payload.bars[symbol];
      if (!bars) return null;
      return bars.close[Math.max(0, Math.min(index, bars.close.length - 1))];
    }}

    function positionPath(symbol, left, right) {{
      const x = [];
      const y = [];
      const text = [];
      for (let idx = left; idx <= right; idx++) {{
        const pos = positionsAt(idx).find((row) => row.symbol === symbol);
        const state = !pos ? 0 : pos.side === 'short' ? -1 : 1;
        x.push(labels[idx]);
        y.push(state);
        text.push(state > 0 ? 'long' : state < 0 ? 'short' : 'flat');
      }}
      return {{x, y, text}};
    }}

    function activeShapes(symbol, left, right, axisIndex) {{
      const path = positionPath(symbol, left, right);
      const shapes = [];
      let start = null;
      let side = 0;
      const shapeEnd = (value) => {{
        const parsed = Date.parse(`${{value}}T00:00:00Z`);
        if (!Number.isFinite(parsed)) return value;
        return new Date(parsed + 86400000).toISOString().slice(0, 10);
      }};
      for (let i = 0; i < path.y.length; i++) {{
        const value = path.y[i];
        if (value !== side) {{
          if (side !== 0 && start !== null) {{
            shapes.push({{
              type: 'rect',
              xref: axisName('x', axisIndex),
              yref: `${{axisName('y', axisIndex)}} domain`,
              x0: path.x[start],
              x1: shapeEnd(path.x[Math.max(start, i - 1)]),
              y0: 0,
              y1: 1,
              fillcolor: side > 0 ? 'rgba(21,128,61,0.10)' : 'rgba(109,40,217,0.10)',
              line: {{width: 0}},
              layer: 'below'
            }});
          }}
          start = value === 0 ? null : i;
          side = value;
        }}
      }}
      if (side !== 0 && start !== null && path.x.length) {{
        shapes.push({{
          type: 'rect',
          xref: axisName('x', axisIndex),
          yref: `${{axisName('y', axisIndex)}} domain`,
          x0: path.x[start],
          x1: shapeEnd(path.x[path.x.length - 1]),
          y0: 0,
          y1: 1,
          fillcolor: side > 0 ? 'rgba(21,128,61,0.10)' : 'rgba(109,40,217,0.10)',
          line: {{width: 0}},
          layer: 'below'
        }});
      }}
      return shapes;
    }}

    function tradeEvents(symbol, left, right) {{
      return trades.filter((event) =>
        event.symbol === symbol && Number(event.bar_index) >= left && Number(event.bar_index) <= right
      );
    }}

    function lastTradeText(symbol, frame) {{
      let latest = null;
      for (const event of trades) {{
        if (event.symbol === symbol && Number(event.bar_index) <= frame) latest = event;
      }}
      if (!latest) return '';
      return `${{latest.x}} ${{latest.side}} @ ${{fmtNum(latest.price)}}`;
    }}

    function relevantSymbols(frame, left, right, maxPanels) {{
      if (modeSelect.value === 'single') return [symbolSelect.value];
      const priority = new Map();
      const add = (symbol, score) => {{
        if (!symbol || !symbols.includes(symbol)) return;
        priority.set(symbol, Math.max(priority.get(symbol) || 0, score));
      }};
      for (const pos of positionsAt(frame)) add(pos.symbol, 100);
      const action = actionAt(frame);
      if (action.side && action.side !== 'flat') add(action.symbol, 85);
      for (let idx = left; idx <= right; idx++) {{
        for (const pos of positionsAt(idx)) add(pos.symbol, 60);
      }}
      for (const event of trades) {{
        const idx = Number(event.bar_index);
        if (idx >= left && idx <= right) add(event.symbol, 45);
      }}
      if (payload.meta.default_symbol) add(payload.meta.default_symbol, 10);
      add(symbolSelect.value, 5);
      const ranked = Array.from(priority.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
      const nextSlots = slotSymbols.slice(0, maxPanels).filter((symbol) => symbols.includes(symbol));
      for (const [symbol, score] of ranked) {{
        if (nextSlots.includes(symbol)) continue;
        if (nextSlots.length < maxPanels) {{
          nextSlots.push(symbol);
          continue;
        }}
        let replaceIndex = -1;
        let replaceScore = Infinity;
        for (let i = 0; i < nextSlots.length; i++) {{
          const existingScore = priority.get(nextSlots[i]) || 0;
          if (existingScore < replaceScore) {{
            replaceIndex = i;
            replaceScore = existingScore;
          }}
        }}
        if (replaceIndex >= 0 && score > replaceScore) {{
          nextSlots[replaceIndex] = symbol;
        }}
      }}
      slotSymbols = nextSlots.slice(0, maxPanels);
      return slotSymbols.length ? slotSymbols : [symbolSelect.value];
    }}

    function updatePositionTable(frame, panelSymbols) {{
      const rows = [];
      const currentPositions = new Map(positionsAt(frame).map((row) => [row.symbol, row]));
      const action = actionAt(frame);
      const tableSymbols = panelSymbols.slice();
      for (const pos of currentPositions.values()) {{
        if (!tableSymbols.includes(pos.symbol)) tableSymbols.push(pos.symbol);
      }}
      for (let i = 0; i < tableSymbols.length; i++) {{
        const symbol = tableSymbols[i];
        const slot = panelSymbols.indexOf(symbol);
        const pos = currentPositions.get(symbol);
        const mark = markPrice(symbol, frame);
        const side = pos ? pos.side : 'flat';
        const weight = pos && pos.weight !== undefined && pos.weight !== null ? Number(pos.weight) : null;
        const qty = pos ? Number(pos.qty) : 0;
        const entry = pos && pos.entry_price !== null ? Number(pos.entry_price) : null;
        let unrealized = '';
        if (pos && Number.isFinite(mark) && Number.isFinite(entry)) {{
          const signed = side === 'short' ? entry - mark : mark - entry;
          unrealized = fmtNum(signed * qty, 6);
        }}
        const actionText = action.symbol === symbol && action.side !== 'flat'
          ? `${{action.side}} action ${{action.action}}`
          : '';
        rows.push(`
          <tr>
            <td>${{slot >= 0 ? slot + 1 : 'off-panel'}}</td>
            <td>${{symbol}}</td>
            <td class="${{side === 'short' ? 'side-short' : side === 'long' ? 'side-long' : 'empty'}}">${{side}}</td>
            <td class="num">${{weight !== null ? fmtPct(weight * 100.0) : ''}}</td>
            <td class="num">${{qty ? fmtNum(qty, 6) : ''}}</td>
            <td class="num">${{entry !== null ? fmtNum(entry, 6) : ''}}</td>
            <td class="num">${{fmtNum(mark, 6)}}</td>
            <td class="num">${{unrealized}}</td>
            <td>${{actionText}}</td>
            <td>${{lastTradeText(symbol, frame)}}</td>
          </tr>
        `);
      }}
      if (!rows.length) {{
        rows.push('<tr><td colspan="10" class="empty">No active panels for this frame.</td></tr>');
      }}
      positionRows.innerHTML = rows.join('');
    }}

    function render() {{
      const frame = Number(frameInput.value);
      const count = Math.max(20, Number(windowInput.value || 140));
      const maxPanels = Math.max(1, Math.min(10, Number(panelsInput.value || 10)));
      const left = Math.max(0, frame - count + 1);
      const right = Math.min(frame, labels.length - 1);
      const slice = (arr) => arr.slice(left, right + 1);
      const panelSymbols = relevantSymbols(frame, left, right, maxPanels);

      const traces = [];
      const shapes = [];
      const layout = {{
        margin: {{l: 58, r: 24, t: 48, b: 34}},
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        hovermode: 'x unified',
        showlegend: true,
        legend: {{orientation: 'h', y: 1.045, x: 0, font: {{size: 10}}}},
        title: {{text: '', x: 0.01, xanchor: 'left', font: {{size: 14}}}},
        shapes
      }};
      const equityDomainTop = 0.145;
      const gap = 0.014;
      const panelArea = 1.0 - equityDomainTop - gap;
      const panelHeight = Math.max(0.05, (panelArea - gap * Math.max(0, panelSymbols.length - 1)) / panelSymbols.length);

      for (let i = 0; i < panelSymbols.length; i++) {{
        const symbol = panelSymbols[i];
        const bars = payload.bars[symbol];
        const axisIndex = i + 1;
        const axis = axisName('x', axisIndex);
        const yaxis = axisName('y', axisIndex);
        const y0 = equityDomainTop + gap + (panelSymbols.length - 1 - i) * (panelHeight + gap);
        const y1 = y0 + panelHeight;
        const held = positionsAt(frame).find((row) => row.symbol === symbol);
        const action = actionAt(frame);
        const axisTitle = held
          ? `${{symbol}} ${{held.side}}`
          : action.symbol === symbol && action.side !== 'flat'
            ? `${{symbol}} target ${{action.side}}`
            : symbol;
        layout[yaxis === 'y' ? 'yaxis' : `${{yaxis.replace('y', 'yaxis')}}`] = {{
          title: {{text: axisTitle, font: {{size: 11}}}},
          domain: [y0, y1],
          fixedrange: false,
          gridcolor: '#eef2f7',
          zeroline: false
        }};
        layout[axis === 'x' ? 'xaxis' : `${{axis.replace('x', 'xaxis')}}`] = {{
          rangeslider: {{visible: false}},
          showgrid: false,
          matches: axisIndex === 1 ? undefined : 'x',
          showticklabels: i === panelSymbols.length - 1
        }};
        traces.push({{
          type: 'candlestick',
          x: slice(bars.x),
          open: slice(bars.open),
          high: slice(bars.high),
          low: slice(bars.low),
          close: slice(bars.close),
          increasing: {{line: {{color: '#15803d', width: 1}}, fillcolor: '#d9f2df'}},
          decreasing: {{line: {{color: '#b91c1c', width: 1}}, fillcolor: '#f7d7d7'}},
          name: `${{symbol}} OHLC`,
          showlegend: i === 0,
          xaxis: axis,
          yaxis
        }});

        const bySide = new Map();
        for (const event of tradeEvents(symbol, left, right)) {{
          if (!bySide.has(event.side)) bySide.set(event.side, []);
          bySide.get(event.side).push(event);
        }}
        for (const [side, events] of bySide.entries()) {{
          const style = eventStyle(side);
          traces.push({{
            type: 'scatter',
            mode: 'markers',
            x: events.map((event) => event.x),
            y: events.map((event) => event.price),
            marker: {{color: style.color, symbol: style.symbol, size: 10, line: {{color: '#111827', width: 0.7}}}},
            name: style.name,
            showlegend: i === 0,
            text: events.map((event) =>
              `${{event.side}}<br>${{event.symbol}} ${{fmtNum(event.price, 6)}}` +
              `<br>qty ${{fmtNum(event.qty, 6)}}<br>${{event.reason}}`
            ),
            hovertemplate: '%{{text}}<extra></extra>',
            xaxis: axis,
            yaxis
          }});
        }}
        shapes.push(...activeShapes(symbol, left, right, axisIndex));
      }}

      const equityAxisIndex = panelSymbols.length + 1;
      const equityX = axisName('x', equityAxisIndex);
      const equityY = axisName('y', equityAxisIndex);
      layout[equityY.replace('y', 'yaxis')] = {{
        title: {{text: 'equity', font: {{size: 11}}}},
        domain: [0.0, equityDomainTop],
        gridcolor: '#eef2f7'
      }};
      layout[equityX.replace('x', 'xaxis')] = {{matches: 'x', showgrid: false}};

      traces.push({{
        type: 'scatter',
        mode: 'lines',
        x: payload.equity.x.slice(left, right + 1),
        y: payload.equity.value.slice(left, right + 1),
        line: {{color: '#2563eb', width: 2}},
        name: 'equity',
        hovertemplate: '%{{x}}<br>$%{{y:,.2f}}<extra></extra>',
        xaxis: equityX,
        yaxis: equityY
      }});

      const held = positionsAt(frame);
      const action = actionAt(frame);
      const holdingText = held.length
        ? held.map((row) => `${{row.side}} ${{row.symbol}}`).join(', ')
        : 'flat';
      const actionText = action.side && action.side !== 'flat'
        ? ` | action ${{action.side}} ${{action.symbol}}`
        : '';

      layout.title.text = `${{labels[frame]}} | ${{holdingText}}${{actionText}}`;
      frameLabel.textContent = `${{labels[frame]}}`;
      updatePositionTable(frame, panelSymbols);
      Plotly.react('chart', traces, layout, {{responsive: true, scrollZoom: true, displaylogo: false}});
    }}

    function togglePlay() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playButton.textContent = 'Play';
        return;
      }}
      playButton.textContent = 'Pause';
      timer = setInterval(() => {{
        const max = Number(frameInput.max);
        let value = Number(frameInput.value);
        value = value >= max ? 0 : value + 1;
        frameInput.value = value;
        render();
      }}, 260);
    }}

    initControls();
    symbolSelect.addEventListener('change', render);
    modeSelect.addEventListener('change', () => {{ slotSymbols = []; render(); }});
    panelsInput.addEventListener('change', () => {{ slotSymbols = []; render(); }});
    windowInput.addEventListener('change', render);
    frameInput.addEventListener('input', render);
    playButton.addEventListener('click', togglePlay);
    render();
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Plotly HTML workspace for a Binance simulator backtest.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE_OUTPUT)
    parser.add_argument("--rule-config", default=DEFAULT_RULE)
    parser.add_argument("--window-start", default="latest")
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--date-start", default="2025-10-01")
    parser.add_argument("--date-end", default="2026-03-14")
    parser.add_argument("--default-symbol", default="")
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=1.25)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    args = parser.parse_args(argv)

    payload = build_trace(args)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_html_document(payload), encoding="utf-8")

    meta = payload["meta"]
    print(f"html={args.output}")
    print(f"trace={args.trace_output}")
    print(
        "result="
        f"{meta['total_return_pct']:+.2f}% "
        f"dd={meta['max_drawdown_pct']:.2f}% "
        f"trades={meta['num_trades']} "
        f"default_symbol={meta['default_symbol'] or payload['symbols'][0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
