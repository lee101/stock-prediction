"""Per-stage marketsim video logger.

Renders an MP4 of one rollout showing OHLC candlesticks for the symbols the
policy actually trades or has pending orders on, with:
  - green/red position-held shading
  - ▲/▼ entry/exit markers at fill price
  - **gold** horizontal line + dot for steps where the policy has an active
    (placed) order at a specific price level — even if it didn't fill yet
  - bottom equity strip with running fees-paid in 10bps-per-fill accounting
  - top header showing live Sortino + PnL recomputed every frame

The set of panels is chosen dynamically from the union of every symbol the
policy held at some point AND every symbol it tried to order, capped at
`num_pairs`. There is no hardcoded symbol list.

Speed/realism:
  - candlesticks are drawn from per-step OHLC (provide `prices_ohlc[T,S,4]`)
  - `frames_per_bar` repeats each simulator bar across N video frames so the
    video plays back hourly-cinematic instead of one-flash-per-bar
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class OrderTick:
    sym: int
    price: float  # the price level the policy is trying to execute at
    is_short: bool


@dataclass
class TradeSegment:
    sym: int
    entry_step: int
    entry_price: float
    exit_step: int
    exit_price: float
    is_short: bool


@dataclass
class TraceFrame:
    step: int
    action_id: int
    position_sym: int  # -1 if flat
    position_is_short: bool
    equity: float
    orders: list[OrderTick] = field(default_factory=list)  # active orders this step
    forecast: np.ndarray | None = None  # [S] predicted next-bar close
    forecast_ohlc: np.ndarray | None = None  # [S, 4] predicted next-bar O,H,L,C
    long_target: np.ndarray | None = None  # [S] price the policy would buy at this bar
    short_target: np.ndarray | None = None  # [S] price the policy would short at this bar


@dataclass
class MarketsimTrace:
    """Records per-step state during a single validation rollout.

    `prices` is the per-step **close** array [T, S]. For candlestick rendering
    pass the full OHLC via `prices_ohlc` [T, S, 4] (O,H,L,C). When omitted the
    renderer falls back to degenerate doji bars at the close.
    """

    symbols: list[str]
    prices: np.ndarray  # [T, S] close
    frames: list[TraceFrame] = field(default_factory=list)
    prices_ohlc: np.ndarray | None = None  # [T, S, 4] O,H,L,C
    trades: list[TradeSegment] = field(default_factory=list)

    def record(
        self,
        step: int,
        action_id: int,
        position_sym: int,
        position_is_short: bool,
        equity: float,
        orders: list[OrderTick] | None = None,
        forecast: np.ndarray | None = None,
        forecast_ohlc: np.ndarray | None = None,
        long_target: np.ndarray | None = None,
        short_target: np.ndarray | None = None,
    ) -> None:
        self.frames.append(
            TraceFrame(
                step=int(step),
                action_id=int(action_id),
                position_sym=int(position_sym),
                position_is_short=bool(position_is_short),
                equity=float(equity),
                orders=list(orders) if orders else [],
                forecast=None if forecast is None else np.asarray(forecast, dtype=np.float32),
                forecast_ohlc=None if forecast_ohlc is None else np.asarray(forecast_ohlc, dtype=np.float32),
                long_target=None if long_target is None else np.asarray(long_target, dtype=np.float32),
                short_target=None if short_target is None else np.asarray(short_target, dtype=np.float32),
            )
        )

    # ── JSON I/O so C code (or any other emitter) can dump traces and the
    # python renderer can consume them. Keep the schema flat and explicit.
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "marketsim_trace_v1",
            "symbols": list(self.symbols),
            "prices_close": self.prices.astype(np.float32).tolist(),
            "prices_ohlc": (
                None if self.prices_ohlc is None
                else np.asarray(self.prices_ohlc, dtype=np.float32).tolist()
            ),
            "frames": [
                {
                    "step": f.step,
                    "action_id": f.action_id,
                    "position_sym": f.position_sym,
                    "position_is_short": f.position_is_short,
                    "equity": f.equity,
                    "orders": [
                        {"sym": o.sym, "price": o.price, "is_short": o.is_short}
                        for o in f.orders
                    ],
                    "forecast": None if f.forecast is None else f.forecast.tolist(),
                    "forecast_ohlc": None if f.forecast_ohlc is None else f.forecast_ohlc.tolist(),
                    "long_target": None if f.long_target is None else f.long_target.tolist(),
                    "short_target": None if f.short_target is None else f.short_target.tolist(),
                }
                for f in self.frames
            ],
            "trades": [
                {
                    "sym": t.sym,
                    "entry_step": t.entry_step,
                    "entry_price": t.entry_price,
                    "exit_step": t.exit_step,
                    "exit_price": t.exit_price,
                    "is_short": t.is_short,
                }
                for t in self.trades
            ],
        }
        path.write_text(json.dumps(payload))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> MarketsimTrace:
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") not in ("marketsim_trace_v1", None):
            raise ValueError(f"Unknown trace schema: {payload.get('schema')!r}")
        prices = np.asarray(payload["prices_close"], dtype=np.float32)
        ohlc = payload.get("prices_ohlc")
        ohlc_np = None if ohlc is None else np.asarray(ohlc, dtype=np.float32)
        tr = cls(symbols=list(payload["symbols"]), prices=prices, prices_ohlc=ohlc_np)
        for f in payload["frames"]:
            tr.record(
                step=f["step"],
                action_id=f["action_id"],
                position_sym=f["position_sym"],
                position_is_short=f["position_is_short"],
                equity=f["equity"],
                orders=[OrderTick(sym=o["sym"], price=o["price"], is_short=o["is_short"]) for o in f.get("orders", [])],
                forecast=None if f.get("forecast") is None else np.asarray(f["forecast"], dtype=np.float32),
                forecast_ohlc=None if f.get("forecast_ohlc") is None else np.asarray(f["forecast_ohlc"], dtype=np.float32),
                long_target=None if f.get("long_target") is None else np.asarray(f["long_target"], dtype=np.float32),
                short_target=None if f.get("short_target") is None else np.asarray(f["short_target"], dtype=np.float32),
            )
        tr.trades = [
            TradeSegment(
                sym=int(t["sym"]),
                entry_step=int(t["entry_step"]),
                entry_price=float(t["entry_price"]),
                exit_step=int(t["exit_step"]),
                exit_price=float(t["exit_price"]),
                is_short=bool(t["is_short"]),
            )
            for t in payload.get("trades", [])
        ]
        return tr

    def num_symbols(self) -> int:
        return self.prices.shape[1]

    def num_steps(self) -> int:
        return len(self.frames)

    def active_symbol_indices(self, k: int) -> list[int]:
        """Symbols the policy actually touched (held or ordered), capped at k.

        Sorted by activity (held + ordered count, descending). Returns only
        symbols with non-zero activity — no padding with dead panels. If
        nothing was traded at all, falls back to the first symbol so the grid
        is non-empty.
        """
        S = self.num_symbols()
        score = np.zeros(S, dtype=np.int64)
        for fr in self.frames:
            if fr.position_sym >= 0:
                score[fr.position_sym] += 2  # held weighs more
            for od in fr.orders:
                if 0 <= od.sym < S:
                    score[od.sym] += 1
        active = [i for i in range(S) if score[i] > 0]
        active.sort(key=lambda i: (-score[i], i))
        if not active:
            return [0]
        return sorted(active[: max(1, int(k))])

    # Back-compat alias
    def select_symbol_indices(self, k: int) -> list[int]:
        return self.active_symbol_indices(k)


def _group_trade_segments(
    trace: MarketsimTrace,
    close: np.ndarray,
) -> dict[int, list[TradeSegment]]:
    """Return explicit trade segments, deriving legacy single-position ones when needed."""
    T, S = close.shape
    if trace.trades:
        grouped = {s: [] for s in range(S)}
        for seg in trace.trades:
            if 0 <= seg.sym < S:
                grouped[seg.sym].append(seg)
        return grouped

    grouped = {s: [] for s in range(S)}
    open_entry: dict[int, tuple[int, float, bool]] = {}
    prev_sym, prev_short = -1, False
    for i, frame in enumerate(trace.frames):
        cur_sym = int(frame.position_sym)
        cur_short = bool(frame.position_is_short)
        if (cur_sym, cur_short) == (prev_sym, prev_short):
            continue
        if prev_sym in grouped and prev_sym in open_entry:
            entry_step, entry_price, was_short = open_entry.pop(prev_sym)
            grouped[prev_sym].append(
                TradeSegment(
                    sym=prev_sym,
                    entry_step=entry_step,
                    entry_price=entry_price,
                    exit_step=i,
                    exit_price=float(close[i, prev_sym]),
                    is_short=was_short,
                )
            )
        if cur_sym in grouped:
            open_entry[cur_sym] = (i, float(close[i, cur_sym]), cur_short)
        prev_sym, prev_short = cur_sym, cur_short

    if T > 0:
        last_step = T - 1
        for sym, (entry_step, entry_price, was_short) in open_entry.items():
            grouped[sym].append(
                TradeSegment(
                    sym=sym,
                    entry_step=entry_step,
                    entry_price=entry_price,
                    exit_step=last_step,
                    exit_price=float(close[last_step, sym]),
                    is_short=was_short,
                )
            )
    return grouped


def _fill_events_from_segments(
    trade_segments: dict[int, list[TradeSegment]],
    symbol_indices: list[int],
) -> dict[int, dict[str, list[float | int]]]:
    fills_per_sym: dict[int, dict[str, list[float | int]]] = {
        s: {"bx": [], "by": [], "sx": [], "sy": []}
        for s in symbol_indices
    }
    for sym in symbol_indices:
        for seg in trade_segments.get(sym, []):
            if seg.is_short:
                fills_per_sym[sym]["sx"].append(seg.entry_step)
                fills_per_sym[sym]["sy"].append(seg.entry_price)
                fills_per_sym[sym]["bx"].append(seg.exit_step)
                fills_per_sym[sym]["by"].append(seg.exit_price)
            else:
                fills_per_sym[sym]["bx"].append(seg.entry_step)
                fills_per_sym[sym]["by"].append(seg.entry_price)
                fills_per_sym[sym]["sx"].append(seg.exit_step)
                fills_per_sym[sym]["sy"].append(seg.exit_price)
    return fills_per_sym


def _trade_fill_counts_by_step(trace: MarketsimTrace, total_steps: int, close: np.ndarray) -> np.ndarray:
    counts = np.zeros(total_steps, dtype=np.float64)
    grouped = _group_trade_segments(trace, close)
    for segments in grouped.values():
        for seg in segments:
            if 0 <= seg.entry_step < total_steps:
                counts[seg.entry_step] += 1.0
            if 0 <= seg.exit_step < total_steps:
                counts[seg.exit_step] += 1.0
    return counts


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n <= 4:
        return (2, 2)
    if n <= 6:
        return (2, 3)
    if n <= 9:
        return (3, 3)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return (rows, cols)


def _rolling_sortino(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sortino over the per-step return series. Returns 0 when undefined."""
    if returns.size < 2:
        return 0.0
    mean = float(np.mean(returns))
    downside = returns[returns < 0.0]
    if downside.size == 0 or float(np.std(downside)) == 0.0:
        return 0.0
    dd = float(np.sqrt(np.mean(downside * downside)))
    if dd == 0.0:
        return 0.0
    return float(mean / dd * np.sqrt(periods_per_year))


def _draw_candles(
    ax,
    ohlc: np.ndarray,
    t_end: int,
    *,
    up_color: str = "#26a69a",
    down_color: str = "#ef5350",
    width: float = 0.64,
    alpha: float = 0.95,
) -> None:
    """Draw OHLC candlesticks for steps 0..t_end inclusive on `ax`."""
    half = width / 2.0
    for i in range(t_end + 1):
        o, h, low, c = float(ohlc[i, 0]), float(ohlc[i, 1]), float(ohlc[i, 2]), float(ohlc[i, 3])
        if not np.isfinite(h) or not np.isfinite(low):
            continue
        up = c >= o
        color = up_color if up else down_color
        ax.vlines(i, low, h, color=color, linewidth=1.0, zorder=2, alpha=alpha)
        body_lo = min(o, c)
        body_hi = max(o, c)
        if body_hi - body_lo < 1e-12:
            ax.hlines(o, i - half, i + half, color=color, linewidth=1.4, zorder=3, alpha=alpha)
        else:
            ax.add_patch(
                __mpl_rect((i - half, body_lo), width, body_hi - body_lo,
                           edgecolor=color, facecolor=color, alpha=alpha,
                           linewidth=0.6, zorder=3)
            )


def _draw_forecast_candle(ax, x: float, ohlc4: np.ndarray, color: str, *, label: bool = False) -> None:
    """Draw a translucent forecast candle one bar ahead of `t` at x-coord `x`."""
    o, h, low, c = float(ohlc4[0]), float(ohlc4[1]), float(ohlc4[2]), float(ohlc4[3])
    if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(low) and np.isfinite(c)):
        return
    half = 0.34
    ax.vlines(x, low, h, color=color, linewidth=2.2, zorder=2.5, alpha=0.95)
    body_lo = min(o, c)
    body_hi = max(o, c)
    if body_hi - body_lo < 1e-12:
        ax.hlines(o, x - half, x + half, color=color, linewidth=2.4, zorder=3.5, alpha=0.95)
    else:
        ax.add_patch(
            __mpl_rect((x - half, body_lo), 2 * half, body_hi - body_lo,
                       edgecolor=color, facecolor=color, alpha=0.45,
                       linewidth=1.6, linestyle="--", zorder=3.5)
        )
    if label:
        ax.annotate(
            "chronos2",
            xy=(x, max(h, body_hi)),
            xytext=(0, 6), textcoords="offset points",
            color=color, fontsize=8, fontweight="bold", ha="center", zorder=7,
        )


def _ffmpeg_has_nvenc() -> bool:
    import subprocess
    import imageio_ffmpeg

    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        out = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return "h264_nvenc" in (out.stdout + out.stderr)
    except Exception:
        return False


def _open_video_writer(out_path: Path, *, fps: int, nvenc: bool):
    """Open an imageio writer; prefer NVENC when available, fall back to libx264."""
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_nvenc = bool(nvenc) and _ffmpeg_has_nvenc()
    base = dict(
        fps=int(fps),
        macro_block_size=16,
        ffmpeg_log_level="error",
    )
    if use_nvenc:
        try:
            return imageio.get_writer(
                str(out_path),
                codec="h264_nvenc",
                output_params=["-preset", "p5", "-cq", "22", "-b:v", "16M"],
                **base,
            )
        except Exception as e:
            print(f"  nvenc init failed ({e}); falling back to libx264")
    return imageio.get_writer(str(out_path), codec="libx264", quality=7, **base)


def _open_software_video_writer(out_path: Path, *, fps: int):
    """Open a conservative software encoder for environments where libx264 flakes."""
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(out_path),
        fps=int(fps),
        codec="mpeg4",
        macro_block_size=16,
        ffmpeg_log_level="error",
    )


def __mpl_rect(xy, w, h, **kw):
    # Lazy import so the module can be imported without matplotlib at module-load time.
    from matplotlib.patches import Rectangle

    return Rectangle(xy, w, h, **kw)


def render_mp4(
    trace: MarketsimTrace,
    out_path: str | Path,
    *,
    num_pairs: int = 4,
    fps: int = 4,
    frames_per_bar: int = 1,
    title: str = "marketsim",
    dpi: int = 100,
    fee_rate: float = 0.001,  # 10 bps per fill (matches production)
    periods_per_year: float = 252.0,
    short_borrow_apr: float = 0.05,  # ~5% APR borrow on shorted shares
    margin_apr: float = 0.0625,  # 6.25% APR on leveraged long notional (matches prod)
    leverage: float = 1.0,  # used for long-side margin interest accrual
    width_px: int = 2560,
    height_px: int = 1440,
    nvenc: bool = True,
    dynamic_panels: bool = True,
    panel_window: int = 20,
) -> Path:
    """Render the recorded trace to an MP4. Returns the output path.

    `frames_per_bar` repeats each simulator bar across N video frames so playback
    feels closer to a moving market instead of a stop-motion update per bar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if trace.num_steps() == 0:
        raise ValueError("MarketsimTrace has no recorded frames")

    # Initial panel set (used as fallback / static layout when dynamic panels
    # is off). When `dynamic_panels` is on, we recompute per bar from a
    # rolling window of recent activity so charts evolve as the policy
    # rotates which symbols it actually trades.
    sym_idxs = trace.active_symbol_indices(num_pairs)
    n_panels = max(1, len(sym_idxs))
    _rows, _cols = _grid_shape(n_panels)

    def _panels_at_step(t: int) -> list[int]:
        """Top-K symbols by activity in the rolling window ending at t."""
        if not dynamic_panels:
            return sym_idxs
        S = trace.num_symbols()
        score = np.zeros(S, dtype=np.int64)
        lo = max(0, t - int(panel_window) + 1)
        for i in range(lo, t + 1):
            fr = trace.frames[i]
            if fr.position_sym >= 0:
                score[fr.position_sym] += 2
            for od in fr.orders:
                if 0 <= od.sym < S:
                    score[od.sym] += 1
        active = [i for i in range(S) if score[i] > 0]
        active.sort(key=lambda i: (-score[i], i))
        if not active:
            # carry over previous active set if nothing happened in the window
            return sym_idxs
        return sorted(active[: max(1, int(num_pairs))])

    T = trace.num_steps()
    S = trace.num_symbols()
    close = trace.prices[:T]
    if trace.prices_ohlc is not None:
        ohlc = np.asarray(trace.prices_ohlc[:T], dtype=np.float64)
    else:
        # Degenerate: O=H=L=C. Still renders as ticks.
        ohlc = np.stack([close, close, close, close], axis=-1).astype(np.float64)

    equity = np.asarray([f.equity for f in trace.frames], dtype=np.float64)
    pos_sym = np.asarray([f.position_sym for f in trace.frames], dtype=np.int64)
    pos_short = np.asarray([f.position_is_short for f in trace.frames], dtype=bool)
    initial_eq = float(equity[0]) if equity.size else 1.0

    # Per-step return for Sortino
    step_ret = np.zeros(T, dtype=np.float64)
    if T > 1:
        step_ret[1:] = np.diff(equity) / np.maximum(np.abs(equity[:-1]), 1e-12)

    # Cumulative fees: 1 fill on close + 1 fill on open per transition.
    fees_step = fee_rate * np.asarray(equity, dtype=np.float64) * _trade_fill_counts_by_step(trace, T, close)
    fees_cum = np.cumsum(fees_step)

    # Borrow / margin interest accrued per step. Charged whenever there is an
    # open short (short borrow APR) or a long with leverage>1 (margin APR on
    # the levered portion). Per-step charge = notional * apr / periods_per_year
    # where notional ≈ equity * effective leverage.
    borrow_step = np.zeros(T, dtype=np.float64)
    lev = max(1.0, float(leverage))
    for i in range(T):
        if pos_sym[i] < 0:
            continue
        notional = float(equity[i]) * lev
        if pos_short[i] and short_borrow_apr > 0.0:
            borrow_step[i] += notional * float(short_borrow_apr) / float(periods_per_year)
        if (not pos_short[i]) and lev > 1.0 and margin_apr > 0.0:
            borrow_step[i] += float(equity[i]) * (lev - 1.0) * float(margin_apr) / float(periods_per_year)
    borrow_cum = np.cumsum(borrow_step)

    all_syms_range = list(range(S))
    trades_per_sym = _group_trade_segments(trace, close)

    order_ticks: dict[int, list[tuple[int, float, bool]]] = {s: [] for s in all_syms_range}
    for i, fr in enumerate(trace.frames):
        for od in fr.orders:
            if 0 <= od.sym < S:
                order_ticks[od.sym].append((i, float(od.price), bool(od.is_short)))

    # 1080p by default. figsize in inches x dpi -> pixels.
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="#0e1117")
    ax_panels: list[tuple[int, object]] = []
    ax_eq = None
    current_panel_set: list[int] = []

    def _build_layout(sym_set: list[int]) -> None:
        nonlocal ax_panels, ax_eq, current_panel_set
        fig.clf()
        n = max(1, len(sym_set))
        r, c = _grid_shape(n)
        gs = fig.add_gridspec(r + 1, c, height_ratios=[3] * r + [1.3],
                              hspace=0.55, wspace=0.28,
                              left=0.04, right=0.985, top=0.93, bottom=0.06)
        ax_panels = []
        for k, sidx in enumerate(sym_set):
            rr, cc = divmod(k, c)
            ax_panels.append((sidx, fig.add_subplot(gs[rr, cc])))
        ax_eq = fig.add_subplot(gs[r, :])
        current_panel_set = list(sym_set)

    _build_layout(sym_idxs)

    # High-contrast palette over dark background
    GOLD = "#ffd400"
    LONG_GREEN = "#26a69a"
    SHORT_RED = "#ef5350"
    LONG_LINE = "#69f0ae"   # bright green for daily long-target line
    SHORT_LINE = "#ff6e6e"  # bright red for daily short-target line
    FORECAST_FILL = "#42a5f5"  # blue forecast bar
    EQUITY_COLOR = "#ce93d8"
    TEXT_COLOR = "#eaeaea"
    GRID_COLOR = "#303339"

    def _style_dark(ax):
        ax.set_facecolor("#15181f")
        for spine in ax.spines.values():
            spine.set_color("#3a3f4b")
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.6)
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)

    def _encode_video(target_path: Path, *, use_nvenc: bool, conservative_software: bool = False) -> None:
        writer = (
            _open_software_video_writer(target_path, fps=fps)
            if conservative_software
            else _open_video_writer(target_path, fps=fps, nvenc=use_nvenc)
        )
        total_video_frames = T * max(1, int(frames_per_bar))
        try:
            for vf in range(total_video_frames):
                t = vf // max(1, int(frames_per_bar))
                # Recompute panel set; rebuild grid only when it changes.
                desired = _panels_at_step(t)
                if desired != current_panel_set:
                    _build_layout(desired)
                for sidx, ax in ax_panels:
                    ax.clear()
                    _draw_candles(ax, ohlc[:, sidx, :], t,
                                  up_color=LONG_GREEN, down_color=SHORT_RED)

                    # Daily long/short target lines: per bar, plot a green segment at
                    # the long target price and a red segment at the short target
                    # price spanning that bar (i-0.45 .. i+0.45).
                    for i in range(t + 1):
                        fr = trace.frames[i]
                        if fr.long_target is not None and 0 <= sidx < len(fr.long_target):
                            lp = float(fr.long_target[sidx])
                            if np.isfinite(lp) and lp > 0:
                                ax.hlines(lp, i - 0.45, i + 0.45, color=LONG_LINE,
                                          linewidth=1.6, alpha=0.85, zorder=4)
                        if fr.short_target is not None and 0 <= sidx < len(fr.short_target):
                            sp = float(fr.short_target[sidx])
                            if np.isfinite(sp) and sp > 0:
                                ax.hlines(sp, i - 0.45, i + 0.45, color=SHORT_LINE,
                                          linewidth=1.6, alpha=0.85, zorder=4)

                    # Held shading from explicit trade segments when present.
                    for seg in trades_per_sym.get(sidx, []):
                        if seg.entry_step > t:
                            continue
                        shade_end = min(seg.exit_step, t)
                        shade_color = SHORT_RED if seg.is_short else LONG_GREEN
                        ax.axvspan(seg.entry_step - 0.4, shade_end + 0.4, alpha=0.13, color=shade_color)

                    # Active-order overlay: vertical gold band + bold ★ star (very
                    # distinct from the green/red triangles used for fills).
                    for (oi, oprice, oshort) in order_ticks[sidx]:
                        if oi == t:
                            ax.axvspan(oi - 0.45, oi + 0.45, color=GOLD, alpha=0.20, zorder=1.5)
                            ax.scatter([oi], [oprice], marker="*", color=GOLD,
                                       edgecolor="black", linewidth=1.0, s=320, zorder=6)
                            ax.annotate(
                                "ORDER " + ("SHORT" if oshort else "LONG"),
                                xy=(oi, oprice),
                                xytext=(8, 10), textcoords="offset points",
                                color=GOLD, fontsize=10, fontweight="bold",
                                zorder=7,
                            )

                    # Round-trip trades: dot-at-entry → line → dot-at-exit, with
                    # entry/exit prices labeled. Color the connecting line by
                    # P&L direction (green = winning, red = losing).
                    for seg in trades_per_sym.get(sidx, []):
                        if seg.entry_step > t:
                            continue
                        visible_exit_step = min(seg.exit_step, t)
                        visible_exit_price = seg.exit_price if seg.exit_step <= t else float(close[t, sidx])
                        # P&L sign — for shorts, profit when exit < entry.
                        if seg.is_short:
                            winning = visible_exit_price < seg.entry_price
                        else:
                            winning = visible_exit_price > seg.entry_price
                        line_color = LONG_GREEN if winning else SHORT_RED
                        # entry dot (large, white-edged)
                        ax.scatter([seg.entry_step], [seg.entry_price], marker="o",
                                   color="white", edgecolor=line_color, linewidth=2.0,
                                   s=110, zorder=6)
                        ax.annotate(
                            f"{'S' if seg.is_short else 'B'} {seg.entry_price:.2f}",
                            xy=(seg.entry_step, seg.entry_price),
                            xytext=(-4, -14 if seg.is_short else 10),
                            textcoords="offset points",
                            color=line_color, fontsize=8, fontweight="bold",
                            ha="right" if seg.entry_step > 2 else "left",
                            zorder=7,
                        )
                        # connecting line
                        ax.plot(
                            [seg.entry_step, visible_exit_step],
                            [seg.entry_price, visible_exit_price],
                            color=line_color, linewidth=1.6, alpha=0.8, zorder=4.5,
                        )
                        # exit dot — only when fully closed within visible range
                        if seg.exit_step <= t:
                            ax.scatter([seg.exit_step], [seg.exit_price], marker="o",
                                       color=line_color, edgecolor="white", linewidth=1.2,
                                       s=110, zorder=6)
                            ax.annotate(
                                f"X {seg.exit_price:.2f}",
                                xy=(seg.exit_step, seg.exit_price),
                                xytext=(6, 10), textcoords="offset points",
                                color=line_color, fontsize=8, fontweight="bold",
                                zorder=7,
                            )

                    # Chronos2 forecast: phantom candle one bar ahead. Prefer
                    # forecast_ohlc when present; else build a degenerate one from
                    # the scalar forecast.
                    fr_t = trace.frames[t]
                    fc_x = t + 1
                    if fr_t.forecast_ohlc is not None and fc_x < T + 1:
                        _draw_forecast_candle(ax, fc_x, fr_t.forecast_ohlc[sidx], FORECAST_FILL, label=True)
                    elif fr_t.forecast is not None and 0 <= sidx < len(fr_t.forecast):
                        fc = float(fr_t.forecast[sidx])
                        if np.isfinite(fc):
                            last_close = float(ohlc[t, sidx, 3])
                            _draw_forecast_candle(
                                ax, fc_x,
                                np.array([last_close, max(last_close, fc), min(last_close, fc), fc],
                                         dtype=np.float32),
                                FORECAST_FILL, label=True,
                            )

                    ax.set_title(trace.symbols[sidx], fontsize=11, color=TEXT_COLOR, pad=4)
                    ax.set_xlim(-0.7, T + 0.5)
                    ymin = float(np.nanmin(ohlc[:, sidx, 2]))
                    ymax = float(np.nanmax(ohlc[:, sidx, 1]))
                    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                        ymin, ymax = float(close[:, sidx].min()) - 1, float(close[:, sidx].max()) + 1
                    pad = (ymax - ymin) * 0.06
                    ax.set_ylim(ymin - pad, ymax + pad)
                    _style_dark(ax)

                ax_eq.clear()
                ax_eq.plot(np.arange(t + 1), equity[: t + 1], color=EQUITY_COLOR, linewidth=2.0)
                ax_eq.axhline(initial_eq, color="#666", linewidth=0.8, linestyle=":")
                pnl_pct = (equity[t] / initial_eq - 1.0) * 100.0 if initial_eq != 0 else 0.0
                sortino = _rolling_sortino(step_ret[: t + 1], periods_per_year=periods_per_year)
                ax_eq.set_title(
                    f"equity \\${equity[t]:,.0f}   PnL {pnl_pct:+.2f}%   "
                    f"Sortino {sortino:.2f}   fees \\${fees_cum[t]:,.2f}   "
                    f"borrow \\${borrow_cum[t]:,.2f}",
                    fontsize=12, color=TEXT_COLOR,
                )
                ax_eq.set_xlim(-0.7, T + 0.5)
                _style_dark(ax_eq)

                fig.suptitle(
                    f"{title}    PnL {pnl_pct:+.2f}%    Sortino {sortino:.2f}    bar {t+1}/{T}",
                    fontsize=15, color=TEXT_COLOR, y=0.985, fontweight="bold",
                )
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                target_path.parent.mkdir(parents=True, exist_ok=True)
                writer.append_data(buf[..., :3])
        finally:
            writer.close()

    try:
        _encode_video(out_path, use_nvenc=nvenc)
        output_size = out_path.stat().st_size if out_path.exists() else 0
        if output_size < 1024:
            out_path.unlink(missing_ok=True)
            _encode_video(out_path, use_nvenc=False, conservative_software=True)
            output_size = out_path.stat().st_size if out_path.exists() else 0
        if output_size < 1024:
            raise RuntimeError(f"render_mp4 wrote suspiciously small output ({output_size} bytes)")
    finally:
        import matplotlib.pyplot as _plt
        _plt.close(fig)

    return out_path


def render_html_plotly(
    trace: MarketsimTrace,
    out_path: str | Path,
    *,
    num_pairs: int = 6,
    title: str = "marketsim",
    animated: bool = True,
) -> Path:
    """Write an interactive Plotly HTML — fully scrubbable timeline.

    Each bar is a Plotly animation frame. The user gets:
      - hover over any candle / fill / forecast point
      - click ▶ to play, drag the slider to scrub to any bar
      - zoom / pan / toggle traces via the legend
      - one row per active symbol + an equity row, all linked on the x axis
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "render_html_plotly requires the optional 'plotly' dependency. "
            "Install plotly to enable interactive HTML rendering."
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sym_idxs = trace.active_symbol_indices(num_pairs)
    n = max(1, len(sym_idxs))
    rows_panels = int(np.ceil(n / 2))
    cols = 1 if n == 1 else 2
    titles = [trace.symbols[s] for s in sym_idxs] + ["equity / PnL"]
    specs = [[{"type": "candlestick"} for _ in range(cols)] for _ in range(rows_panels)]
    specs.append([{"type": "scatter", "colspan": cols}] + [None] * (cols - 1))
    fig = make_subplots(rows=rows_panels + 1, cols=cols, subplot_titles=titles,
                        shared_xaxes=False, vertical_spacing=0.06, specs=specs)

    T = trace.num_steps()
    if trace.prices_ohlc is not None:
        ohlc = np.asarray(trace.prices_ohlc[:T], dtype=np.float64)
    else:
        cp = trace.prices[:T]
        ohlc = np.stack([cp, cp, cp, cp], axis=-1).astype(np.float64)
    x_all = list(range(T))

    trade_segments = _group_trade_segments(trace, trace.prices[:T])
    fills_per_sym = _fill_events_from_segments(trade_segments, sym_idxs)

    eq = np.asarray([f.equity for f in trace.frames], dtype=np.float64)
    initial_eq = float(eq[0]) if eq.size else 1.0

    # Order events with prices per symbol
    orders_per_sym: dict[int, dict[str, list]] = {s: {"x": [], "y": [], "side": []} for s in sym_idxs}
    forecast_per_sym: dict[int, dict[str, list]] = {s: {"x": [], "y": []} for s in sym_idxs}
    for i, f in enumerate(trace.frames):
        for od in f.orders:
            if od.sym in orders_per_sym:
                orders_per_sym[od.sym]["x"].append(i)
                orders_per_sym[od.sym]["y"].append(float(od.price))
                orders_per_sym[od.sym]["side"].append("SHORT" if od.is_short else "LONG")
        if f.forecast is not None:
            for sidx in sym_idxs:
                if 0 <= sidx < len(f.forecast):
                    forecast_per_sym[sidx]["x"].append(i + 1)
                    forecast_per_sym[sidx]["y"].append(float(f.forecast[sidx]))

    # Each panel always emits 6 traces in this exact order — even if empty —
    # so the initial figure and every animation frame have a consistent
    # shape and Plotly's slider can update them positionally.
    def _panel_traces_fixed(sidx: int, t_max: int) -> list:
        m = t_max + 1
        # 1) candlestick (clipped to t_max)
        cdl = go.Candlestick(
            x=x_all[:m], open=ohlc[:m, sidx, 0], high=ohlc[:m, sidx, 1],
            low=ohlc[:m, sidx, 2], close=ohlc[:m, sidx, 3],
            name=trace.symbols[sidx], showlegend=False,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
        # 2) buy fills
        bx = [v for v in fills_per_sym[sidx]["bx"] if v <= t_max]
        by = fills_per_sym[sidx]["by"][: len(bx)]
        buy = go.Scatter(
            x=bx or [], y=by or [], mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#26a69a",
                        line=dict(color="white", width=1.5)),
            name="buy", showlegend=False,
            hovertemplate="bar %{x}<br>BUY @ %{y:.2f}<extra></extra>",
        )
        # 3) sell/exit fills
        sx = [v for v in fills_per_sym[sidx]["sx"] if v <= t_max]
        sy = fills_per_sym[sidx]["sy"][: len(sx)]
        sell = go.Scatter(
            x=sx or [], y=sy or [], mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="#ef5350",
                        line=dict(color="white", width=1.5)),
            name="sell/exit", showlegend=False,
            hovertemplate="bar %{x}<br>EXIT/SHORT @ %{y:.2f}<extra></extra>",
        )
        # 4) round-trip path segments
        trade_x: list[int | None] = []
        trade_y: list[float | None] = []
        trade_text: list[str] = []
        for seg in trade_segments.get(sidx, []):
            if seg.entry_step > t_max:
                continue
            visible_exit_step = min(seg.exit_step, t_max)
            visible_exit_price = seg.exit_price if seg.exit_step <= t_max else float(ohlc[t_max, sidx, 3])
            trade_x.extend([seg.entry_step, visible_exit_step, None])
            trade_y.extend([seg.entry_price, visible_exit_price, None])
            side_label = "SHORT" if seg.is_short else "LONG"
            trade_text.extend(
                [
                    f"{side_label} entry {seg.entry_price:.2f}",
                    f"{side_label} exit {visible_exit_price:.2f}",
                    "",
                ]
            )
        trade_path = go.Scatter(
            x=trade_x or [],
            y=trade_y or [],
            mode="lines+markers",
            marker={"size": 7, "color": "#f5f5f5"},
            line={"color": "#9ec5fe", "width": 2},
            text=trade_text or [],
            name="trade path",
            showlegend=False,
            hovertemplate="bar %{x}<br>%{text}<extra></extra>",
        )
        # 5) order stars (gold)
        ox = [v for v in orders_per_sym[sidx]["x"] if v <= t_max]
        oy = orders_per_sym[sidx]["y"][: len(ox)]
        os_ = orders_per_sym[sidx]["side"][: len(ox)]
        order = go.Scatter(
            x=ox or [], y=oy or [], mode="markers",
            marker={"symbol": "star", "size": 22, "color": "#ffd400", "line": {"color": "black", "width": 1.2}},
            text=os_ or [], name="order", showlegend=False,
            hovertemplate="bar %{x}<br>ORDER %{text} @ %{y:.2f}<extra></extra>",
        )
        # 6) chronos2 forecast (next-bar markers)
        fx = [v for v in forecast_per_sym[sidx]["x"] if v <= t_max + 1]
        fy = forecast_per_sym[sidx]["y"][: len(fx)]
        forecast = go.Scatter(
            x=fx or [], y=fy or [], mode="markers+lines",
            marker={"symbol": "diamond-open", "size": 10, "color": "#42a5f5"},
            line={"color": "#42a5f5", "width": 1, "dash": "dash"},
            name="chronos2", showlegend=False,
            hovertemplate="bar %{x}<br>chronos2 @ %{y:.2f}<extra></extra>",
        )
        return [cdl, buy, sell, trade_path, order, forecast]

    def _equity_trace_fixed(t_max: int) -> go.Scatter:
        return go.Scatter(
            x=x_all[: t_max + 1], y=eq[: t_max + 1].tolist(),
            mode="lines", line=dict(color="#ce93d8", width=2.5),
            name="equity", showlegend=False,
            hovertemplate="bar %{x}<br>equity %{y:,.0f}<extra></extra>",
        )

    # Initial figure: t = T-1 (full series).
    panel_specs = []
    for k, sidx in enumerate(sym_idxs):
        r, c = divmod(k, cols)
        r += 1
        c += 1
        panel_specs.append((sidx, r, c))
        for tr_obj in _panel_traces_fixed(sidx, T - 1):
            fig.add_trace(tr_obj, row=r, col=c)
        for seg in trade_segments.get(sidx, []):
            shade_color = "rgba(38,166,154,0.14)" if not seg.is_short else "rgba(239,83,80,0.14)"
            fig.add_vrect(
                x0=seg.entry_step - 0.45,
                x1=seg.exit_step + 0.45,
                row=r,
                col=c,
                fillcolor=shade_color,
                opacity=1.0,
                line_width=0,
                layer="below",
            )
    fig.add_trace(_equity_trace_fixed(T - 1), row=rows_panels + 1, col=1)

    # Animation frames: identical trace count and order.
    frames = []
    if animated:
        for tt in range(T):
            data = []
            for (sidx, _r, _c) in panel_specs:
                data.extend(_panel_traces_fixed(sidx, tt))
            data.append(_equity_trace_fixed(tt))
            pnl_pct = (eq[tt] / initial_eq - 1.0) * 100.0 if initial_eq != 0 else 0.0
            sl = eq[: tt + 1]
            rets = np.zeros_like(sl)
            if sl.size > 1:
                rets[1:] = np.diff(sl) / np.maximum(np.abs(sl[:-1]), 1e-12)
            sortino = _rolling_sortino(rets)
            frames.append(go.Frame(
                data=data, name=str(tt),
                layout=go.Layout(title_text=f"{title}    bar {tt+1}/{T}    PnL {pnl_pct:+.2f}%    Sortino {sortino:.2f}"),
            ))
        fig.frames = frames

    slider_steps = [
        dict(method="animate", label=str(i),
             args=[[str(i)], dict(mode="immediate", frame=dict(duration=120, redraw=True),
                                  transition=dict(duration=0))])
        for i in range(T)
    ] if animated else []

    fig.update_layout(
        title=title, template="plotly_dark",
        height=320 * rows_panels + 280, width=max(1600, width_for_panels(cols)),
        margin=dict(t=80, l=50, r=30, b=70),
        hovermode="x unified",
        updatemenus=([dict(
            type="buttons", showactive=False,
            x=0.02, y=1.08, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )] if animated else None),
        sliders=([dict(
            active=T - 1, currentvalue=dict(prefix="bar: ", font=dict(size=14)),
            pad=dict(t=40), steps=slider_steps,
        )] if animated else None),
    )
    for r in range(1, rows_panels + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(rangeslider_visible=False, row=r, col=c)

    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_play=False)
    return out_path


def width_for_panels(cols: int) -> int:
    return 900 * max(1, int(cols)) + 200


def trace_from_portfolio_result(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    result,
    symbols: list[str] | None = None,
    predicted_close_col: str = "predicted_close_p50_h1",
) -> MarketsimTrace:
    """Convert a multi-symbol simulator run into a renderable marketsim trace."""
    bars_df = bars.copy()
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
    actions_df = actions.copy()
    if not actions_df.empty:
        actions_df["timestamp"] = pd.to_datetime(actions_df["timestamp"], utc=True)
        actions_df["symbol"] = actions_df["symbol"].astype(str).str.upper()
    bars_df["symbol"] = bars_df["symbol"].astype(str).str.upper()

    resolved_symbols = symbols or list(dict.fromkeys(bars_df["symbol"].tolist()))
    resolved_symbols = [str(symbol).upper() for symbol in resolved_symbols]
    timestamps = pd.DatetimeIndex(sorted(pd.unique(bars_df["timestamp"])))
    sym_index = pd.Index(resolved_symbols, name="symbol")
    full_index = pd.MultiIndex.from_product([timestamps, sym_index], names=["timestamp", "symbol"])
    dense = (
        bars_df.set_index(["timestamp", "symbol"])
        .sort_index()
        .reindex(full_index)
    )

    def _dense_column(name: str, *, fill_from: str | None = None) -> np.ndarray:
        if name in dense.columns:
            values = dense[name]
        elif fill_from is not None and fill_from in dense.columns:
            values = dense[fill_from]
        else:
            values = pd.Series(np.nan, index=full_index, dtype=np.float64)
        return values.to_numpy(dtype=np.float32).reshape(len(timestamps), len(resolved_symbols))

    close = _dense_column("close")
    ohlc = np.stack(
        [
            _dense_column("open", fill_from="close"),
            _dense_column("high", fill_from="close"),
            _dense_column("low", fill_from="close"),
            close,
        ],
        axis=-1,
    )
    trace = MarketsimTrace(symbols=resolved_symbols, prices=close, prices_ohlc=ohlc)

    equity_series = pd.Series(getattr(result, "equity_curve", pd.Series(dtype=float)))
    if not equity_series.empty:
        equity_series.index = pd.to_datetime(equity_series.index, utc=True)
        equity_series = equity_series.reindex(timestamps).ffill().bfill()
    else:
        initial_cash = float(getattr(getattr(result, "metrics", {}), "get", lambda *_: 1.0)("final_equity", 1.0))
        equity_series = pd.Series(np.full(len(timestamps), initial_cash, dtype=np.float64), index=timestamps)

    actions_by_ts = (
        actions_df.groupby("timestamp", sort=False) if not actions_df.empty else None
    )
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(resolved_symbols)}

    for step, timestamp in enumerate(timestamps):
        orders: list[OrderTick] = []
        forecast = np.full(len(resolved_symbols), np.nan, dtype=np.float32)
        long_target = np.full(len(resolved_symbols), np.nan, dtype=np.float32)
        short_target = np.full(len(resolved_symbols), np.nan, dtype=np.float32)

        if actions_by_ts is not None and timestamp in actions_by_ts.groups:
            rows = actions_by_ts.get_group(timestamp)
            for row in rows.itertuples(index=False):
                sym = str(row.symbol).upper()
                sym_idx = symbol_to_idx.get(sym)
                if sym_idx is None:
                    continue
                buy_price = getattr(row, "buy_price", None)
                sell_price = getattr(row, "sell_price", None)
                side = str(getattr(row, "side", "")).strip().lower()
                buy_amount = float(getattr(row, "buy_amount", 0.0) or 0.0)
                sell_amount = float(getattr(row, "sell_amount", 0.0) or 0.0)
                if predicted_close_col and hasattr(row, predicted_close_col):
                    value = getattr(row, predicted_close_col)
                    if value is not None and np.isfinite(float(value)):
                        forecast[sym_idx] = float(value)
                if side == "short" or sell_amount > 0.0:
                    if sell_price is not None and np.isfinite(float(sell_price)):
                        short_target[sym_idx] = float(sell_price)
                        orders.append(OrderTick(sym=sym_idx, price=float(sell_price), is_short=True))
                elif buy_amount > 0.0 and buy_price is not None and np.isfinite(float(buy_price)):
                    long_target[sym_idx] = float(buy_price)
                    orders.append(OrderTick(sym=sym_idx, price=float(buy_price), is_short=False))

        trace.record(
            step=step,
            action_id=0,
            position_sym=-1,
            position_is_short=False,
            equity=float(equity_series.iloc[step]),
            orders=orders,
            forecast=forecast if np.isfinite(forecast).any() else None,
            long_target=long_target if np.isfinite(long_target).any() else None,
            short_target=short_target if np.isfinite(short_target).any() else None,
        )

    open_entries: dict[str, tuple[int, float, bool]] = {}
    for trade in getattr(result, "trades", []):
        symbol = str(getattr(trade, "symbol", "")).upper()
        sym_idx = symbol_to_idx.get(symbol)
        if sym_idx is None:
            continue
        timestamp = pd.Timestamp(trade.timestamp)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        if timestamp not in timestamps:
            continue
        step = int(timestamps.get_loc(timestamp))
        price = float(getattr(trade, "price", 0.0))
        side = str(getattr(trade, "side", "")).strip().lower()
        if side in {"buy", "short_sell"}:
            open_entries[symbol] = (step, price, side == "short_sell")
            continue
        if side in {"sell", "buy_cover"} and symbol in open_entries:
            entry_step, entry_price, is_short = open_entries.pop(symbol)
            trace.trades.append(
                TradeSegment(
                    sym=sym_idx,
                    entry_step=entry_step,
                    entry_price=entry_price,
                    exit_step=step,
                    exit_price=price,
                    is_short=is_short,
                )
            )

    if len(timestamps) > 0:
        final_step = len(timestamps) - 1
        for symbol, (entry_step, entry_price, is_short) in open_entries.items():
            sym_idx = symbol_to_idx[symbol]
            trace.trades.append(
                TradeSegment(
                    sym=sym_idx,
                    entry_step=entry_step,
                    entry_price=entry_price,
                    exit_step=final_step,
                    exit_price=float(close[final_step, sym_idx]),
                    is_short=is_short,
                )
            )

    return trace


def make_recording_policy(
    base_policy_fn,
    trace: MarketsimTrace,
    *,
    forecast_extractor=None,
):
    """Wrap a `policy_fn(obs_np) -> int` so each call appends a frame to `trace`.

    Caller is expected to patch position/equity/orders into the appended frame
    after the simulator step (or use the higher-level prod-render script which
    does its own recording).
    """
    step_counter = {"i": 0}

    def wrapped(obs_np: np.ndarray) -> int:
        action_id = int(base_policy_fn(obs_np))
        forecast = None
        if forecast_extractor is not None:
            try:
                forecast = forecast_extractor(obs_np)
            except Exception:
                forecast = None
        trace.record(
            step=step_counter["i"],
            action_id=action_id,
            position_sym=-1,
            position_is_short=False,
            equity=float("nan"),
            forecast=forecast,
        )
        step_counter["i"] += 1
        return action_id

    return wrapped
