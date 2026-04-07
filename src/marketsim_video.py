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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class OrderTick:
    sym: int
    price: float  # the price level the policy is trying to execute at
    is_short: bool


@dataclass
class TraceFrame:
    step: int
    action_id: int
    position_sym: int  # -1 if flat
    position_is_short: bool
    equity: float
    orders: list[OrderTick] = field(default_factory=list)  # active orders this step
    forecast: Optional[np.ndarray] = None  # [S] predicted next-bar close
    forecast_ohlc: Optional[np.ndarray] = None  # [S, 4] predicted next-bar O,H,L,C
    long_target: Optional[np.ndarray] = None  # [S] price the policy would buy at this bar
    short_target: Optional[np.ndarray] = None  # [S] price the policy would short at this bar


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
    prices_ohlc: Optional[np.ndarray] = None  # [T, S, 4] O,H,L,C

    def record(
        self,
        step: int,
        action_id: int,
        position_sym: int,
        position_is_short: bool,
        equity: float,
        orders: Optional[list[OrderTick]] = None,
        forecast: Optional[np.ndarray] = None,
        forecast_ohlc: Optional[np.ndarray] = None,
        long_target: Optional[np.ndarray] = None,
        short_target: Optional[np.ndarray] = None,
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
        import json

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
        }
        path.write_text(json.dumps(payload))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "MarketsimTrace":
        import json

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
        return tr

    def num_symbols(self) -> int:
        return self.prices.shape[1]

    def num_steps(self) -> int:
        return len(self.frames)

    def active_symbol_indices(self, k: int) -> list[int]:
        """Union of (held at any step) ∪ (had a pending order at any step), capped at k.

        Sorted by activity (held + ordered count, descending), then by index
        for stable layout. If fewer than k active, pad with leading indices.
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
        if len(active) >= k:
            return sorted(active[:k])
        # Pad with first unused indices to keep grid full.
        used = set(active)
        for i in range(S):
            if len(active) >= k:
                break
            if i not in used:
                active.append(i)
        return sorted(active[:k])

    # Back-compat alias
    def select_symbol_indices(self, k: int) -> list[int]:
        return self.active_symbol_indices(k)


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
        o, h, l, c = float(ohlc[i, 0]), float(ohlc[i, 1]), float(ohlc[i, 2]), float(ohlc[i, 3])
        if not np.isfinite(h) or not np.isfinite(l):
            continue
        up = c >= o
        color = up_color if up else down_color
        ax.vlines(i, l, h, color=color, linewidth=1.0, zorder=2, alpha=alpha)
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
    o, h, l, c = float(ohlc4[0]), float(ohlc4[1]), float(ohlc4[2]), float(ohlc4[3])
    if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
        return
    half = 0.34
    ax.vlines(x, l, h, color=color, linewidth=2.2, zorder=2.5, alpha=0.95)
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
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5)
        return "h264_nvenc" in (out.stdout + out.stderr)
    except Exception:
        return False


def _open_video_writer(out_path: Path, *, fps: int, nvenc: bool):
    """Open an imageio writer; prefer NVENC when available, fall back to libx264."""
    import imageio.v2 as imageio

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
) -> Path:
    """Render the recorded trace to an MP4. Returns the output path.

    `frames_per_bar` repeats each simulator bar across N video frames so playback
    feels closer to a moving market instead of a stop-motion update per bar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if trace.num_steps() == 0:
        raise ValueError("MarketsimTrace has no recorded frames")

    sym_idxs = trace.active_symbol_indices(num_pairs)
    n_panels = max(1, len(sym_idxs))
    rows, cols = _grid_shape(n_panels)

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
    fees_step = np.zeros(T, dtype=np.float64)
    _ps, _psh = -1, False
    for i in range(T):
        cs, csh = int(pos_sym[i]), bool(pos_short[i])
        if (cs, csh) != (_ps, _psh):
            n_fills = (1 if _ps >= 0 else 0) + (1 if cs >= 0 else 0)
            if n_fills > 0:
                fees_step[i] = fee_rate * float(equity[i]) * n_fills
            _ps, _psh = cs, csh
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

    # Trade event lists per panel
    buys: dict[int, list[tuple[int, float]]] = {s: [] for s in sym_idxs}
    sells: dict[int, list[tuple[int, float]]] = {s: [] for s in sym_idxs}
    _ps, _psh = -1, False
    for i in range(T):
        cs, csh = int(pos_sym[i]), bool(pos_short[i])
        if (cs, csh) != (_ps, _psh):
            if _ps in buys:
                sells[_ps].append((i, float(close[i, _ps])))
            if cs in buys:
                if csh:
                    sells[cs].append((i, float(close[i, cs])))
                else:
                    buys[cs].append((i, float(close[i, cs])))
            _ps, _psh = cs, csh

    # Active-order overlays per panel: list of (step, price, is_short)
    order_ticks: dict[int, list[tuple[int, float, bool]]] = {s: [] for s in sym_idxs}
    for i, fr in enumerate(trace.frames):
        for od in fr.orders:
            if od.sym in order_ticks:
                order_ticks[od.sym].append((i, float(od.price), bool(od.is_short)))

    # 1080p by default. figsize in inches × dpi → pixels.
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="#0e1117")
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[3] * rows + [1.3], hspace=0.55, wspace=0.28,
                          left=0.04, right=0.985, top=0.93, bottom=0.06)
    ax_panels = []
    for k, sidx in enumerate(sym_idxs):
        r, c = divmod(k, cols)
        ax_panels.append((sidx, fig.add_subplot(gs[r, c])))
    ax_eq = fig.add_subplot(gs[rows, :])

    writer = _open_video_writer(out_path, fps=fps, nvenc=nvenc)

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

    try:
        total_video_frames = T * max(1, int(frames_per_bar))
        for vf in range(total_video_frames):
            t = vf // max(1, int(frames_per_bar))
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

                # Held shading
                in_run = False
                run_start = 0
                for i in range(t + 1):
                    if pos_sym[i] == sidx:
                        if not in_run:
                            in_run = True
                            run_start = i
                    elif in_run:
                        c_run = SHORT_RED if pos_short[run_start] else LONG_GREEN
                        ax.axvspan(run_start - 0.4, i - 1 + 0.4, alpha=0.13, color=c_run)
                        in_run = False
                if in_run:
                    c_run = SHORT_RED if pos_short[run_start] else LONG_GREEN
                    ax.axvspan(run_start - 0.4, t + 0.4, alpha=0.13, color=c_run)

                # Active-order overlay: vertical gold band + bold marker on
                # the current bar so it's unmistakable that the policy is
                # placing an order here.
                for (oi, oprice, oshort) in order_ticks[sidx]:
                    if oi == t:
                        ax.axvspan(oi - 0.45, oi + 0.45, color=GOLD, alpha=0.18, zorder=1.5)
                        ax.scatter([oi], [oprice], marker="D", color=GOLD,
                                   edgecolor="black", linewidth=1.0, s=140, zorder=6)
                        ax.annotate(
                            "ORDER " + ("SHORT" if oshort else "LONG"),
                            xy=(oi, oprice),
                            xytext=(6, 8), textcoords="offset points",
                            color=GOLD, fontsize=9, fontweight="bold",
                            zorder=7,
                        )

                # Fill markers (executed buys/sells) at the close price of the fill bar
                for (bi, bp) in buys[sidx]:
                    if bi <= t:
                        ax.scatter([bi], [bp], marker="^", color=LONG_GREEN,
                                   edgecolor="white", linewidth=0.6, s=75, zorder=5)
                for (si, sp) in sells[sidx]:
                    if si <= t:
                        ax.scatter([si], [sp], marker="v", color=SHORT_RED,
                                   edgecolor="white", linewidth=0.6, s=75, zorder=5)

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
            writer.append_data(buf[..., :3])
    finally:
        writer.close()
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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

    pos_sym = np.asarray([f.position_sym for f in trace.frames], dtype=np.int64)
    pos_short = np.asarray([f.position_is_short for f in trace.frames], dtype=bool)

    # Precompute buy/sell event lists per symbol once.
    fills_per_sym: dict[int, dict[str, list]] = {s: {"bx": [], "by": [], "sx": [], "sy": []} for s in sym_idxs}
    prev, prev_short = -1, False
    for i, f in enumerate(trace.frames):
        cs, csh = f.position_sym, f.position_is_short
        if (cs, csh) != (prev, prev_short):
            if prev in fills_per_sym:
                fills_per_sym[prev]["sx"].append(i)
                fills_per_sym[prev]["sy"].append(float(ohlc[i, prev, 3]))
            if cs in fills_per_sym:
                if csh:
                    fills_per_sym[cs]["sx"].append(i)
                    fills_per_sym[cs]["sy"].append(float(ohlc[i, cs, 3]))
                else:
                    fills_per_sym[cs]["bx"].append(i)
                    fills_per_sym[cs]["by"].append(float(ohlc[i, cs, 3]))
            prev, prev_short = cs, csh

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

    def _panel_traces(sidx: int, t_max: int, r: int, c: int) -> list:
        """Build the per-symbol trace list clipped to t_max."""
        m = t_max + 1
        tlist = []
        tlist.append(go.Candlestick(
            x=x_all[:m], open=ohlc[:m, sidx, 0], high=ohlc[:m, sidx, 1],
            low=ohlc[:m, sidx, 2], close=ohlc[:m, sidx, 3],
            name=trace.symbols[sidx], showlegend=False,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            xaxis=f"x{(r - 1) * cols + c}" if (r, c) != (1, 1) else "x",
            yaxis=f"y{(r - 1) * cols + c}" if (r, c) != (1, 1) else "y",
        ))
        bx = [v for v in fills_per_sym[sidx]["bx"] if v <= t_max]
        by = fills_per_sym[sidx]["by"][: len(bx)]
        if bx:
            tlist.append(go.Scatter(
                x=bx, y=by, mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#26a69a",
                            line=dict(color="white", width=1.5)),
                name="fill long", hovertemplate="bar %{x}<br>fill long @ %{y:.2f}<extra></extra>",
                showlegend=False,
                xaxis=f"x{(r - 1) * cols + c}" if (r, c) != (1, 1) else "x",
                yaxis=f"y{(r - 1) * cols + c}" if (r, c) != (1, 1) else "y",
            ))
        sx = [v for v in fills_per_sym[sidx]["sx"] if v <= t_max]
        sy = fills_per_sym[sidx]["sy"][: len(sx)]
        if sx:
            tlist.append(go.Scatter(
                x=sx, y=sy, mode="markers",
                marker=dict(symbol="triangle-down", size=14, color="#ef5350",
                            line=dict(color="white", width=1.5)),
                name="fill short/exit", hovertemplate="bar %{x}<br>exit/short @ %{y:.2f}<extra></extra>",
                showlegend=False,
                xaxis=f"x{(r - 1) * cols + c}" if (r, c) != (1, 1) else "x",
                yaxis=f"y{(r - 1) * cols + c}" if (r, c) != (1, 1) else "y",
            ))
        ox = [v for v in orders_per_sym[sidx]["x"] if v <= t_max]
        oy = orders_per_sym[sidx]["y"][: len(ox)]
        os = orders_per_sym[sidx]["side"][: len(ox)]
        if ox:
            tlist.append(go.Scatter(
                x=ox, y=oy, mode="markers",
                marker=dict(symbol="diamond", size=18, color="#ffd400",
                            line=dict(color="black", width=1.2)),
                name="order", text=os,
                hovertemplate="bar %{x}<br>ORDER %{text} @ %{y:.2f}<extra></extra>",
                showlegend=False,
                xaxis=f"x{(r - 1) * cols + c}" if (r, c) != (1, 1) else "x",
                yaxis=f"y{(r - 1) * cols + c}" if (r, c) != (1, 1) else "y",
            ))
        fx = [v for v in forecast_per_sym[sidx]["x"] if v <= t_max + 1]
        fy = forecast_per_sym[sidx]["y"][: len(fx)]
        if fx:
            tlist.append(go.Scatter(
                x=fx, y=fy, mode="markers+lines",
                marker=dict(symbol="diamond-open", size=10, color="#42a5f5",
                            line=dict(color="#42a5f5", width=2)),
                line=dict(color="#42a5f5", width=1, dash="dash"),
                name="chronos2 forecast",
                hovertemplate="bar %{x}<br>forecast @ %{y:.2f}<extra></extra>",
                showlegend=False,
                xaxis=f"x{(r - 1) * cols + c}" if (r, c) != (1, 1) else "x",
                yaxis=f"y{(r - 1) * cols + c}" if (r, c) != (1, 1) else "y",
            ))
        return tlist

    def _equity_trace(t_max: int) -> go.Scatter:
        n_eq_xy = (rows_panels) * cols + 1
        return go.Scatter(
            x=x_all[: t_max + 1], y=eq[: t_max + 1],
            mode="lines", line=dict(color="#ce93d8", width=2.5),
            name="equity", showlegend=False,
            hovertemplate="bar %{x}<br>equity %{y:,.0f}<extra></extra>",
            xaxis=f"x{n_eq_xy}", yaxis=f"y{n_eq_xy}",
        )

    # Initial frame data (full series so the user sees everything when they
    # land on the page; the slider then walks through bar-by-bar via frames).
    init_traces = []
    panel_specs = []
    for k, sidx in enumerate(sym_idxs):
        r, c = divmod(k, cols)
        r += 1
        c += 1
        panel_specs.append((sidx, r, c))
        for tr in _panel_traces(sidx, T - 1, r, c):
            fig.add_trace(tr, row=r, col=c)
            init_traces.append(tr)
    fig.add_trace(_equity_trace(T - 1), row=rows_panels + 1, col=1)

    # Build animation frames if requested.
    frames = []
    if animated:
        for tt in range(T):
            data = []
            for (sidx, r, c) in panel_specs:
                data.extend(_panel_traces(sidx, tt, r, c))
            data.append(_equity_trace(tt))
            pnl_pct = (eq[tt] / initial_eq - 1.0) * 100.0 if initial_eq != 0 else 0.0
            frames.append(go.Frame(
                data=data, name=str(tt),
                layout=go.Layout(title_text=f"{title}    bar {tt+1}/{T}    PnL {pnl_pct:+.2f}%"),
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
