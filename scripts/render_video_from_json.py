#!/usr/bin/env python
"""Render an MP4 (and optional Plotly HTML) from a marketsim trace JSON.

The JSON schema (`marketsim_trace_v1`) is what `MarketsimTrace.to_json()`
emits. C code in pufferlib_market can write the same shape directly so the
training loop never has to import matplotlib.

Usage:
    python scripts/render_video_from_json.py trace.json --out video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.marketsim_video import MarketsimTrace, render_mp4, render_html_plotly


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("trace_json", type=str)
    p.add_argument("--out", type=str, required=True, help="Output .mp4 path")
    p.add_argument("--num-pairs", type=int, default=4)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--frames-per-bar", type=int, default=2)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--periods-per-year", type=float, default=252.0)
    p.add_argument("--title", type=str, default="marketsim")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--html", type=str, default=None,
                   help="Also write an interactive Plotly HTML to this path")
    args = p.parse_args()

    trace = MarketsimTrace.from_json(args.trace_json)
    render_mp4(
        trace, args.out,
        num_pairs=args.num_pairs, fps=args.fps,
        frames_per_bar=args.frames_per_bar,
        fee_rate=args.fee_rate, periods_per_year=args.periods_per_year,
        title=args.title, width_px=args.width, height_px=args.height,
    )
    print(f"wrote {args.out}")
    if args.html:
        render_html_plotly(trace, args.html, num_pairs=args.num_pairs, title=args.title)
        print(f"wrote {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
