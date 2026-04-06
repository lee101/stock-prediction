from __future__ import annotations

"""Compatibility wrapper for the autoresearch Binance agent scheduler."""

import sys

import src.autoresearch_binance.agent_scheduler as _impl


if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
