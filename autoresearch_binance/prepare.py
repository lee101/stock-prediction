from __future__ import annotations

"""Compatibility wrapper for autoresearch Binance task preparation."""

import sys

import src.autoresearch_binance.prepare as _impl


if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
