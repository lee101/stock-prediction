from __future__ import annotations

"""Compatibility wrapper for autoresearch stock task preparation."""

import sys

import src.autoresearch_stock.prepare as _impl


if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
