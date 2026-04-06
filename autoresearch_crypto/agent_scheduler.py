from __future__ import annotations

"""Compatibility wrapper for the autoresearch crypto agent scheduler."""

import sys

import src.autoresearch_crypto.agent_scheduler as _impl


if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
