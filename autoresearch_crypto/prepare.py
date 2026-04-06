from __future__ import annotations

"""Compatibility wrapper for autoresearch crypto task preparation."""

import sys

import src.autoresearch_crypto.prepare as _impl


sys.modules[__name__] = _impl
