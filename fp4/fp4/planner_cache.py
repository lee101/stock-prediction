"""Disk-backed SQLite cache for daily LLM planner calls.

Keyed by (date_iso, portfolio_hash, universe_hash). Avoids re-billing the
Gemini API for the same trading day / portfolio state.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_CACHE_PATH = Path(os.path.expanduser("~/.cache/fp4_gemini_planner.sqlite"))


def _hash_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class PlannerCache:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                key TEXT PRIMARY KEY,
                date_iso TEXT NOT NULL,
                portfolio_hash TEXT NOT NULL,
                universe_hash TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                created_ts REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(date_iso: str, portfolio_state: Any, universe: Any) -> tuple[str, str, str, str]:
        ph = _hash_obj(portfolio_state)
        uh = _hash_obj(universe)
        key = f"{date_iso}:{ph}:{uh}"
        return key, date_iso, ph, uh

    def get(self, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT plan_json FROM plans WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(self, key: str, date_iso: str, portfolio_hash: str, universe_hash: str, plan: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO plans (key, date_iso, portfolio_hash, universe_hash, plan_json, created_ts) VALUES (?,?,?,?,?,?)",
            (key, date_iso, portfolio_hash, universe_hash, json.dumps(plan), time.time()),
        )
        self._conn.commit()

    def get_or_compute(
        self,
        date_iso: str,
        portfolio_state: Any,
        universe: Any,
        compute_fn: Callable[[], dict],
    ) -> dict:
        key, d, ph, uh = self.make_key(date_iso, portfolio_state, universe)
        cached = self.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        plan = compute_fn()
        self.put(key, d, ph, uh, plan)
        return plan

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        self._conn.close()
