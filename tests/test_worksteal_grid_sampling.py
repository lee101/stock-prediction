"""Tests for Cartesian-product grid sampling helpers."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.grid_sampling import (
    cartesian_product_size,
    combo_from_cartesian_index,
    sample_cartesian_product,
)


class TestCartesianProductSize:
    def test_matches_itertools_product_count(self):
        value_sets = [
            [1, 2],
            ["a", "b", "c"],
            [True, False],
        ]

        total = cartesian_product_size(value_sets)

        assert total == len(list(itertools.product(*value_sets)))


class TestComboFromCartesianIndex:
    def test_matches_itertools_product_order(self):
        value_sets = [
            [1, 2],
            ["a", "b", "c"],
            [True, False],
        ]
        expected = list(itertools.product(*value_sets))

        actual = [
            combo_from_cartesian_index(value_sets, index)
            for index in range(len(expected))
        ]

        assert actual == expected


class TestSampleCartesianProduct:
    def test_sampling_is_reproducible(self):
        value_sets = [
            list(range(10)),
            list("abc"),
            [True, False],
        ]

        total_1, combos_1 = sample_cartesian_product(value_sets, max_trials=7, seed=42)
        total_2, combos_2 = sample_cartesian_product(value_sets, max_trials=7, seed=42)

        assert total_1 == total_2 == 60
        assert combos_1 == combos_2

    def test_sampled_path_avoids_materializing_full_product(self, monkeypatch):
        value_sets = [
            list(range(1000)),
            list(range(1000)),
            list(range(1000)),
        ]

        def fail_product(*args, **kwargs):
            raise AssertionError("sampled path should not call itertools.product")

        monkeypatch.setattr("binance_worksteal.grid_sampling.itertools.product", fail_product)

        total, combos = sample_cartesian_product(value_sets, max_trials=5, seed=42)

        assert total == 1_000_000_000
        assert len(combos) == 5
