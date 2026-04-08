#!/usr/bin/env python3
"""Compatibility import surface for the smart test runner."""

from scripts._smart_test_runner import (
    _is_project_test_file,
    get_changed_files,
    main,
    map_file_to_tests,
    prioritize_tests,
    run_tests,
)


__all__ = [
    "_is_project_test_file",
    "get_changed_files",
    "main",
    "map_file_to_tests",
    "prioritize_tests",
    "run_tests",
]


if __name__ == "__main__":
    main()
