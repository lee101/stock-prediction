from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Iterable, Tuple, Union

import pandas as pd


def read_csv_tail(
    path: Union[str, Path],
    max_rows: int,
    *,
    chunksize: int = 100_000,
    return_total: bool = False,
    **kwargs: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
    """Read the tail of a CSV without loading the full file into memory.

    Args:
        path: CSV file path
        max_rows: Maximum number of rows to return from the end of the file
        chunksize: Number of rows per chunk
        return_total: If True, return (frame, total_rows_seen)
        **kwargs: Passed to pandas.read_csv (except chunksize)

    Returns:
        DataFrame containing up to max_rows from the end of the file.
        If return_total=True, also returns the total row count seen.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    buffer: Deque[pd.DataFrame] = deque()
    kept_rows = 0
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs):
        if chunk.empty:
            continue
        total_rows += len(chunk)
        buffer.append(chunk)
        kept_rows += len(chunk)

        while kept_rows > max_rows and buffer:
            drop = kept_rows - max_rows
            head = buffer[0]
            if len(head) <= drop:
                buffer.popleft()
                kept_rows -= len(head)
            else:
                buffer[0] = head.iloc[drop:].reset_index(drop=True)
                kept_rows -= drop

    if buffer:
        tail = pd.concat(list(buffer), ignore_index=True)
    else:
        tail = pd.DataFrame()

    if return_total:
        return tail, total_rows
    return tail
