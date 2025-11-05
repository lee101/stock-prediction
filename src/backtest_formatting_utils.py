"""Formatting utilities for backtest output and logging."""

from typing import List, Optional


def fmt_number(value: Optional[float], precision: int = 4) -> str:
    """Format a number with specified precision, or "-" if None.

    Args:
        value: Number to format
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def format_table(headers: List[str], rows: List[List[str]], indent: str = "  ") -> str:
    """Format a table with aligned columns.

    Args:
        headers: Column headers
        rows: Table rows
        indent: String to prefix each line

    Returns:
        Formatted table as string
    """
    if not rows:
        return ""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = indent + " ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    separator_line = indent + " ".join("-" * widths[idx] for idx in range(len(headers)))
    row_lines = [
        indent + " ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def log_table(title: str, headers: List[str], rows: List[List[str]], logger) -> None:
    """Log a formatted table with title.

    Args:
        title: Title for the table
        headers: Column headers
        rows: Table rows
        logger: Logger instance to use
    """
    table = format_table(headers, rows)
    if table:
        logger.info(f"\n{title}:\n{table}")
