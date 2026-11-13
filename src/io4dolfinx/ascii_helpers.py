# Copyright (C) 2026 Henrik Finsberg and Simula Research Laboratory
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""
Helpers for reading and writing ASCII files with a header followed by one row per line
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["read_count", "count_rows", "check_row_count", "read_rows", "write_rows"]


def read_count(path: Path) -> int:
    """Read the number on the first line of a file."""
    with open(path) as f:
        return int(f.readline().split()[0])


def count_rows(path: Path, header: int = 1) -> int:
    """Count the non-empty lines after the header of a file."""
    with open(path) as f:
        return sum(1 for line in f if line.strip()) - header


def check_row_count(path: Path, expected: int, header: int = 1):
    """Check that a file has ``expected`` non-empty lines after its header.

    Raises:
        ValueError: If the number of lines differs.
    """
    if (num_rows := count_rows(path, header)) != expected:
        raise ValueError(f"{path} has {num_rows} rows after its header, expected {expected}.")


def read_rows(
    path: Path, local_range: tuple[int, int] | list[int], num_columns: int, header: int = 1
) -> npt.NDArray[np.str_]:
    """Read a range of rows after the header of a file, as strings.

    Args:
        path: The file
        local_range: Range ``[start, end)`` of rows to read
        num_columns: Number of columns, only used to shape an empty range
        header: Number of header lines

    Returns:
        The rows, of shape ``(end - start, num_columns)``
    """
    start, end = int(local_range[0]), int(local_range[1])
    if end == start:
        return np.zeros((0, num_columns), dtype=np.str_)
    return np.loadtxt(path, dtype=np.str_, skiprows=header + start, max_rows=end - start, ndmin=2)


def write_rows(path: Path, header: list[str], rows: npt.NDArray, fmt: str):
    """Write header lines followed by one formatted line per row."""
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, rows, fmt=fmt)
