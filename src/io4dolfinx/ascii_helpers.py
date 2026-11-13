# Copyright (C) 2026 Henrik Finsberg and Simula Research Laboratory
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""
Helpers for reading and writing ASCII files with a header followed by one row per line
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["read_count", "check_row_count", "read_rows", "write_rows"]


def read_count(path: Path) -> int:
    """Read the number on the first line of a file."""
    with open(path) as f:
        return int(f.readline().split()[0])


def check_row_count(path: Path, expected: int, header: int = 1):
    """Check that a file has ``expected`` non-empty lines after its header.

    Raises:
        ValueError: If the number of lines differs.
    """
    with open(path) as f:
        num_rows = sum(1 for line in f if line.strip()) - header
    if num_rows != expected:
        raise ValueError(f"{path} has {num_rows} rows after its header, expected {expected}.")


def read_rows(
    path: Path,
    num_columns: int,
    local_range: tuple[int, int] | list[int] | None = None,
    header: int = 1,
    dtype: npt.DTypeLike = np.str_,
    usecols: int | Sequence[int] | None = None,
) -> npt.NDArray:
    """Read rows after the header of a file.

    Args:
        path: The file
        num_columns: Number of columns read, used to shape an empty result
        local_range: Range ``[start, end)`` of rows to read, all rows if ``None``
        header: Number of header lines
        dtype: Type of the values. Read numeric columns with a numeric type: as
            strings, every value takes several times the memory.
        usecols: Columns to read, all if ``None``

    Returns:
        The rows, of shape ``(num_rows, num_columns)``
    """
    start = 0 if local_range is None else int(local_range[0])
    max_rows = None if local_range is None else int(local_range[1]) - start
    if max_rows == 0:
        return np.zeros((0, num_columns), dtype=dtype)
    with warnings.catch_warnings():
        # A file without rows is valid, e.g. a marker without entities
        warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
        rows = np.loadtxt(
            path,
            dtype=dtype,
            skiprows=header + start,
            max_rows=max_rows,
            ndmin=2,
            usecols=usecols,
        )
    return rows.reshape(0, num_columns) if rows.size == 0 else rows


def write_rows(path: Path, header: list[str], rows: npt.NDArray, fmt: str):
    """Write header lines followed by one formatted line per row."""
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, rows, fmt=fmt)
