# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

from pathlib import Path
from typing import Any

import dolfinx

from .backends import FileMode, get_backend

__all__ = [
    "snapshot_checkpoint",
]


def snapshot_checkpoint(
    uh: dolfinx.fem.Function,
    file: Path,
    mode: FileMode,
    backend_args: dict[str, Any] | None = None,
    backend: str = "adios2",
):
    """Read or write a snapshot checkpoint

    This checkpoint is only meant to be used on the same mesh during the same simulation.

    :param uh: The function to write data from or read to
    :param file: The file to write to or read from
    :param mode: Either read or write
    """

    backend_cls = get_backend(backend)
    default_args = backend_cls.get_default_backend_args(backend_args)
    if mode not in [FileMode.write, FileMode.read]:
        raise ValueError(f"Got invalid mode {mode}")
    backend_cls.snapshot_checkpoint(file, mode, uh, default_args)
