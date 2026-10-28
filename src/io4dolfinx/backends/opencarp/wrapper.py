# Copyright (C) 2026 Henrik Finsberg
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Write a mesh, its markers and fibers to openCARP in one call."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dolfinx

from ...checkpointing import write_cell_data, write_meshtags
from ...original_checkpoint import write_mesh_input_order
from .. import FileMode

__all__ = ["write_opencarp"]
logger = logging.getLogger(__name__)


def _check_direction(u: dolfinx.fem.Function, name: str):
    """Check that a fiber or sheet field can be interpolated into cell midpoints."""
    if u.function_space.ufl_element().family_name == "quadrature":
        raise ValueError(
            f"Cannot write the {name} field {u.name!r} from a quadrature space to openCARP, as it "
            "cannot be interpolated into DG-0. Generate the fibers in a P-1 or DG-0 space."
        )
    gdim = u.function_space.mesh.geometry.dim
    if u.ufl_shape not in ((gdim,), (3,)):
        raise ValueError(
            f"Expected the {name} field to be a {gdim}D or 3D vector, got shape {u.ufl_shape}."
        )


def write_opencarp(
    filename: Path | str,
    mesh: dolfinx.mesh.Mesh,
    scale: float,
    cell_tags: dolfinx.mesh.MeshTags | None = None,
    facet_tags: dolfinx.mesh.MeshTags | None = None,
    fiber: dolfinx.fem.Function | None = None,
    sheet: dolfinx.fem.Function | None = None,
    markers: dict[str, Any] | None = None,
    backend_args: dict[str, Any] | None = None,
):
    """Write a linear mesh with markers and fibers to openCARP files.

    Writes ``<base>.pts`` and ``<base>.elem``, with the region tag of each element from
    ``cell_tags``; ``<base>.lon`` with the fiber (and sheet) direction of each element;
    and ``<base>_<marker>.vtx`` and ``<base>_<marker>.surf`` for each value of
    ``facet_tags`` (``.surf`` only for 3D meshes). Nodes and elements are in the
    ordering of the input mesh, so the files do not depend on the number of processes. See
    :mod:`io4dolfinx.backends.opencarp.backend` for the format.

    Args:
        filename: Base name of the files
        mesh: The mesh
        scale: Factor converting mesh coordinates to µm, e.g. ``1000.0`` for a mesh in mm
        cell_tags: Region tag of each cell. Untagged cells get
            ``backend_args["default_tag"]`` (``0`` by default).
        facet_tags: Facet markers, e.g. for stimuli and boundary conditions
        fiber: Fiber direction, interpolated into cell midpoints and normalised
        sheet: Sheet direction, written next to the fibers
        markers: Map from marker name to value, or to ``(value, dim)``, naming the
            files of each facet marker
        backend_args: Further arguments to the openCARP backend
    """
    if sheet is not None and fiber is None:
        raise ValueError("Cannot write a sheet direction without a fiber direction.")
    for u, name in ((fiber, "fiber"), (sheet, "sheet")):
        if u is not None:
            _check_direction(u, name)

    args = dict(backend_args or {})
    args["scale"] = scale
    if markers is not None:
        args["markers"] = markers

    logger.debug(f"Writing openCARP mesh to {filename} with arguments {args}")
    write_mesh_input_order(filename, mesh, backend="opencarp", backend_args=args)
    for tags in (cell_tags, facet_tags):
        if tags is not None:
            write_meshtags(
                filename, mesh, tags, backend="opencarp", backend_args=args, on_input_mesh=True
            )
    for u, mode in ((fiber, FileMode.write), (sheet, FileMode.append)):
        if u is not None:
            write_cell_data(
                filename,
                u,
                time=0.0,
                mode=mode,
                backend_args=args,
                backend="opencarp",
                on_input_mesh=True,
            )
