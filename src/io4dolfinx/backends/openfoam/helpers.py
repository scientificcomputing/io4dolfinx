"""
Helpers for the OpenFOAM backend: reading a case through VTK, merging regions, and
reading the boundary patches and cell zones of ``polyMesh`` files.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken and Simula Research Laboratory
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, NamedTuple

import dolfinx
import numpy as np
import numpy.typing as npt

try:
    import pyvista
except ImportError:
    raise ModuleNotFoundError("The openfoam-backend requires pyvista")

from ..vtk_cell_types import _first_order_vtk

__all__ = [
    "Case",
    "Patch",
    "read_case",
    "get_regions",
    "read_boundary",
    "read_faces",
    "read_zones",
    "field_exists",
]

#: VTK cell type of OpenFOAM polyhedra, which DOLFINx cannot represent
VTK_POLYHEDRON = 42


class Patch(NamedTuple):
    """A boundary patch of a region."""

    name: str  #: The patch name, ``region/patch`` in a case with several regions
    faces: list[npt.NDArray[np.int64]]  #: Each face, as merged point indices
    entries: dict[str, str]  #: The entries of the patch in the ``boundary`` file


@dataclass
class Case:
    """An OpenFOAM case read through VTK: its internal mesh and the fields asked for, with
    the regions merged into one mesh."""

    path: Path  #: The case directory
    regions: list[str]  #: Region names, ``""`` for the default region
    points: npt.NDArray[np.floating]  #: Points, merged over regions
    #: Cells by DOLFINx cell type, as merged point indices in DOLFINx node order. The cells
    #: of a type are in region order, and then in OpenFOAM cell order.
    cells: dict[str, npt.NDArray[np.int64]]
    #: For each region, the merged index of each of its points, by OpenFOAM point label
    node_maps: list[npt.NDArray[np.int64]]
    #: The index of the first cell of each region, and the number of cells. For one cell
    #: type, cell ``i`` of a region is merged cell ``cell_offsets[region] + i``.
    cell_offsets: npt.NDArray[np.int64]
    #: The cell fields asked for, of shape ``(num_cells, num_components)``
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    #: The point fields asked for, of shape ``(num_points, num_components)``
    point_data: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def cell_types(self) -> list[str]:
        """The DOLFINx cell types of the mesh, sorted."""
        return sorted(self.cells, key=lambda c: dolfinx.mesh.to_type(c).value)

    def get_single_type_cells(self, what: str) -> npt.NDArray[np.int64]:
        """The cells of a mesh of one cell type.

        Raises:
            NotImplementedError: For a mesh with several cell types.
        """
        if len(types := self.cell_types) > 1:
            raise NotImplementedError(
                f"{what} is not supported on meshes with several cell types {types}."
            )
        return self.cells[types[0]]

    def get_polymesh(self, region: str) -> Path:
        """The ``polyMesh`` directory of a region."""
        return self.path / "constant" / region / "polyMesh"

    def _get_name(self, region: str, name: str) -> str:
        """The name of a patch or zone, qualified by its region if there are several."""
        return name if len(self.regions) == 1 else f"{region}/{name}"

    @cached_property
    def _faces(self) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """The faces of each region, as offsets into OpenFOAM point labels."""
        return [read_faces(self.get_polymesh(region)) for region in self.regions]

    def _get_region_faces(
        self, i: int, face_labels: npt.NDArray[np.int64]
    ) -> list[npt.NDArray[np.int64]]:
        """Faces of region ``i``, by face label, as merged point indices."""
        offsets, labels = self._faces[i]
        return [
            self.node_maps[i][labels[offsets[f] : offsets[f + 1]]] for f in face_labels.tolist()
        ]

    @cached_property
    def patches(self) -> list[Patch]:
        """The boundary patches of each region, in region order and then as in each
        ``boundary`` file, read by their labels from the ``polyMesh`` files.

        Raises:
            ValueError: If a patch mapped to a patch of another region does not coincide
                with it, so that the regions were not merged where they meet.
        """
        patches = []
        for i, region in enumerate(self.regions):
            for entries in read_boundary(self.get_polymesh(region)):
                start, num = int(entries["startFace"]), int(entries["nFaces"])
                faces = self._get_region_faces(i, np.arange(start, start + num, dtype=np.int64))
                patches.append(Patch(self._get_name(region, entries["name"]), faces, entries))
        by_name = {p.name: {tuple(sorted(f)) for f in p.faces} for p in patches}
        for name, other in self.get_interfaces(patches):
            if by_name[name] != by_name[other]:
                raise ValueError(
                    f"The faces of patch {name} do not coincide with those of {other} after "
                    "merging the regions, whose points must match exactly."
                )
        return patches

    def get_interfaces(self, patches: list[Patch]) -> list[tuple[str, str]]:
        """The pairs of patches between regions: patches mapped to a patch of another
        region (``sampleRegion``, ``samplePatch``), which hold the same faces after the
        regions are merged."""
        if len(self.regions) == 1:
            return []
        names = {p.name for p in patches}
        pairs = []
        for patch in patches:
            region = patch.name.split("/")[0]
            other = patch.entries.get("sampleRegion")
            sample = patch.entries.get("samplePatch")
            if other is not None and other != region and f"{other}/{sample}" in names:
                pairs.append((patch.name, f"{other}/{sample}"))
        return pairs

    def _get_zones(
        self, kind: str, convert: Callable[[int, npt.NDArray[np.int64]], Any]
    ) -> dict[str, Any]:
        """The zones of a kind (``"cell"``, ``"face"``, ``"point"``) of each region, by
        name, with ``convert(region index, labels)`` applied to the labels of each."""
        return {
            self._get_name(region, name): convert(i, labels)
            for i, region in enumerate(self.regions)
            for name, labels in read_zones(self.get_polymesh(region), kind).items()
        }

    @cached_property
    def cell_zones(self) -> dict[str, npt.NDArray[np.int64]]:
        """The cells of each ``cellZone`` of each region, as merged cell indices."""
        return self._get_zones("cell", lambda i, labels: self.cell_offsets[i] + labels)

    @cached_property
    def face_zones(self) -> dict[str, list[npt.NDArray[np.int64]]]:
        """The faces of each ``faceZone`` of each region, as merged point indices."""
        return self._get_zones("face", self._get_region_faces)

    @cached_property
    def point_zones(self) -> dict[str, npt.NDArray[np.int64]]:
        """The points of each ``pointZone`` of each region, as merged point indices."""
        return self._get_zones("point", lambda i, labels: self.node_maps[i][labels])


def _as_rows(values: npt.ArrayLike) -> np.ndarray:
    """Field values as an array of shape ``(num_values, num_components)``."""
    values = np.asarray(values)
    return values.reshape(len(values), -1)


def _time_value(reader: pyvista.POpenFOAMReader, time: float | str | None) -> float | None:
    """The time step of the case to read; the first one if ``time`` is ``None``."""
    times = list(reader.time_values)
    if len(times) == 0:
        return None
    if time is None:
        return times[0]
    matches = [t for t in times if np.isclose(t, float(time))]
    if len(matches) != 1:
        raise KeyError(f"No time step {time} in {reader.path}. Available times: {times}")
    return matches[0]


def _grid_cells(
    grid: pyvista.UnstructuredGrid, node_map: npt.NDArray[np.int64]
) -> dict[str, npt.NDArray[np.int64]]:
    """The cells of a grid by DOLFINx cell type, as merged point indices in DOLFINx order."""
    if (num_polyhedra := np.count_nonzero(grid.celltypes == VTK_POLYHEDRON)) > 0:
        raise NotImplementedError(
            f"The mesh has {num_polyhedra} polyhedral cells, which DOLFINx does not support."
        )
    cells: dict[str, npt.NDArray[np.int64]] = {}
    for vtk_type, vtk_nodes in grid.cells_dict.items():
        nodes = np.asarray(vtk_nodes, dtype=np.int64)
        if int(vtk_type) not in _first_order_vtk:
            raise NotImplementedError(f"Unsupported VTK cell type {int(vtk_type)}.")
        cell_type = _first_order_vtk[int(vtk_type)]
        perm = dolfinx.cpp.io.perm_vtk(dolfinx.mesh.to_type(cell_type), nodes.shape[1])
        cells[cell_type] = node_map[nodes[:, perm]]
    return cells


def get_regions(reader: pyvista.POpenFOAMReader, region: str | None) -> list[str]:
    """The regions of a case to read, from the names of the meshes VTK can read.

    Args:
        reader: VTK reader of the case
        region: Region to read, ``""`` for the default region, all regions if ``None``

    Returns:
        The region names, ``""`` for the default region, in the order VTK reads them
    """
    names = reader.patch_array_names
    # The default region (constant/polyMesh) is called "", so that a case without regions
    # is read like a case with one region
    named = sorted({n.split("/")[1] for n in names if n.startswith("/")})
    available = ([""] if "internalMesh" in names else []) + named
    if region is None:
        # Without a choice, read the named regions and leave out the default region: in a
        # multi-region case it usually holds the mesh from before it was split into regions
        return named if named else available
    if region not in available:
        raise KeyError(f"No region '{region}' in {reader.path}. Regions: {available}")
    return [region]


def read_case(
    filename: Path | str,
    time: float | str | None,
    region: str | None,
    cell_fields: Sequence[str] = (),
    point_fields: Sequence[str] = (),
) -> Case:
    """Read the internal mesh of each region of an OpenFOAM case through VTK, with only
    the given fields.

    VTK keeps the point and cell order of OpenFOAM in each internal mesh, so point ``i``
    and cell ``i`` of a region are OpenFOAM point and cell label ``i``. The points of
    several regions are merged where they coincide exactly; a single region keeps its
    point order.

    Args:
        filename: The ``.foam`` file of the case
        time: Time step to read, the first one if ``None``
        region: Region to read, ``""`` for the default region, all regions if ``None``
        cell_fields: Names of cell-centred fields (``vol*Field``) to read
        point_fields: Names of point fields (``point*Field``) to read

    Raises:
        KeyError: If a field is not in the case, or not in every region read at this time.
        ValueError: If a field is asked for as cell data but is a point field, or the
            other way round.
    """
    reader = pyvista.POpenFOAMReader(str(filename))
    # By default VTK also makes a point field of every cell field, by averaging the cells
    # around each point. Cell fields are read as cell data only, so that point data holds
    # only the point fields of the case.
    reader.cell_to_point_creation = False
    # Check the fields asked for before reading anything, against the fields of the case
    case_fields = {"cell": list(reader.cell_array_names), "point": list(reader.point_array_names)}
    reader_name = {"cell": "read_cell_data", "point": "read_point_data"}
    description = {"cell": "cell-centred", "point": "a point field"}
    for kind, other, names in (("cell", "point", cell_fields), ("point", "cell", point_fields)):
        for name in names:
            if name in case_fields[kind]:
                continue
            if name in case_fields[other]:
                raise ValueError(
                    f"'{name}' is {description[other]}; read it with {reader_name[other]}."
                )
            raise KeyError(
                f"No {kind} field '{name}' in {filename}. {kind.capitalize()} fields: "
                f"{case_fields[kind]}"
            )
    # VTK reads every field of the time step unless told otherwise: read only those asked for
    reader.disable_all_cell_arrays()
    reader.disable_all_point_arrays()
    for name in cell_fields:
        reader.enable_cell_array(name)
    for name in point_fields:
        reader.enable_point_array(name)
    # Read only the internal meshes, not the boundary patches: patches are read from the
    # polyMesh files by their labels, and skipping them saves time and memory on meshes
    # with many boundary faces.
    for name in reader.patch_array_names:
        if name.endswith("internalMesh"):
            reader.enable_patch_array(name)
        else:
            reader.disable_patch_array(name)
    if (t := _time_value(reader, time)) is not None:
        reader.set_active_time_value(t)
    regions = get_regions(reader, region)
    data = reader.read()

    # The internal mesh of each region to read, by region name
    grids: dict[str, pyvista.UnstructuredGrid] = {}
    for name in regions:
        block = data if name == "" else data[name]
        grid = block["internalMesh"] if isinstance(block, pyvista.MultiBlock) else None
        if not isinstance(grid, pyvista.UnstructuredGrid):
            raise TypeError(f"Expected an unstructured internal mesh in {filename}.")
        grids[name] = grid

    if len(regions) == 1:
        points = np.asarray(grids[regions[0]].points)
        node_maps = [np.arange(len(points), dtype=np.int64)]
    else:
        # Regions are separate meshes, joined where their points coincide exactly.
        # Adding 0 turns -0.0 into 0.0, which compare unequal as bytes.
        all_points = np.vstack([np.asarray(grids[r].points) for r in regions]) + 0.0
        points, inverse = np.unique(all_points, axis=0, return_inverse=True)
        offsets = np.cumsum(np.array([0] + [grids[r].n_points for r in regions]))
        node_maps = [
            inverse.reshape(-1)[offsets[i] : offsets[i + 1]].astype(np.int64)
            for i in range(len(regions))
        ]

    # Merge the cells of each type, region after region
    region_cells = [_grid_cells(grids[r], m) for r, m in zip(regions, node_maps)]
    types = {cell_type for cells in region_cells for cell_type in cells}
    cells = {t: np.vstack([c[t] for c in region_cells if t in c]) for t in types}
    counts = [grids[r].n_cells for r in regions]

    def check_regions(name: str, kind: str, has_field: list[bool]):
        """Raise if a field is missing from some of the regions read, at this time step."""
        if all(has_field):
            return
        at = f" at time {t}" if t is not None else ""
        if len(regions) == 1:
            raise KeyError(f"No {kind} field '{name}'{at} in {filename}.")
        missing = [r for r, has in zip(regions, has_field) if not has]
        found = [r for r, has in zip(regions, has_field) if has]
        raise KeyError(
            f"No {kind} field '{name}'{at} in region(s) {missing} of {filename}"
            + (f"; found in {found}." if found else ".")
        )

    # Merge the fields asked for
    cell_data: dict[str, np.ndarray] = {}
    for name in cell_fields:
        check_regions(name, "cell", [name in grids[r].cell_data for r in regions])
        cell_data[name] = np.vstack([_as_rows(grids[r].cell_data[name]) for r in regions])
    point_data: dict[str, np.ndarray] = {}
    for name in point_fields:
        check_regions(name, "point", [name in grids[r].point_data for r in regions])
        # A point shared by regions takes the value of the last of them
        first = _as_rows(grids[regions[0]].point_data[name])
        values = np.zeros((len(points), first.shape[1]), dtype=first.dtype)
        for r, node_map in zip(regions, node_maps):
            values[node_map] = _as_rows(grids[r].point_data[name])
        point_data[name] = values

    return Case(
        path=Path(filename).parent,
        regions=regions,
        points=points,
        cells=cells,
        node_maps=node_maps,
        cell_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        cell_data=cell_data,
        point_data=point_data,
    )


def _find(path: Path) -> Path | None:
    """The file at ``path``, or its gzipped version ``path.gz``; ``None`` if neither exists."""
    for candidate in (path, path.with_name(path.name + ".gz")):
        if candidate.exists():
            return candidate
    return None


def field_exists(case_dir: Path, time_name: str, region: str, name: str) -> bool:
    """Whether a field file exists in a time directory, gzipped or not."""
    return _find(case_dir / time_name / region / name) is not None


# ---------------------------------------------------------------------------------------
# Reading polyMesh files. Only the entries needed here are read: the faces and patches,
# and the cell zones. Lists can be ASCII or binary, and files gzipped.


def _read_bytes(path: Path) -> bytes:
    """The contents of a file that may be gzipped."""
    if (found := _find(path)) is None:
        raise FileNotFoundError(f"Could not find {path} or {path.name}.gz")
    return gzip.decompress(found.read_bytes()) if found.suffix == ".gz" else found.read_bytes()


class _Scanner:
    """Reads the tokens and lists of a ``FoamFile``, after its header."""

    def __init__(self, path: Path):
        self.path = path
        self.data = _read_bytes(path)
        match = re.search(rb"FoamFile\s*\{(.*?)\}", self.data, re.S)
        if match is None:
            raise ValueError(f"No FoamFile header in {path}")
        entries = re.findall(rb'(\w+)\s+("[^"]*"|[^;]*);', match.group(1))
        self.header = {k.decode(): v.strip().strip(b'"').decode() for k, v in entries}
        self.binary = self.header.get("format", "ascii") == "binary"
        arch = re.search(r"label=(\d+)", self.header.get("arch", ""))
        self.label_dtype = np.dtype("<i8") if arch and arch.group(1) == "64" else np.dtype("<i4")
        self.pos = match.end()

    def skip(self):
        """Skip white space and comments."""
        data = self.data
        while self.pos < len(data):
            if data[self.pos : self.pos + 1].isspace():
                self.pos += 1
            elif data.startswith(b"//", self.pos):
                end = data.find(b"\n", self.pos)
                self.pos = len(data) if end < 0 else end
            elif data.startswith(b"/*", self.pos):
                end = data.find(b"*/", self.pos)
                self.pos = len(data) if end < 0 else end + 2
            else:
                break

    def peek(self) -> bytes:
        self.skip()
        return self.data[self.pos : self.pos + 1]

    def expect(self, char: bytes):
        if self.peek() != char:
            raise ValueError(f"Expected {char!r} at byte {self.pos} of {self.path}")
        self.pos += 1

    def word(self) -> str:
        """The next token: a word, number or punctuation character."""
        self.skip()
        if self.data[self.pos : self.pos + 1] in (b"(", b")", b"{", b"}", b";"):
            self.pos += 1
            return self.data[self.pos - 1 : self.pos].decode()
        match = re.compile(rb"[^\s(){};]+").match(self.data, self.pos)
        if match is None:
            raise ValueError(f"Unexpected end of {self.path}")
        self.pos = match.end()
        return match.group().decode()

    def label_list(self, binary_dtype: np.dtype | None = None) -> npt.NDArray[np.int64]:
        """A list of integers: ``N(...)``, binary or ASCII, or uniform ``N{value}``.

        Args:
            binary_dtype: Type of the items in a binary list, labels by default. Lists of
                booleans have one byte per item.
        """
        dtype = self.label_dtype if binary_dtype is None else binary_dtype
        count = int(self.word())
        if self.peek() == b"{":
            self.pos += 1
            value = int(self.word())
            self.expect(b"}")
            return np.full(count, value, dtype=np.int64)
        self.expect(b"(")
        if self.binary:
            labels = np.frombuffer(self.data, dtype, count, self.pos)
            self.pos += count * dtype.itemsize
        else:
            end = self.data.index(b")", self.pos)
            labels = np.array(self.data[self.pos : end].split(), dtype=np.int64)
            self.pos = end
        self.expect(b")")
        if len(labels) != count:
            raise ValueError(f"Expected {count} items in {self.path}, found {len(labels)}")
        return labels.astype(np.int64)

    def dictionaries(self, read_value: Callable[[str], Any]) -> list[tuple[str, dict[str, Any]]]:
        """A list of named dictionaries, ``N ( name { key value; ... } ... )``, as in the
        ``boundary`` and zones files.

        Args:
            read_value: Reads the value of an entry, from after its key up to and
                including its ``;``, given the key

        Returns:
            The name and entries of each dictionary, in order
        """
        count = int(self.word())
        self.expect(b"(")
        result = []
        for _ in range(count):
            name = self.word()
            self.expect(b"{")
            entries = {}
            while self.peek() != b"}":
                key = self.word()
                entries[key] = read_value(key)
            self.expect(b"}")
            result.append((name, entries))
        return result

    def skip_entry(self):
        """Skip the value of a dictionary entry, up to and including its ``;``."""
        depth = 0
        while True:
            token = self.word()
            if token in ("(", "{"):
                depth += 1
            elif token in (")", "}"):
                depth -= 1
            elif token == ";" and depth == 0:
                return


def read_faces(polymesh: Path) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The faces of a mesh, from its ``faces`` file.

    Each face is given by the indices of its points in the ``points`` file (OpenFOAM's
    "point labels"), e.g. ``4(0 1 5 4)`` for a quadrilateral.

    Returns:
        The offsets of the faces, and the point indices of all faces, one face after
        another: face ``f`` is ``indices[offsets[f]:offsets[f + 1]]``
    """
    scanner = _Scanner(polymesh / "faces")
    if scanner.header.get("class") == "faceCompactList":
        offsets = scanner.label_list()
        labels = scanner.label_list()
        return offsets, labels
    if scanner.binary:
        raise NotImplementedError(f"Binary faceList in {scanner.path} is not supported.")
    count = int(scanner.word())
    scanner.expect(b"(")
    sizes, labels_list = [], []
    for _ in range(count):
        faces = scanner.label_list()
        sizes.append(len(faces))
        labels_list.append(faces)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    labels = np.concatenate(labels_list) if labels_list else np.zeros(0, dtype=np.int64)
    return offsets, labels


def read_boundary(polymesh: Path) -> list[dict[str, str]]:
    """The patches of a mesh, in order, with their simple entries, e.g. ``nFaces``,
    ``startFace``, ``type``, and for mapped patches ``sampleRegion`` and ``samplePatch``."""
    scanner = _Scanner(polymesh / "boundary")

    def raw_text(key: str) -> str:
        start = scanner.pos
        scanner.skip_entry()
        return scanner.data[start : scanner.pos - 1].decode(errors="replace").strip()

    return [{"name": name, **entries} for name, entries in scanner.dictionaries(raw_text)]


def read_zones(polymesh: Path, kind: str) -> dict[str, npt.NDArray[np.int64]]:
    """The labels of each zone of a mesh, in order; empty if there are no such zones.

    Args:
        polymesh: The ``polyMesh`` directory
        kind: ``"cell"``, ``"face"`` or ``"point"``, for the ``cellZones``, ``faceZones``
            or ``pointZones`` file and the cell, face or point labels of each zone
    """
    if _find(polymesh / f"{kind}Zones") is None:
        return {}
    scanner = _Scanner(polymesh / f"{kind}Zones")

    def labels(key: str) -> npt.NDArray[np.int64] | None:
        if key not in (f"{kind}Labels", "flipMap"):
            scanner.skip_entry()
            return None
        if not scanner.peek().isdigit():
            scanner.word()  # List<label> or List<bool>
        # The flipMap of a face zone is a list of booleans, one byte each if binary
        values = scanner.label_list(np.dtype("u1") if key == "flipMap" else None)
        scanner.expect(b";")
        return values

    zones = scanner.dictionaries(labels)
    empty = np.zeros(0, dtype=np.int64)
    return {name: entries.get(f"{kind}Labels", empty) for name, entries in zones}
