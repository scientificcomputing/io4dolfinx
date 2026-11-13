.. _opencarpbackend:

openCARP-backend
----------------

Writes and reads linear meshes in the ASCII mesh format of the cardiac simulator
`openCARP <https://opencarp.org>`_: node coordinates (``.pts``), elements with region tags
(``.elem``), fiber and sheet directions per element (``.lon``), and markers as vertex (``.vtx``)
and surface (``.surf``) files. The backend needs no extra dependencies.

The mesh, its tags and its fiber and sheet directions are written with
:func:`io4dolfinx.write_mesh`, :func:`io4dolfinx.write_meshtags` and
:func:`io4dolfinx.write_cell_data`, with the same ``on_input_mesh`` in every call, as shown in the
module documentation below. Rank 0 writes all files. openCARP expects coordinates in µm, so the factor
converting the mesh coordinates to µm must always be given as ``backend_args["scale"]``.

With ``markers={"ENDO": (1, 2), "EPI": (2, 2)}``, a heart mesh gives ``heart.pts``,
``heart.elem``, ``heart.lon``, ``heart_ENDO.vtx``, ``heart_ENDO.surf``, ``heart_EPI.vtx`` and
``heart_EPI.surf``, and ``heart.markers.json`` with the name, value and dimension of each marker.
Writing the mesh again removes the ``.lon`` and marker files of the old mesh, and writing facet tags
again replaces the facet markers.

- Triangle (``Tr``) and quadrilateral (``Qd``) meshes in 2D, and tetrahedral (``Tt``) and hexahedral
  (``Hx``) meshes in 3D are supported. 2D meshes are written with ``z = 0``, and read back with 2D
  coordinates. 2D meshes with 3D coordinates in a plane of constant ``z`` are also written.
- Elements get the orientation used by openCARP's ``mesher``: positive volume in 3D,
  counter-clockwise in 2D.
- Cells without a cell tag get region ``0`` (set ``backend_args={"default_tag": ...}`` to change it).
- Fibers and sheets are interpolated into cell midpoints and normalised. 2D directions get
  ``z = 0``, and are read back with two components. Fields in a quadrature space cannot be
  written; generate them in a P-1 or DG-0 space.
- Each facet marker value gets a ``.vtx`` file with its nodes and, for 3D meshes, a ``.surf`` file
  with its facets (``Tr`` or ``Qd``), oriented out of the mesh. A facet between two elements
  points out of the element whose centre comes first in lexicographic ``(x, y, z)`` order. The files
  are named from ``markers``, or by the value if it has no name; values that would give the same
  file name are rejected.
  Pass ``backend_args={"surf": False}`` to skip the ``.surf`` files.

Reading takes the path of any of the mesh files. Coordinates are divided by ``scale``:

.. code-block:: python

    args = {"scale": 1000.0, "markers": {"ENDO": (1, 2), "EPI": (2, 2)}}
    mesh = io4dolfinx.read_mesh("heart.elem", MPI.COMM_WORLD, backend="opencarp", backend_args=args)
    regions = io4dolfinx.read_meshtags("heart.elem", mesh, "cell", backend="opencarp")
    facets = io4dolfinx.read_meshtags("heart.elem", mesh, "facet", backend="opencarp", backend_args=args)
    endo = io4dolfinx.read_meshtags("heart.elem", mesh, "ENDO", backend="opencarp", backend_args=args)
    f0 = io4dolfinx.read_cell_data("heart.elem", "fiber", mesh, backend="opencarp")

``"facet"`` reads the ``.surf`` files of the facet markers listed in ``heart.markers.json``, with
their values. A marker name reads its ``.surf`` file, or its ``.vtx`` file as vertex tags if there is
no ``.surf`` (as for 2D meshes). This also works for marker files written by other tools; their value
is taken from ``markers``, else from the name if it is an integer, else it is ``1``.

Examples that write a mesh with this backend, run openCARP on it and read the results back (a planar
wave in a square, and the Niederer et al. (2011) benchmark) are in
`opencarp-io4dolfinx-example <https://github.com/finsberg/opencarp-io4dolfinx-example>`_.

.. automodule:: io4dolfinx.backends.opencarp.backend
    :members:
    :exclude-members: read_attributes, read_timestamps, write_attributes, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array, read_point_data
