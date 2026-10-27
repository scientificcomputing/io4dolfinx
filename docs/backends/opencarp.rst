.. _opencarpbackend:

openCARP-backend
----------------

Writes and reads linear tetrahedral meshes in the ASCII mesh format of the cardiac simulator
`openCARP <https://opencarp.org>`_: node coordinates (``.pts``), elements with region tags
(``.elem``), fiber and sheet directions per element (``.lon``), and markers as vertex (``.vtx``)
and surface (``.surf``) files. The backend needs no extra dependencies.

Nodes and elements are written in the ordering of the input mesh, gathered to and written by
rank 0, so the files are the same whatever the number of processes. openCARP expects coordinates
in µm, so the factor converting the mesh coordinates to µm must always be given.

:func:`io4dolfinx.backends.opencarp.write_opencarp` writes everything in one call:

.. code-block:: python

    from io4dolfinx.backends.opencarp import write_opencarp

    write_opencarp(
        "heart",
        mesh,
        scale=1000.0,  # mesh in mm
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        fiber=f0,
        sheet=s0,
        markers={"ENDO": (1, 2), "EPI": (2, 2)},
    )

This gives ``heart.pts``, ``heart.elem``, ``heart.lon``, and ``heart_ENDO.vtx``,
``heart_ENDO.surf``, ``heart_EPI.vtx`` and ``heart_EPI.surf``.

- Elements with negative volume have two nodes swapped, so all elements are positively oriented.
- Cells without a cell tag get region ``0`` (set ``backend_args={"default_tag": ...}`` to change it).
- Fibers and sheets are interpolated into cell midpoints and normalised. Fields in a quadrature
  space cannot be written; generate them in a P-1 or DG-0 space.
- Each facet marker value gets a ``.vtx`` file with its nodes and a ``.surf`` file with its facets,
  oriented out of the mesh. The files are named from ``markers``, or by the value if it has no name.
  Pass ``backend_args={"surf": False}`` to skip the ``.surf`` files.

The same files can be written with the generic functions, as long as everything is in the input
ordering:

.. code-block:: python

    args = {"scale": 1000.0}
    io4dolfinx.write_mesh_input_order("heart", mesh, backend="opencarp", backend_args=args)
    io4dolfinx.write_meshtags("heart", mesh, cell_tags, backend="opencarp", on_input_mesh=True)
    io4dolfinx.write_cell_data(
        "heart", f0, 0.0, io4dolfinx.FileMode.write, args, "opencarp", on_input_mesh=True
    )
    io4dolfinx.write_cell_data(
        "heart", s0, 0.0, io4dolfinx.FileMode.append, args, "opencarp", on_input_mesh=True
    )

Reading takes the path of any of the mesh files. Coordinates are divided by ``scale``:

.. code-block:: python

    args = {"scale": 1000.0, "markers": {"ENDO": (1, 2), "EPI": (2, 2)}}
    mesh = io4dolfinx.read_mesh("heart.elem", MPI.COMM_WORLD, backend="opencarp", backend_args=args)
    regions = io4dolfinx.read_meshtags("heart.elem", mesh, "cell", backend="opencarp")
    facets = io4dolfinx.read_meshtags("heart.elem", mesh, "facet", backend="opencarp", backend_args=args)
    endo = io4dolfinx.read_meshtags("heart.elem", mesh, "ENDO", backend="opencarp", backend_args=args)
    f0 = io4dolfinx.read_cell_data("heart.elem", "fiber", mesh, backend="opencarp")

``"facet"`` reads all ``.surf`` files of the mesh, and a marker name reads its ``.surf`` file, or its
``.vtx`` file as vertex tags if there is no ``.surf``.

.. automodule:: io4dolfinx.backends.opencarp.backend
    :members:
    :exclude-members: read_attributes, read_timestamps, write_attributes, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array, read_point_data
