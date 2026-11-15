.. _openfoambackend:

OpenFOAM-backend
----------------

Reads `OpenFOAM <https://www.openfoam.com>`_ cases through the OpenFOAM reader of
`pyvista <https://pyvista.org>`_ (:code:`pip install io4dolfinx[openfoam]`). The backend
is read-only. Pass the ``.foam`` file of the case, which can be created with
``touch case.foam``. All file access happens on rank 0.

The backend reads:

- The mesh, of hexahedra, prisms, pyramids or tetrahedra. A mesh with several of these
  cell types is read as a mixed mesh.
- Cell fields with :func:`io4dolfinx.read_cell_data`, as DG-0 functions, and point fields
  with :func:`io4dolfinx.read_point_data`, at any time step listed by
  :func:`io4dolfinx.read_timestamps`.
- Mesh tags of the boundary patches and the cell, face and point zones: ``"cell"`` for
  all cell zones, ``"facet"`` for all patches and face zones, and ``"peak"`` for all
  point zones, or a single patch or zone by its name. ``"regions"`` tags the cells by
  region. Use :func:`io4dolfinx.backends.openfoam.backend.read_ids` to list the ids.
- Cases with several regions, merged into one mesh. Read a single region with
  ``backend_args={"region": name}``.

.. code-block:: python

    mesh = io4dolfinx.read_mesh("case.foam", MPI.COMM_WORLD, backend="openfoam")
    T = io4dolfinx.read_cell_data("case.foam", "T", mesh, time=1.0, backend="openfoam")
    facets = io4dolfinx.read_meshtags("case.foam", mesh, "facet", backend="openfoam")
    inlet = io4dolfinx.read_meshtags("case.foam", mesh, "inlet", backend="openfoam")

Polyhedral cells are not supported, as DOLFINx has no polyhedra. Fields and mesh tags
cannot be read on mixed meshes, and facet tags not on prism or pyramid meshes. Cell
fields are not interpolated to point data. The points and fields are read in single
precision, and returned in ``backend_args["dtype"]``, by default
:data:`dolfinx.default_real_type`.

.. automodule:: io4dolfinx.backends.openfoam.backend
    :members:
    :exclude-members: read_attributes, write_attributes, write_mesh, write_meshtags, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array, write_data
