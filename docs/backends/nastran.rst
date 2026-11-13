.. _nastranbackend:

Nastran-backend
---------------

Reads the mesh and mesh tags of Nastran bulk data files (``.bdf``, ``.nas``) through the
`Gmsh <https://gmsh.info>`_ Python API (:code:`pip install io4dolfinx[nastran]`), reusing
:py:mod:`dolfinx.io.gmsh`. The cells are the elements of the highest topological
dimension. Meshes with several cell types, such as the hexahedra, pyramids and tetrahedra
of a "hexcore" mesh, are read as mixed meshes, which have no UFL domain. Mesh tags are
the property IDs (PID) of the ``PSOLID`` (cells) or ``PSHELL`` (facets) elements, and
must be of one entity type. GRID IDs need not be contiguous: input node ``i`` is the
node with the ``i``-th smallest GRID ID. On meshes with one cell type, the GRID IDs can
be read as point data, a function on the geometry nodes, and as vertex tags, both called
``"GRID"``. All file access happens on rank 0.

.. note::
    The Nastran backend requires DOLFINx 0.11 or later, and raises a
    ``NotImplementedError`` with older versions, whose
    :py:func:`dolfinx.io.gmsh.extract_topology_and_markers` reads only one element type
    per property ID.

.. code-block:: python

    mesh = io4dolfinx.read_mesh("model.bdf", MPI.COMM_WORLD, backend="nastran")
    cell_tags = io4dolfinx.read_meshtags("model.bdf", mesh, "PSOLID", backend="nastran")
    grid_ids = io4dolfinx.read_point_data("model.bdf", "GRID", mesh, backend="nastran")
    vertex_tags = io4dolfinx.read_meshtags("model.bdf", mesh, "GRID", backend="nastran")

.. automodule:: io4dolfinx.backends.nastran.backend
    :members: read_mesh_data, read_meshtags_data, read_point_data
