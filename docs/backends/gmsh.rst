.. _gmshbackend:

Gmsh-backend
------------

Reads and writes Gmsh ``.msh`` files through the `Gmsh <https://gmsh.info>`_ Python API
(:code:`pip install io4dolfinx[gmsh]`), reusing :py:mod:`dolfinx.io.gmsh`. Meshes, mesh
tags (physical groups), point data (``$NodeData``) and cell data (``$ElementData``) are
supported, with time-dependent data stored as the steps of a view. Complex data is split
into the views ``real_{name}`` and ``imag_{name}``, as :py:class:`dolfinx.io.XDMFFile` does. All file access
happens on rank 0. Requires DOLFINx 0.11 or later.

.. automodule:: io4dolfinx.backends.gmsh.backend
    :members:
    :exclude-members: read_attributes, write_attributes, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array
