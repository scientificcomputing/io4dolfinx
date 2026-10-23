.. _medbackend:

MED-backend
-----------

Reads meshes and mesh tags from `MED <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html>`_
files, as exported by Salome and Code_Aster. The backend is read-only and relies on :code:`h5py`,
which must be MPI compatible to read in parallel (see the :ref:`H5PY backend<h5pybackend>`).

In a MED file each entity stores a *family* id, and each family lists the named *groups* it belongs to.
An entity in several groups gets a family of its own for that combination of groups.
Mesh tags can be read in two ways:

- ``meshtag_name="cell"`` or ``meshtag_name="facet"`` returns the family ids of the cells or facets
  (entities without a family are skipped). MED element family ids are negative, while integral
  subdomain ids must be non-negative, so the ids are negated: family ``-6`` gives tag ``6``.
- Any other ``meshtag_name`` is a group name, and returns the entities of that group tagged with ``1``.

.. code-block:: python

    mesh = io4dolfinx.read_mesh("mesh.med", MPI.COMM_WORLD, backend="med")
    facet_families = io4dolfinx.read_meshtags("mesh.med", mesh, meshtag_name="facet", backend="med")
    inlet = io4dolfinx.read_meshtags("mesh.med", mesh, meshtag_name="inlet", backend="med")

Use :func:`io4dolfinx.backends.med.backend.read_groups` to list the groups and their family ids.
If a file holds several meshes, select one with ``mesh_name``.

A time-dependent mesh stores one step per time. Select the step with ``time`` when reading the mesh,
and with ``backend_args={"time": t}`` when reading its tags (the first step is used if no time is
given). A mesh with a single step is read whatever the time.

.. code-block:: python

    mesh = io4dolfinx.read_mesh("mesh.med", MPI.COMM_WORLD, backend="med", time=0.5)
    cells = io4dolfinx.read_meshtags(
        "mesh.med", mesh, meshtag_name="cell", backend="med", backend_args={"time": 0.5}
    )

The cells must be of a single type. Linear and quadratic Lagrange cells (``SE3``, ``TR6``, ``QU9``,
``T10``, ``P18``, ``H27``) and the quadratic serendipity cells ``QU8`` and ``H20`` are supported.
The serendipity cells require the main branch of DOLFINx.
MED node orderings are converted with tables derived from MEDCoupling's MED cell model.

.. automodule:: io4dolfinx.backends.med.backend
    :members:
    :exclude-members: read_attributes, read_timestamps, write_attributes, write_mesh, write_meshtags, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array, read_point_data, read_cell_data, read_function_names, write_data
