.. _vtkhdfbackend:

VTKHDF-backend
--------------

Relies on MPI compatible :code:`h5py`, similar to the  :ref:`H5PY backend<h5pybackend>`.
See that backend for detailed installation instructions.

.. automodule:: io4dolfinx.backends.vtkhdf.backend
    :members:
    :exclude-members: read_attributes, read_timestamps, write_attributes, write_mesh, write_meshtags, read_meshtags_data, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array