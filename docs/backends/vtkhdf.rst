.. _vtkhdfbackend:

VTKHDF-backend
--------------

Relies on MPI compatible :code:`h5py`, similar to the  :ref:`H5PY backend<h5pybackend>`.
See that backend for detailed installation instructions. With :code:`h5py` built without MPI,
it runs as :code:`vtkhdf-serial`, which writes through rank 0 in the same way as
:code:`h5py-serial`.

Lossless compression
~~~~~~~~~~~~~~~~~~~~

Datasets can be stored with lossless HDF5 filters, set through ``backend_args``; by default,
the integer mesh arrays use scale-offset. See :doc:`../hdf5_compression` for the options, an
example and a comparison of file sizes and write/read times.

Vectors and tensors
~~~~~~~~~~~~~~~~~~~

VTK reads arrays with 3 components as vectors and arrays with 9 components as tensors.
Point and cell data with fewer components are therefore padded with zeros when written:
vectors of length 1 or 2 are stored with 3 components, and tensors of at most 3 x 3 are
embedded in a 3 x 3 tensor (row by row), so that, for instance, a 2D displacement can warp
the mesh in ParaView. The value shape of each array is stored as the attribute
``ValueShape`` of the array's metadata group,
``/io4dolfinx/<mesh name>/<Point|Cell>Data/<array name>``, and io4dolfinx removes the
padding when it reads the array back with its value shape. Arrays without this
attribute, written by other programs or by older versions of io4dolfinx, are read with
as many components as are stored: one component as a scalar, more as a vector.

Time steps
~~~~~~~~~~

A VTKHDF array has data at every time step, the time steps increase, and all meshes
and mesh tags in a file share them. io4dolfinx keeps to these rules when writing:

1. A mesh or an array can be written only once at a time; a second write raises a
   ``ValueError`` and leaves the file unchanged. Times that differ by round-off are
   the same time.
2. A new time must be later than the last time step of the file.
3. A new time step is added to every mesh and mesh tags in the file.
4. At a step where an array or the geometry was not written, it holds its latest
   earlier write, and the steps before its first write hold its first write.
5. The steps at which io4dolfinx wrote each array and mesh are recorded in
   ``/io4dolfinx``, which VTK ignores. Files without this record, from older versions
   of io4dolfinx, are assumed to have a write at the first step and wherever the
   offsets of the data change.

Only the geometry of a mesh can change between time steps: its topology is stored once,
so writing a mesh with another number of points or cells under the same name raises a
``ValueError``.

VTK 9.7 reads only the first time step of a MultiBlockDataSet when a time is requested
(`vtk#20246 <https://gitlab.kitware.com/vtk/vtk/-/issues/20246>`_), unless the root has a
``Steps`` group. Until this is fixed, io4dolfinx links ``/VTKHDF/Steps`` to the time steps
of the first mesh, which earlier versions of VTK ignore.

:func:`io4dolfinx.read_timestamps` returns every time step of the mesh an array is
stored on. See :doc:`../vtkhdf_time_steps` for an executed example with a splitting
scheme, and the write orders that are rejected.

.. automodule:: io4dolfinx.backends.vtkhdf.backend
    :members:
    :exclude-members: read_attributes, read_timestamps, write_attributes, write_mesh, write_meshtags, read_meshtags_data, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array
