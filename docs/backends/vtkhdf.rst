.. _vtkhdfbackend:

VTKHDF-backend
--------------

Relies on MPI compatible :code:`h5py`, similar to the  :ref:`H5PY backend<h5pybackend>`.
See that backend for detailed installation instructions. With :code:`h5py` built without MPI,
it runs as :code:`vtkhdf-serial`, which writes through rank 0 in the same way as
:code:`h5py-serial`.

Lossless compression
~~~~~~~~~~~~~~~~~~~~

Datasets can be stored with lossless HDF5 filters through ``backend_args``. See
:doc:`../vtkhdf_compression` for an executed example and a comparison of file sizes
and write/read times. The filter options have the names and meaning of the
corresponding arguments to :meth:`h5py.Group.create_dataset`; see the h5py
:ref:`filter pipeline<h5py:dataset_compression>`. Supported options are:

- ``compression`` (default ``None``): ``"gzip"`` or ``None``.
- ``compression_opts``: an integer from 0 to 9, requiring ``compression="gzip"``;
  the gzip default is 4.
- ``shuffle`` (default ``False``): lossless byte reordering that can improve gzip
  compression; see :ref:`h5py:dataset_shuffle`.
- ``fletcher32`` (default ``False``): store a checksum per chunk, so that
  corrupted data raise an error when read; see :ref:`h5py:dataset_fletcher32`.
- ``scaleoffset`` (default ``True``): store integer arrays (connectivity,
  topology offsets and cell types) with the minimum number of bits per chunk.
  This is lossless for integers; floating-point data are left unchanged, as
  scale-offset would be lossy for them. It is a built-in HDF5 filter, available
  in every HDF5 build. The default applies to new datasets only; set
  ``scaleoffset=False`` to turn it off. h5py cannot combine it with
  ``fletcher32``, so ``fletcher32=True`` turns it off unless ``scaleoffset=True``
  is also given, which raises an error. Unlike h5py, this is a boolean; see
  :ref:`h5py:dataset_scaleoffset`.

All boolean options must be ``True`` or ``False``. Other compression filters
and explicit chunk shapes are not supported, as files using them may not be
readable in ParaView.

Coordinates, connectivity, topology offsets, cell types, point/cell values and
mesh tag values are filtered independently, each with h5py
:ref:`automatic chunking<h5py:dataset_chunks>`. Small bookkeeping
arrays, including timestep metadata, remain unfiltered.

Settings apply when a dataset is first created. On append, omitted options retain
that dataset's existing settings; explicitly conflicting settings raise
``ValueError`` before datasets are resized. Newly created fields use the options
of their own write call and do not inherit settings from ``write_mesh``.

Invalid options and unavailable filters raise errors rather than silently
disabling compression.

.. note::
    In parallel, all datasets are written with collective MPI-IO, as in DOLFINx:
    every rank in the communicator must call the writer, including ranks with no
    local data, and must pass identical options. Parallel filtered writes require
    MPI-enabled h5py; see the
    `HDF5 parallel compression guidance <https://portal.hdfgroup.org/documentation/hdf5/latest/_par_compr.html>`_.

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
