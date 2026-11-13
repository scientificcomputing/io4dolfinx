.. _vtkhdfbackend:

VTKHDF-backend
--------------

Relies on MPI compatible :code:`h5py`, similar to the  :ref:`H5PY backend<h5pybackend>`.
See that backend for detailed installation instructions.

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

.. automodule:: io4dolfinx.backends.vtkhdf.backend
    :members:
    :exclude-members: read_attributes, read_timestamps, write_attributes, write_mesh, write_meshtags, read_meshtags_data, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array
