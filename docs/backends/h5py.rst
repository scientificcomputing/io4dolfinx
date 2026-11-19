.. _h5pybackend:

H5PY-backend
-------------

The H5PY-backend relies on an MPI compatible build of :code:`HDF5`, which should be linked to the same MPI
implementation as DOLFINx relies on.

h5py without MPI
~~~~~~~~~~~~~~~~

With :code:`h5py` built without MPI, as the wheels on PyPI are, the backend runs as
:code:`h5py-serial`, with a warning to install :code:`h5py` with MPI. It writes the same files:
every process reads its part of a file on its own, while all data written is gathered on
rank 0, which writes it. This works on any number of processes, but rank 0 holds and writes
all the data, so build :code:`h5py` with MPI, as below, for large problems. The
:code:`h5py-serial` backend can also be chosen directly.

The DOLFINx docker images (:code:`ghcr.io/fenics/dolfinx/dolfinx:stable`) comes with an already configures MPI compatible HDF5 installation, and `h5py` can in
turn be installed with

.. code-block:: bash

    HDF5_MPI="ON" HDF5_DIR="${HDF5_ROOT:-/usr/local}" python3 -m pip install --no-binary=h5py h5py --no-build-isolation

If you are using `apt` on Ubuntu, this can for instance be achieved with the following commands (here using Docker).
Note that this code block does not install DOLFINx, it just illustrates how to get the correct `h5py`.

.. code-block:: dockerfile

    FROM ubuntu:24.04 AS base
    RUN apt-get update && apt-get install -y python3-dev python3-pip python3-venv libhdf5-mpi-dev libopenmpi-dev


    ENV VIRTUAL_ENV=/test-env
    ENV PATH=/test-env/bin:$PATH
    RUN python3 -m venv ${VIRTUAL_ENV}

    ENV HDF5_MPI="ON"
    ENV HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi/
    ENV C_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include/:${C_PATH}
    RUN python3 -m pip install setuptools cython numpy pkgconfig mpi4py
    RUN CC=mpicc python3 -m pip install --no-binary=h5py h5py --no-build-isolation

Lossless compression
~~~~~~~~~~~~~~~~~~~~

Datasets can be stored with lossless HDF5 filters through ``backend_args``. See
:doc:`../h5py_compression` for an executed example and a comparison of file sizes
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
- ``scaleoffset`` (default ``True``): store integer arrays (topology, dofmaps,
  offsets, cell permutations, partition data and integer mesh tags) with the minimum
  number of bits per chunk.
  This is lossless for integers; floating-point data are left unchanged, as
  scale-offset would be lossy for them. It is a built-in HDF5 filter, available
  in every HDF5 build. The default applies to new datasets only; set
  ``scaleoffset=False`` to turn it off. h5py cannot combine it with
  ``fletcher32``, so ``fletcher32=True`` turns it off unless ``scaleoffset=True``
  is also given, which raises an error. Unlike h5py, this is a boolean; see
  :ref:`h5py:dataset_scaleoffset`.

All boolean options must be ``True`` or ``False``. Only gzip compression is
supported, using the same storage options as VTKHDF.

Chunking
~~~~~~~~

Mesh coordinates, topology, partition data, function values, dofmaps, offsets,
cell permutations, mesh tags and snapshot values are each
:ref:`chunked<h5py:dataset_chunks>` by whole rows and filtered independently.
Attributes, including timestamps and format metadata, remain unfiltered.

- ``chunk_bytes`` (default ``1048576``, 1 MiB): the size of a chunk in bytes,
  rounded down to whole rows and at least one row. A chunk is never longer than
  the dataset when it is created.

Each rank writes, and each reader reads, a contiguous block of rows, which then
shares at most two chunks with its neighbours, whatever the number of processes.

The chunk size, like the filters, must be the same on every rank. Existing
objects retain their chunk size on append; another ``chunk_bytes`` is not an error.

Settings apply when an object is first written. Each geometry or function time
step has its own dataset; later steps retain the previous step's settings when
options are omitted. Explicitly conflicting filters raise ``ValueError`` before
timestamps or data are changed. New meshes and functions use the options of their
own write call and do not inherit settings from ``write_mesh``. Files written by
older versions keep their existing layout when appended without options.

Invalid options and unavailable filters raise errors rather than silently
disabling compression.

Function-only checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~

The same storage options apply to :func:`io4dolfinx.write_function` when the mesh
is stored separately or is not written to a checkpoint. Reading still requires a
function space on a compatible mesh. With the default ``on_input_mesh=False``,
read the mesh from its separate checkpoint before constructing that space. With
``on_input_mesh=True``, read the function onto the original input mesh; see
:ref:`input-ordering`. See :ref:`h5py-function-only` for an executed example.
These options also apply to ``h5py-serial``.

.. note::
    In parallel, all datasets are written with collective MPI-IO, as in DOLFINx:
    every rank in the communicator must call the writer, including ranks with no
    local data, and must pass identical options. Parallel filtered writes require
    MPI-enabled h5py; see the
    `HDF5 parallel compression guidance <https://portal.hdfgroup.org/documentation/hdf5/latest/_par_compr.html>`_.

.. automodule:: io4dolfinx.backends.h5py.backend
    :members:
    :exclude-members: read_point_data

.. automodule:: io4dolfinx.backends.serial
    :members:
