.. _vtkhdfbackend:

VTKHDF-backend
--------------

Relies on MPI compatible :code:`h5py`, similar to the  :ref:`H5PY backend<h5pybackend>`.
See that backend for detailed installation instructions.

Lossless compression
~~~~~~~~~~~~~~~~~~~~

Compression is opt-in through ``backend_args``. The public writer signatures are
unchanged. Reuse the same dictionary on each call to compress both the mesh and
its point or cell data:

.. code-block:: python

    from mpi4py import MPI
    from dolfinx.fem import Function, functionspace
    from dolfinx.mesh import create_unit_square
    import io4dolfinx

    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16)
    u = Function(functionspace(mesh, ("Lagrange", 1)), name="u")
    u.interpolate(lambda x: x[0] + x[1])
    q = Function(functionspace(mesh, ("DG", 0)), name="q")
    q.interpolate(lambda x: x[0] - x[1])

    storage_args = {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }
    io4dolfinx.write_mesh(
        "results.vtkhdf", mesh, backend="vtkhdf", backend_args=storage_args
    )
    io4dolfinx.write_point_data(
        "results.vtkhdf", u, time=0.0, mode=io4dolfinx.FileMode.append,
        backend="vtkhdf", backend_args=storage_args,
    )
    io4dolfinx.write_cell_data(
        "results.vtkhdf", q, time=0.0, mode=io4dolfinx.FileMode.append,
        backend="vtkhdf", backend_args=storage_args,
    )

Supported options are ``compression`` (``"gzip"`` or ``None``),
``compression_opts`` (an integer from 0 to 9, requiring ``compression="gzip"``;
the gzip default is 4), and ``shuffle`` (a boolean, default ``False``).
Shuffle is lossless and can improve gzip compression. Other compression filters,
lossy scale-offset, checksums, and explicit chunk shapes are not supported by this
interface. Omitting these options creates uncompressed datasets.

Coordinates, connectivity, topology offsets, cell types, and point/cell values
are compressed independently. Small bookkeeping arrays, including timestep
metadata, remain uncompressed. Each dataset uses h5py automatic chunking;
``chunks=True`` is accepted, but a fixed chunk shape is not. A chunk contains
elements of one dataset, not a combination of mesh and field arrays. See the
`h5py chunking guidance <https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage>`_.

Settings apply when a dataset is first created. On append, omitted options retain
that dataset's existing settings; explicitly conflicting settings raise
``ValueError`` before datasets are resized. Newly created fields use the options
of their own write call and do not inherit settings from ``write_mesh``.
Changing an existing dataset's filters requires rewriting it to a new file.

MPI writes to filtered datasets are collective automatically. Every rank in the
communicator must call the writer, including ranks with no local data, and must
pass identical filter options. Parallel compression requires MPI-enabled h5py
and HDF5 with parallel filtered-write support (introduced in HDF5 1.10.2).
Invalid options and unavailable filters raise errors rather than silently
disabling compression. Readers do not need compression arguments.

Compression reduces storage for compressible data, but CPU work, chunk sharing,
and MPI coordination can increase write time, particularly for small datasets.
There is no automatic size threshold or speed guarantee. Benchmark representative
data and rank counts; the repository's ``benchmarks/vtkhdf_compression.py`` reports
file sizes and mesh/field write and read times for uncompressed output, gzip,
and gzip with shuffle. For example, from the repository root:

.. code-block:: bash

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 4 \
        python benchmarks/vtkhdf_compression.py \
        --output output/compression/ranks-4 --sizes 16 256 768 --repeats 5 --sync

Each mode is warmed up before measurement and the measured order rotates between
repetitions. ``--sync`` includes a final filesystem ``fsync`` in write time;
reads use warm caches. The workload contains smooth scalar/vector fields, so
compression ratios do not predict those of noisy data. To combine runs in
``ranks-*`` directories into CSV and charts, run
``python benchmarks/report_vtkhdf_compression.py output/compression`` in an
environment with matplotlib. Compression enables collective writes, so timing
differences include the change of I/O mode as well as the filter itself. See the
`HDF5 parallel compression guidance <https://portal.hdfgroup.org/documentation/hdf5/latest/_par_compr.html>`_.

.. automodule:: io4dolfinx.backends.vtkhdf.backend
    :members:
    :exclude-members: read_attributes, read_timestamps, write_attributes, write_mesh, write_meshtags, read_meshtags_data, read_dofmap, read_dofs, read_cell_perms, write_function, read_legacy_mesh, snapshot_checkpoint, read_hdf5_array
