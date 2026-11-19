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

Datasets can be stored with lossless HDF5 filters, set through ``backend_args``; by default,
integer arrays such as topologies and dofmaps use scale-offset. See :doc:`../compression` for
the options, an example and a comparison of file sizes and write/read times.

.. automodule:: io4dolfinx.backends.h5py.backend
    :members:
    :exclude-members: read_point_data

.. automodule:: io4dolfinx.backends.serial
    :members:
