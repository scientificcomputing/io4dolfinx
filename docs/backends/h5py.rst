.. _h5pybackend:

H5PY-backend
-------------

The H5PY-backend relies on an MPI compatible build of :code:`HDF5`, which should be linked to the same MPI
implementation as DOLFINx relies on.

The DOLFINx docker images (:code:`ghcr.io/fenics/dolfinx/dolfinx:stable`) comes with an already configures MPI compatible HDF5 installation, and `h5py` can in
turn be installed with

.. code-block:: bash

    HDF5_MPI="ON" HDF5_DIR="/usr/local" python3 -m pip install --no-binary=h5py h5py --no-build-isolation

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

.. automodule:: io4dolfinx.backends.h5py.backend
    :members:
    :exclude-members: read_point_data