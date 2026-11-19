ADIOS2-backend
--------------

The library depends on the Python-interface of [DOLFINx](https://github.com/) and an MPI-build of [ADIOS2](https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html#as-package),
installed through Conda, Spack or from source.

ADIOS2 without MPI
~~~~~~~~~~~~~~~~~~

With ADIOS2 built without MPI, as the wheels on PyPI are, the backend runs as
:code:`adios2-serial`, with a warning to install ADIOS2 with MPI. It writes the same files:
every process reads its part of a file on its own, while all data written is gathered on
rank 0, which writes it. This works on any number of processes, but rank 0 holds and writes
all the data, so use an MPI build of ADIOS2 for large problems. The :code:`adios2-serial`
backend can also be chosen directly.

.. important::
    ADIOS2<2.10.2 does not work properly with :code:`numpy>=2.0.0`. Everyone is advised to use the newest version of ADIOS2.
    This is for instance available through :code:`conda` or the :code:`ghcr.io/fenics/dolfinx/dolfinx:nightly` Docker-image.


.. automodule:: io4dolfinx.backends.adios2.backend
    :members:
    :exclude-members: read_point_data
