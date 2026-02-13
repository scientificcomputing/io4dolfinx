ADIOS2-backend
--------------

The library depends on the Python-interface of [DOLFINx](https://github.com/) and an MPI-build of [ADIOS2](https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html#as-package).
Therefore `ADIOS2` should not be install through PYPI/pip, but has to be installed through Conda, Spack or from source.

.. important::
    ADIOS2<2.10.2 does not work properly with :code:`numpy>=2.0.0`. Everyone is advised to use the newest version of ADIOS2.
    This is for instance available through :code:`conda` or the :code:`ghcr.io/fenics/dolfinx/dolfinx:nightly` Docker-image.


.. automodule:: io4dolfinx.backends.adios2.backend
    :members:
    :exclude-members: read_point_data