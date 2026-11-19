# Installation

The main way to install `io4dolfinx` is through [PYPI](https://pypi.org/project/io4dolfinx/), which provides pre-built binary wheels for most platforms. This is the recommended method for most users.

```bash
python3 -m pip install io4dolfinx
```

`io4dolfinx` installs `h5py`, which its default backend uses. Other backends (like ADIOS2) have optional dependencies, installed with the appropriate extras:

- Test dependencies (for running the test suite):

```bash
python3 -m pip install "io4dolfinx[test]"
```

- Documentation dependencies (for building the docs):

```bash
python3 -m pip install "io4dolfinx[docs]"
```

- The `h5py` wheels on PyPI are built without MPI. With them, the default `h5py` backend still runs in parallel, but writes all data through rank 0 (see the [H5PY backend](./backends/h5py.rst)). For parallel IO, you need an HDF5 library with MPI support, and `h5py` built against it. You can build `h5py` with MPI support using pip:

```bash
HDF5_MPI=ON python3 -m pip install --no-binary=h5py h5py
```

  For the official DOLFINx Docker images, which have an MPI build of HDF5 but not of `h5py`, see [Docker](#docker).

- For pyvista support, you can install the `pyvista` package:

```bash
python3 -m pip install pyvista
```

or equivalently

```bash
python3 -m pip install "io4dolfinx[pyvista]"
```

- For ADIOS2 support you should have ADIOS2 installed with Python bindings, see https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html for more info. `python3 -m pip install "io4dolfinx[adios2]"` installs the ADIOS2 wheels from PyPI, which are built without MPI: the `adios2` backend then runs as `adios2-serial`, which writes all data through rank 0.

## Spack

The FEniCS Spack packages uses a separate [spack repo](https://github.com/FEniCS/spack-fenics) to be possible to maintain and keep up to date.
We do the same for the [packages](https://github.com/scientificcomputing/spack_repos.git) maintained by Scientific Computing at Simula Research Laboratory.
To install `py-io4dolfinx`, one should first install spack on your system, then use the following commands in a new spack environment:

```bash
spack repo add https://github.com/FEniCS/spack-fenics.git
spack repo add https://github.com/scientificcomputing/spack_repos.git
spack add py-io4dolfinx@1.1 ^py-fenics-dolfinx+petsc4py ^adios2+python+hdf5 ^petsc+mumps
```

to get an installation of `io4dolfinx` with all backends installed. If you require further petsc packages you should activate them by adding them to `^petsc+....`.
See [Spack PETSc](https://packages.spack.io/package.html?name=petsc) for options.

## Docker

The official DOLFINx containers have MPI builds of HDF5 and ADIOS2, but not of `h5py`, so `pip install io4dolfinx` installs `h5py` without MPI, which writes through rank 0. For parallel IO with the default `h5py` backend, build `h5py` against the HDF5 of the image. Images from the main branch of DOLFINx (`nightly`) point `HDF5_ROOT` to it, while the v0.10 and v0.11 images have it in `/usr/local`:

```bash
HDF5_MPI=ON HDF5_DIR=${HDF5_ROOT:-/usr/local} python3 -m pip install --no-binary=h5py --no-build-isolation h5py
```

or use the `adios2` backend, for instance with `io4dolfinx.set_default_backend("adios2")`.

Create a Docker container, named for instance `dolfinx-checkpoint`.
Use the `nightly` tag to get the main branch of DOLFINx, or `stable` to get the latest stable release

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --name=dolfinx-checkpoint ghcr.io/fenics/dolfinx/dolfinx:nightly
```

For the latest version compatible with nightly (with the ability to run the test suite), use

```bash
export HDF5_MPI=ON
export HDF5_DIR=${HDF5_ROOT:-/usr/local}
python3 -m pip install --no-binary=h5py --no-build-isolation io4dolfinx[test]@git+https://github.com/scientificcomputing/io4dolfinx@main
```

If you are using the `stable` image, you can install `io4dolfinx` from [PYPI](https://pypi.org/project/io4dolfinx/) with

```bash
python3 -m pip install io4dolfinx[test]
```

This docker container can be opened with

```bash
docker container start -i dolfinx-checkpoint
```

at a later instance

## Conda

```{note}
Conda supports the stable release of DOLFINx, and thus the appropriate version should be installed, see the section above for more details.
```

Following is a minimal recipe of how to install io4dolfinx, given that you have conda installed on your system.

```bash
conda create -n dolfinx-checkpoint python=3.12
conda activate dolfinx-checkpoint
conda install -c conda-forge io4dolfinx
```

```{note}
Remember to download the appropriate version of `io4dolfinx` from Github [io4dolfinx: Releases](https://github.com/scientificcomputing/io4dolfinx/releases)
```

To run the test suite, you should also install `ipyparallel`, `pytest` and `coverage`, which can all be installed with conda

```bash
conda install -c conda-forge ipyparallel pytest coverage
```
