# Installation


## Spack

io4dolfinx is a [spack package](https://packages.spack.io/package.html?name=py-io4dolfinx)
which can be installed with

```bash
spack add py-io4dolfinx ^py-fenics-dolfinx+petsc4py+slepc4py
spack concretize
spack install
```

once you have downloaded spack and set up a new environment, as described in [Spack: Installation notes](https://github.com/spack/spack?tab=readme-ov-file#installation).
To ensure that the spack packages are up to date, please call

```bash
spack repo update builtin
```

prior to concretizing.

## Docker

An MPI build of ADIOS2 is installed in the official DOLFINx containers, and thus there are no additional dependencies required to install `io4dolfinx`
on top of DOLFINx in these images.

Create a Docker container, named for instance `dolfinx-checkpoint`.
Use the `nightly` tag to get the main branch of DOLFINx, or `stable` to get the latest stable release

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --name=dolfinx-checkpoint ghcr.io/fenics/dolfinx/dolfinx:nightly
```

For the latest version compatible with nightly (with the ability to run the test suite), use

```bash
export HDF5_MPI=ON
export HDF5_DIR=/usr/local
python3 -m pip install --no-binary-h5py --no-build-isolation io4dolfinx[test]@git+https://github.com/jorgensd/io4dolfinx@main
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

> [!NOTE]  
> Conda supports the stable release of DOLFINx, and thus the appropriate version should be installed, see the section above for more details.

Following is a minimal recipe of how to install io4dolfinx, given that you have conda installed on your system.

```bash
conda create -n dolfinx-checkpoint python=3.10
conda activate dolfinx-checkpoint
conda install -c conda-forge io4dolfinx
```

> [!NOTE]
> Remember to download the appropriate version of `io4dolfinx` from Github [io4dolfinx: Releases](https://github.com/jorgensd/io4dolfinx/releases)

To run the test suite, you should also install `ipyparallel`, `pytest` and `coverage`, which can all be installed with conda

```bash
conda install -c conda-forge ipyparallel pytest coverage
```
