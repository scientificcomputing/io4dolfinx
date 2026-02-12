# io4dolfinx - A framework for reading and writing data to various mesh formats

io4dolfinx is an extension for [DOLFINx](https://github.com/FEniCS/dolfinx/) that supports reading and writing various mesh formats using a variety of backends:

## Reading meshes, node data and cell data
Most meshing formats supports associating data with the nodes of the mesh (the mesh can be higher order) and the cells of the mesh. The node data can be read in as P-th order Lagrange functions (where P is the order of the grid), while the cell data can be read in as piecewise constant (DG-0) functions.

- [VTKHDF](./docs/backends/vtkhdf.rst): The new scalable format from VTK, called [VTKHDF](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/index.html) is supported by the `vtkhdf` backend. 
- [XDMF](./docs/backends/xdmf.rst) (eXtensible Model Data Format): `.xdmf`. The `xdmf` backend supports the `HDF5` encoding, to ensure performance in parallel.
- [PyVista](./docs/backends/pyvista.rst) (IO backend is meshio): The [pyvista](https://pyvista.org/) backend uses {py:func}`pyvista.read` to read in meshes, point data and cell data. `pyvista` relies on [meshio](https://github.com/nschloe/meshio) for most reading operations (including the XDMF ascii format).

## Checkpointing
Many finite element applications requires storage of functions that cannot be associated with the nodes or cells of the mesh. Therefore, we have implemented our own, native checkpointing format that supports N-to-M checkpointing (write data on N processors, read in on M) through the following backends:
- [h5py](./docs/backends/h5py.rst): Requires HDF5 with MPI support to work, but can store, meshes, partitioning info, meshtags, function data and more.
- [adios2](./docs/backends/adios2.rst): Requires [ADIOS 2](https://adios2.readthedocs.io/en/latest/) compiled with MPI support and Python bindings. Supports the same set of operations as the `h5py` backend.

The code uses the ADIOS2/Python-wrappers and h5py module to write DOLFINx objects to file, supporting N-to-M (_recoverable_) and N-to-N (_snapshot_) checkpointing.
See: [Checkpointing in DOLFINx - FEniCS 23](https://jsdokken.com/checkpointing-presentation/#/) or the examples in the [Documentation](https://jsdokken.com/io4dolfinx/) for more information.

For scalability, the code uses [MPI Neighbourhood collectives](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node200.htm) for communication across processes.

## Relation to adios4dolfinx

This library is an evolution of the [adios4dolfinx](https://doi.org/10.21105/joss.06451) code and has all the functionality of that library, and can read the checkpointing files from `adios4dolfinx`.

As `adios4dolfinx` was solely relying on ADIOS2 it was hard to interface other meshing formats and still keep the library structure sane.

`io4dolfinx` support custom user backends, which can be provided as a string through all functions via the `backend` keyword arg, i.e. `backend="name_of_my_module"`.

## Statement of Need

As the usage of high performance computing clusters increases, more and more large-scale, long-running simulations are deployed.
The need for storing intermediate solutions from such simulations are crucial, as the HPC system might crash, or the simulation might crash or exceed the alloted computational budget.
Having a checkpoint of related variables, such as the solutions to partial differential equations (PDEs) is therefore essential.
The `io4dolfinx` library extends the [DOLFINx](https://github.com/FEniCS/dolfinx/) computational framework for solving PDEs with checkpointing functionality, such that immediate solutions and mesh information can be stored and re-used in another simulation.

## Installation

The library is backwards compatible against the DOLFINx API of the nightly release of DOLFINx, v0.10.0 and v0.9.0.

The library can be installed through `pip` with `python3 -m pip install io4dolfinx`.
For notes on installation of specific backends see the individual backend documentation.

See [./docs/installation.md](./docs/installation.md) for further info.


## Functionality

### DOLFINx

- Reading and writing meshes, using `io4dolfinx.read/write_mesh`
- Reading and writing meshtags associated to meshes `io4dolfinx.read/write_meshtags`
- Reading checkpoints for any element (serial and parallel, arbitrary number of functions and timesteps per file). Use `io4dolfinx.read/write_function`.
- Writing standalone function checkpoints relating to "original meshes", i.e. meshes read from `XDMFFile`. Use `io4dolfinx.write_function_on_input_mesh` for this.
- Store mesh partitioning and re-read the mesh with this information, avoiding calling SCOTCH, Kahip or Parmetis.

> [!IMPORTANT]  
> For checkpoints written with `write_function` to be valid, you first have to store the mesh with `write_mesh` to the checkpoint file.

> [!IMPORTANT]  
> A checkpoint file supports multiple functions and multiple time steps, as long as the functions are associated with the same mesh

> [!IMPORTANT]  
> Only one mesh per file is allowed


## Example Usage

The repository contains many documented examples of usage, in the `docs`-folder, including

- [Reading and writing mesh checkpoints](./docs/writing_mesh_checkpoint.py)
- [Storing mesh partitioning data](./docs/partitioned_mesh.py)
- [Writing mesh-tags to a checkpoint](./docs/meshtags.py)
- [Reading and writing function checkpoints](./docs/writing_functions_checkpoint.py)
- [Checkpoint on input mesh](./docs/original_checkpoint.py)
  Further examples can be found at [io4dolfinx examples](https://jsdokken.com/io4dolfinx/)

### Legacy DOLFIN

Only checkpoints for `Lagrange` or `DG` functions are supported from legacy DOLFIN

- Reading meshes from the DOLFIN HDF5File-format
- Reading checkpoints from the DOLFIN HDF5File-format (one checkpoint per file only)
- Reading checkpoints from the DOLFIN XDMFFile-format (one checkpoint per file only, and only uses the `.h5` file)

See the [API](./docs/api) for more information.

## Testing

This library uses `pytest` for testing.
To execute the tests, one should first install the library and its dependencies, as listed above.
Then, can execute all tests by calling

```bash
python3 -m pytest .
```

### Testing against data from legacy dolfin

Some tests check the capability of reading data created with the legacy version of DOLFIN.
To create this dataset, start a docker container with legacy DOLFIN, for instance:

```bash
docker run -ti -v $(pwd):/root/shared -w /root/s
hared --rm ghcr.io/scientificcomputing/fenics:2024-02-19
```

Then, inside this container, call

```bash
python3 ./tests/create_legacy_data.py --output-dir=legacy
```

### Testing against data from older versions of io4dolfinx

Some tests check the capability to read data generated by `io4dolfinx<0.7.2`.
To generate data for these tests use the following commands:

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm ghcr.io/fenics/dolfinx/dolfinx:v0.7.3
```

Then, inside the container, call

```bash
python3 -m pip install io4dolfinx==0.7.1
python3 ./tests/create_legacy_checkpoint.py --output-dir=legacy_checkpoint
```
