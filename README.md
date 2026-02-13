# io4dolfinx - A framework for reading and writing data to various mesh formats

**io4dolfinx** is an extension for [DOLFINx](https://github.com/FEniCS/dolfinx/) that provides advanced input/output capabilities. It focuses on **N-to-M checkpointing** (writing data on N processors, reading on M processors) and supports reading/writing various mesh formats using interchangeable backends.


## Installation

The library is compatible with the DOLFINx nightly release, v0.10.0, and v0.9.0.

```bash
python3 -m pip install io4dolfinx
```

For specific backend requirements (like ADIOS2 or H5PY), see the [Installation Guide](./docs/installation.md).


## Quick Start

Here is a minimal example of saving and loading a simulation state (Checkpointing).

```python
from pathlib import Path
from mpi4py import MPI
import dolfinx
import io4dolfinx

# 1. Create a mesh and function
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_unit_square(comm, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: x[0] + x[1])
u.name = "my_function"

# 2. Write checkpoint
# The mesh must be written before the function
filename = Path("checkpoint.bp")
io4dolfinx.write_mesh(filename, mesh)
io4dolfinx.write_function(filename, u, time=0.0)

# 3. Read checkpoint
# This works even if the number of MPI processes changes (N-to-M)
mesh_new = io4dolfinx.read_mesh(filename, comm)
V_new = dolfinx.fem.functionspace(mesh_new, ("Lagrange", 1))
u_new = dolfinx.fem.Function(V_new)
io4dolfinx.read_function(filename, u_new, time=0.0, name="my_function")
```


## Features and Backends

`io4dolfinx` supports custom user backends. You can switch backends by passing `backend="name"` to IO functions.

### Checkpointing (N-to-M)
Many finite element applications requires storage of functions that cannot be associated with the nodes or cells of the mesh. Therefore, we have implemented our own, native checkpointing format that supports N-to-M checkpointing (write data on N processors, read in on M) through the following backends:

- [h5py](./docs/backends/h5py.rst): Requires HDF5 with MPI support to work, but can store, meshes, partitioning info, meshtags, function data and more.
- [adios2](./docs/backends/adios2.rst): Requires [ADIOS 2](https://adios2.readthedocs.io/en/latest/) compiled with MPI support and Python bindings. Supports the same set of operations as the `h5py` backend.

The code uses the ADIOS2/Python-wrappers and h5py module to write DOLFINx objects to file, supporting N-to-M (_recoverable_) and N-to-N (_snapshot_) checkpointing.
See: [Checkpointing in DOLFINx - FEniCS 23](https://jsdokken.com/checkpointing-presentation/#/) or the examples in the [Documentation](https://jsdokken.com/io4dolfinx/) for more information.

For scalability, the code uses [MPI Neighbourhood collectives](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node200.htm) for communication across processes.


### Mesh IO (Import/Export)
Most meshing formats supports associating data with the nodes of the mesh (the mesh can be higher order) and the cells of the mesh. The node data can be read in as P-th order Lagrange functions (where P is the order of the grid), while the cell data can be read in as piecewise constant (DG-0) functions.

- [VTKHDF](./docs/backends/vtkhdf.rst): The new scalable format from VTK, called [VTKHDF](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/index.html) is supported by the `vtkhdf` backend. 
- [XDMF](./docs/backends/xdmf.rst) (eXtensible Model Data Format): `.xdmf`. The `xdmf` backend supports the `HDF5` encoding, to ensure performance in parallel.
- [PyVista](./docs/backends/pyvista.rst) (IO backend is meshio): The [pyvista](https://pyvista.org/) backend uses {py:func}`pyvista.read` to read in meshes, point data and cell data. `pyvista` relies on [meshio](https://github.com/nschloe/meshio) for most reading operations (including the XDMF ascii format).



## Advanced Usage

The repository contains detailed documented examples in the `docs` folder:

* [Reading and writing mesh checkpoints](./docs/writing_mesh_checkpoint.py)
* [Storing mesh partitioning data](./docs/partitioned_mesh.py) (Avoid re-partitioning when restarting)
* [Writing mesh-tags](./docs/meshtags.py)
* [Writing function checkpoints](./docs/writing_functions_checkpoint.py)
* [Checkpoint on input mesh](./docs/original_checkpoint.py)

For a full API reference and backend details, see the [Documentation](https://jsdokken.com/io4dolfinx/).

### Legacy DOLFIN Support
`io4dolfinx` can read checkpoints created by the legacy version of DOLFIN (Lagrange or DG functions).
* Reading meshes from DOLFIN HDF5File-format.
* Reading checkpoints from DOLFIN HDF5File and XDMFFile.

## Project Background

### Relation to adios4dolfinx
This library is an evolution of [adios4dolfinx](https://doi.org/10.21105/joss.06451). It includes all functionality of the original library but has been refactored to support multiple IO backends (not just ADIOS2), making it easier to interface with different meshing formats while keeping the library structure sane.

### Statement of Need
As large-scale, long-running simulations on HPC clusters become more common, the need to store intermediate solutions is crucial. If a system crashes or a computational budget is exceeded, checkpoints allow the simulation to resume without restarting from scratch. `io4dolfinx` extends DOLFINx with this essential functionality.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Testing

`io4dolfinx` includes a comprehensive test suite that ensures functionality across different backends and compatibility with legacy data formats, see the [Testing Guide](./docs/testing.md) for details.


## LICENSE
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.