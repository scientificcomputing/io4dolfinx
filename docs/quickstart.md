# Quick Start Guide

This document provides a quick start overview of the functions available in
`io4dolfinx`. The library is designed to extend DOLFINx with advanced Input/Output
capabilities, focusing on flexible checkpointing and support for various data formats.

## Core Checkpointing

The primary purpose of `io4dolfinx` is to support **N-to-M checkpointing**. This means
you can run a simulation on $N$ processes, save the state, and restart the simulation
on $M$ processes.

### Meshes

Before storing any functions, the mesh must be written to the checkpoint file.
The mesh topology and geometry are saved in a distributed format.

```python
from mpi4py import MPI
import dolfinx
import io4dolfinx
```

```python
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_unit_square(comm, 10, 10)
filename = "checkpoint.bp"
```

Write mesh to file

```python
#
io4dolfinx.write_mesh(filename, mesh)
```

Read mesh from file. The mesh is redistributed according to the current communicator size.

```python
mesh_new = io4dolfinx.read_mesh(filename, comm)
```

### Functions

Functions can be stored associated with a timestamp. They effectively store the
coefficients of the finite element function.

```python
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = dolfinx.fem.Function(V)
u.name = "my_solution"
```

Write function

```python
io4dolfinx.write_function(filename, u, time=0.5)
```

Read function
```{note}
You must read the mesh first (or have a compatible mesh ready, see [Checkpoint on input mesh](./original_checkpoint.py) for details).
````

```python
u_new = dolfinx.fem.Function(V)
io4dolfinx.read_function(filename, u_new, time=0.5, name="my_solution")
```

## Mesh Tags and Data

`io4dolfinx` supports storing auxiliary data associated with the mesh, such as subdomain
markers (`MeshTags`) or raw data arrays.

### MeshTags

MeshTags (markers for cells, facets, etc.) can be written to the same checkpoint file
as the mesh. They are re-distributed correctly when reading back on a different number of processes.

Create some dummy tags
```python
subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, [0], [1])
```

Write tags
```python
io4dolfinx.write_meshtags(filename, mesh, subdomains, meshtag_name="subdomains")
```

Read tags
```python
tags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="subdomains")
```

## Advanced Checkpointing Strategies

Beyond standard N-to-M checkpointing, the library offers specialized strategies for
specific use cases.

### Snapshot Checkpointing

A **snapshot** is a lightweight checkpoint intended for use within the *same* simulation
run (N-to-N). It is ideal for temporary storage (e.g., for an adjoint solver or
saving state before a risky operation) where you know the process count and mesh partitioning
will not change.

```python
snapshot_file = "temp_snapshot.bp"
io4dolfinx.snapshot_checkpoint(u, snapshot_file, io4dolfinx.FileMode.write)
```

Read back (must be on same mesh distribution)
```python
io4dolfinx.snapshot_checkpoint(u, snapshot_file, io4dolfinx.FileMode.read)
```

See the [Snapshot Checkpointing Guide](./snapshot_checkpoint.py) for more details and examples.

### Original Mesh Checkpointing

Sometimes you want to save a solution that corresponds exactly to the input mesh file
(e.g., an `.xdmf` file you started with), rather than the current partitioned mesh.
This is useful for visualization or post-processing on the original geometry.

```python
io4dolfinx.write_function_on_input_mesh("solution_on_input.bp", u)
```

See the [Checkpoint on input mesh](./original_checkpoint.py) for more details and examples.

## Legacy DOLFIN Support

The library provides readers for migrating data from legacy DOLFIN (FEniCS).

* **`read_mesh_from_legacy_h5`**: Reads a mesh from a legacy HDF5 file.
* **`read_function_from_legacy_h5`**: Reads a function from a legacy HDF5 file
    (supports both `HDF5File` and `XDMFFile` archives).

See the [reading_legacy_data.md](./reading_legacy_data.md) guide for detailed examples.



## Metadata and Utilities

Helper functions are available to query the contents of a checkpoint file.

* **`read_function_names`**: Returns a list of all functions stored in a file.

* **`read_timestamps`**: Returns the time steps available for a specific function.

* **`read_attributes` / `write_attributes`**: Allows storing arbitrary metadata dictionaries.


## Backends

`io4dolfinx` is backend-agnostic. You can choose the storage engine by passing
the `backend` argument to most functions.

1.  **`adios2` (Default)**: Uses the ADIOS2 library. Best for large-scale parallel IO.
    Supports engines like "BP4", "BP5", and "HDF5".


2.  **`h5py`**: Uses the standard HDF5 library via `h5py`. Requires an MPI-enabled HDF5 build.
    Good for compatibility with other HDF5 tools.


3.  **`vtkhdf`**: Supports reading and writing the VTKHDF format (scalable VTK).


4.  **`pyvista`**: Primarily for reading unstructured grids (`.vtu`) via PyVista/meshio.


5.  **`xdmf`**: Basic support for reading XDMF data.


