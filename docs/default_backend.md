# Setting a Global Default Backend

By default, `io4dolfinx` uses the `h5py` backend for all read and write operations. It is possible to pass in the `backend` argument to individual function calls to override this default. For example the following code snippet will use the `adios2` backend for writing the first mesh, and use the default `h5py` backend for writing the second mesh.

```python
import dolfinx
from mpi4py import MPI
import io4dolfinx

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

# Write the mesh using the adios2 backend
io4dolfinx.write_mesh("first_mesh.bp", mesh, backend="adios2")
io4dolfinx.write_mesh("second_mesh.h5", mesh)  # Uses the default h5py backend
```
If you want to more explicit you could also pass in the `backend` argument to the second write operation, i.e

```python
io4dolfinx.write_mesh("second_mesh.h5", mesh, backend="h5py")
```
but it is not necessary since `h5py` is the default backend.

```{note}
Before io4dolfinx 2.0, the default backend was `adios2`. Code that relies on the default to
read or write ADIOS2 files should pass `backend="adios2"`, or set it as the default as below.
```

If you prefer to use a different backend (such as `adios2`) across your entire application without having to pass the `backend` argument to every individual function call, you can configure a global default.

## Usage

Use the `io4dolfinx.set_default_backend` function at the start of your script to change the backend globally.

```python
import dolfinx
from mpi4py import MPI
import io4dolfinx

# Set the global default backend to adios2
io4dolfinx.set_default_backend("adios2")

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

# The following write operation will automatically use the "adios2" backend
io4dolfinx.write_mesh("my_mesh.bp", mesh)
```
