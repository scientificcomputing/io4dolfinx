from mpi4py import MPI
import dolfinx
import io4dolfinx


# comm = MPI.COMM_WORLD
# mesh = dolfinx.mesh.create_unit_square(
#     comm, 10, 10, cell_type=dolfinx.mesh.CellType.quadrilateral
# )
# filename = "checkpoint.bp"
# io4dolfinx.write_mesh(filename, mesh)

# mesh_new = io4dolfinx.read_mesh(filename, comm)


filename = "thermal_steady_out.e"

mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")

tags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="cell", backend="exodus")


from dolfinx import plot
import pyvista

pyvista.set_jupyter_backend("html")

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
