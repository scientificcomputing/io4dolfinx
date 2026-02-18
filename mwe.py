from mpi4py import MPI
import dolfinx
import io4dolfinx


# comm = MPI.COMM_WORLD
# mesh = dolfinx.mesh.create_unit_square(comm, 2, 2, cell_type=dolfinx.mesh.CellType.quadrilateral)
# entities = [0, 1, 2, 3]
# values = [1, 2, 3, 4]
# subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, entities, values)
# filename = "checkpoint.bp"
# io4dolfinx.write_mesh(filename, mesh)
# io4dolfinx.write_meshtags(filename, mesh, subdomains, meshtag_name="subdomains")

# mesh_new = io4dolfinx.read_mesh(filename, comm)
# tags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="subdomains")

filename = "thermal_steady_out.e"

mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")

celltags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="cell", backend="exodus")

facet_meshtags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="facet", backend="exodus")
values = facet_meshtags.values

from dolfinx import plot
import pyvista

pyvista.set_jupyter_backend("html")

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.cell_data["Cell Marker"] = celltags.values
grid.set_active_scalars("Cell Marker")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")

fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(mesh, fdim, facet_meshtags.indices)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = facet_meshtags.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_markers.png")
