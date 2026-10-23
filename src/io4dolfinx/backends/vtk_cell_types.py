import numpy as np

# Cell types can be found at
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html

_first_order_vtk = {
    1: "point",
    3: "interval",
    5: "triangle",
    9: "quadrilateral",
    10: "tetrahedron",
    12: "hexahedron",
}

_arbitrary_lagrange_vtk = {
    68: "interval",
    69: "triangle",
    70: "quadrilateral",
    71: "tetrahedron",
    72: "hexahedron",
    73: "prism",
    74: "pyramid",
}


def _cell_degree(cell_type: str, num_nodes: int) -> int:
    if cell_type == "point":
        return 1
    elif cell_type == "interval":
        return int(num_nodes - 1)
    elif cell_type == "triangle":
        degree = (np.sqrt(1 + 8 * num_nodes) - 1) / 2
        if 2 * num_nodes != degree * (degree + 1):
            raise ValueError(f"Unknown triangle layout. Number of nodes: {num_nodes}")
        return int(degree - 1)
    elif cell_type == "tetrahedron":
        degree = 0
        while degree * (degree + 1) * (degree + 2) < 6 * num_nodes:
            degree += 1
        if degree * (degree + 1) * (degree + 2) != 6 * num_nodes:
            raise ValueError(f"Unknown tetrahedron layout. Number of nodes: {num_nodes}")
        return int(degree - 1)
    elif cell_type == "quadrilateral":
        degree = np.sqrt(num_nodes)
        if num_nodes != degree * degree:
            raise ValueError(f"Unknown quadrilateral layout. Number of nodes: {num_nodes}")
        return int(degree - 1)
    elif cell_type == "hexahedron":
        degree = np.cbrt(num_nodes)
        if num_nodes != degree * degree * degree:
            raise ValueError(f"Unknown hexahedron layout. Number of nodes: {num_nodes}")
        return int(degree - 1)
    elif cell_type == "prism":
        if num_nodes == 6:
            return 1
        elif num_nodes == 15:
            return 2
        raise ValueError(f"Unknown prism layout. Number of nodes: {num_nodes}")
    elif cell_type == "pyramid":
        if num_nodes == 5:
            return 1
        elif num_nodes == 13:
            return 2
        raise ValueError(f"Unknown pyramid layout. Number of nodes: {num_nodes}")
    raise ValueError(f"Unknown cell type {cell_type} with {num_nodes=}.")
