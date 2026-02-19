from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import io4dolfinx

netcdf4 = pytest.importorskip("netCDF4")


def download_file_if_not_exists(url, filename):
    if not filename.exists():
        import urllib.request

        urllib.request.urlretrieve(url, filename)


def test_read_mesh_and_cell_data(tmp_path):
    filename = tmp_path / "openmc_master_out_openmc0.e"
    url = "https://github.com/neams-th-coe/cardinal/blob/devel/test/tests/neutronics/feedback/single_level/gold/openmc_master_out_openmc0.e?raw=true"
    download_file_if_not_exists(url, filename)

    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")
    celltags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="cell", backend="exodus")
    facet_meshtags = io4dolfinx.read_meshtags(
        filename, mesh, meshtag_name="facet", backend="exodus"
    )
    u = io4dolfinx.read_cell_data(
        filename, name="cell_temperature", mesh=mesh, backend="exodus", time=1.0
    )


def test_read_mesh_point_data(tmp_path):
    filename = tmp_path / "openmc_master_out_openmc0.e"
    url = "https://github.com/idaholab/moose/blob/next/test/tests/kernels/2d_diffusion/gold/matdiffusion_out.e?raw=true"
    download_file_if_not_exists(url, filename)

    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")
    point_data = io4dolfinx.read_point_data(
        filename, name="u", mesh=mesh, backend="exodus", time=1.0
    )
