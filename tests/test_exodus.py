import urllib.request
from enum import Enum

from mpi4py import MPI

import pytest

import io4dolfinx

netcdf4 = pytest.importorskip("netCDF4")


class DownloadStatus(Enum):
    success = 1
    failed = -1
    no_connection = -2


def download_file_if_not_exists(
    url, filename, comm: MPI.Intracomm = MPI.COMM_WORLD, rank: int = 0
) -> DownloadStatus:
    status = DownloadStatus.failed
    if comm.rank == rank:
        if not filename.exists():
            try:
                urllib.request.urlretrieve(url, filename)
                status = DownloadStatus.success
            except urllib.error.URLError as e:
                if str(e) == "<urlopen error [Errno -3] Temporary failure in name resolution>":
                    status = DownloadStatus.no_connection
                else:
                    status = DownloadStatus.failed
        else:
            status = DownloadStatus.success
    status = comm.bcast(status, root=rank)
    comm.Barrier()
    return status


def test_read_mesh_and_cell_data(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = tmp_path / "openmc_master_out_openmc0.e"
    url = "https://github.com/neams-th-coe/cardinal/blob/devel/test/tests/neutronics/feedback/single_level/gold/openmc_master_out_openmc0.e?raw=true"
    status = download_file_if_not_exists(url, filename)
    if status == DownloadStatus.no_connection:
        pytest.skip("No internet connection")
    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")
    io4dolfinx.read_meshtags(filename, mesh, meshtag_name="cell", backend="exodus")
    io4dolfinx.read_meshtags(filename, mesh, meshtag_name="facet", backend="exodus")
    io4dolfinx.read_cell_data(
        filename, name="cell_temperature", mesh=mesh, backend="exodus", time=1.0
    )


def test_read_mesh_point_data(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    filename = tmp_path / "openmc_master_out_openmc0.e"
    url = "https://github.com/idaholab/moose/blob/next/test/tests/kernels/2d_diffusion/gold/matdiffusion_out.e?raw=true"
    status = download_file_if_not_exists(url, filename)
    if status == DownloadStatus.no_connection:
        pytest.skip("No internet connection")

    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")
    io4dolfinx.read_point_data(filename, name="u", mesh=mesh, backend="exodus", time=1.0)
