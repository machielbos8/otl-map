"""Resampling a coarse tide model onto a fine global lattice (otl_map.resample)."""
import os

import numpy as np

from otl_map.resample import read_grid, resample_tidemodel, target_lattice, write_grid
from otl_map.paths import tidemodel_dir


def test_lattice_matches_fes2014b():
    t = target_lattice(0.0625)
    assert (t["nx"], t["ny"]) == (5760, 2881)
    assert t["x"][0] == 0.0 and t["y"][0] == 90.0 and t["y"][-1] == -90.0


def test_resample_schw(tmp_path, monkeypatch):
    monkeypatch.setenv("OTL_MAP_DATA", str(tmp_path))
    d = resample_tidemodel("SCHW", "SCHW_2", spacing=0.5, harmonics=["m2"])
    A = read_grid(os.path.join(d, "amp.m2"))
    P = read_grid(os.path.join(d, "pha.m2"))
    S = read_grid(os.path.join(tidemodel_dir("SCHW"), "amp.m2"))
    Sp = read_grid(os.path.join(tidemodel_dir("SCHW"), "pha.m2"))
    assert A["z"].shape == (361, 720)
    # a target node on a source cell centre in the open Pacific (0.5 N, 200.5 E) reproduces the source value
    j, i = int(round((90 - 0.5) / 0.5)), int(round(200.5 / 0.5))
    js, is_ = int(round(89.5 - 0.5)), int(round(200.5 - 0.5))
    assert abs(A["z"][j, i] - S["z"][js, is_]) < 1e-5
    assert abs(((P["z"][j, i] - Sp["z"][js, is_]) + 180) % 360 - 180) < 1e-3
    # central Asia stays land, south of the Schwiderski grid stays land
    assert A["z"][int((90 - 45) / 0.5), int(90 / 0.5)] >= 999
    assert (A["z"][-10:, :] >= 999).all()


def test_duplicate_seam_column_is_dropped(tmp_path):
    """FES2022b/TPXO10-atlas as distributed: cell centres 0..360 inclusive (nx = 360/sp + 1)."""
    from otl_map.resample import has_seam_duplicate
    sp = 30.0
    nx, ny = 13, 7                                       # centres 0, 30, ..., 360
    z = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    z[:, -1] = z[:, 0]
    xr, yr = np.array([-sp / 2, 360 + sp / 2]), np.array([-90 - sp / 2, 90 + sp / 2])
    write_grid(str(tmp_path / "amp.m2"), z, xr, yr, np.array([sp, sp]), "amplitude", "seam")
    assert has_seam_duplicate(nx, sp)
    g = read_grid(str(tmp_path / "amp.m2"))
    assert g["seam_dropped"] and g["nx"] == 12 and g["z"].shape == (7, 12)
    assert np.isclose(g["xr"][1], 360 - sp / 2) and np.isclose(g["x"][-1], 330.0)
    assert not read_grid(str(tmp_path / "amp.m2"), drop_seam=False)["seam_dropped"]
