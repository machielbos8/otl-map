"""otl-map prepare: one-cell extension over the coast and scaling by the water fraction."""
import numpy as np

from otl_map.prepare import extend_one_cell, prepare
from otl_map.resample import read_grid, target_lattice, write_grid


def test_extension_wraps_in_longitude_not_over_poles():
    u = np.zeros((4, 6))
    v = np.zeros((4, 6))
    land = np.ones((4, 6), bool)
    land[1, 0] = False
    u[1, 0], v[1, 0] = 2.0, -1.0
    land[0, 3] = False
    u[0, 3] = 4.0
    u2, v2, land2 = extend_one_cell(u, v, land)
    assert u2[1, 5] == 2.0 and v2[1, 5] == -1.0          # western neighbour across the 0/360 seam
    assert u2[0, 0] == 2.0 and u2[2, 1] == 2.0           # diagonal neighbours
    assert land2[3, 3] and not land2[1, 3]               # one pass only: row 3 not reached from row 0
    assert u2[1, 3] == 4.0 and u2[1, 4] == 4.0
    assert u2[0, 4] == 4.0                               # neighbour in the same (pole) row
    assert land2[2, 3]                                   # two rows away: still land


def test_prepare_scales_by_water_fraction(tmp_path, monkeypatch):
    monkeypatch.setenv("OTL_MAP_TIDEMODELS", str(tmp_path / "tm"))
    monkeypatch.setenv("OTL_MAP_CACHE", str(tmp_path / "cache"))
    t = target_lattice(30.0)                             # 12 x 7 cells
    amp = np.full((t["ny"], t["nx"]), 999.9)
    pha = np.full_like(amp, 999.9)
    amp[3, 2:5], pha[3, 2:5] = 1.0, 90.0
    (tmp_path / "tm" / "TOY").mkdir(parents=True)
    write_grid(str(tmp_path / "tm" / "TOY" / "amp.m2"), amp, t["xr"], t["yr"], t["sp"], "amplitude", "toy")
    write_grid(str(tmp_path / "tm" / "TOY" / "pha.m2"), pha, t["xr"], t["yr"], t["sp"], "phase lag", "toy")
    w = np.full_like(amp, 0.5)
    (tmp_path / "cache" / "masks").mkdir(parents=True)
    write_grid(str(tmp_path / "cache" / "masks" / "mask_TOY.nc"), w, t["xr"], t["yr"], t["sp"], "amplitude", "w")
    d = prepare("TOY")
    a, p = read_grid(f"{d}/amp.m2")["z"], read_grid(f"{d}/pha.m2")["z"]
    assert np.isclose(a[3, 3], 0.5) and np.isclose(p[3, 3], 90.0)
    assert np.isclose(a[3, 1], 0.5) and np.isclose(a[2, 3], 0.5)      # extended cells, scaled
    assert a[0, 0] == 0.0 and a[3, 8] == 0.0                          # beyond the extension: no tide
