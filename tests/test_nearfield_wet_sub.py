"""wet_sub: point-load sub-cells per side for whole wet cells beyond the 0.15 deg fit range (nearfield._fine).

Bundles without the key keep the old fixed 10 (the deployed FES/TPXO bundles); build_bundle writes 10 up to
1/16 deg and 40 for coarser lattices (GOT5.5), see paper_JoG/paper_scripts/nf_coarse_grid_owncell_test.py.
"""
import json

import numpy as np

from otl_map.bundle import wet_sub_for
from otl_map.nearfield import WET_SUB, Bundle


def _tiny_bundle(path, sp=0.125, wet_sub=None):
    nx, ny, M = int(round(360 / sp)), int(round(180 / sp)) + 1, 5
    meta = dict(model="T", nx=nx, ny=ny, x0=0.0, y0=90.0, sp=sp, ncell=4, M=M, cf=8, harmonics=["m2"],
                earth_models=["PREM"], rho=1030.0)
    if wet_sub is not None:
        meta["wet_sub"] = wet_sub
    path.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(path / "meta.json", "w"))
    for n, a in (("idx", np.zeros(1, np.int64)), ("w0", np.ones(1, np.float32)), ("tide", np.ones((1, 1), np.complex64)),
                 ("sptr", np.full(1, -1, np.int32)), ("sub", np.zeros((1, 64), np.uint8))):
        np.save(path / f"{n}.npy", a)
    np.save(path / "coarse_PREM.npy", np.zeros((1, 3, 2 * M + 1, 2 * M + 1)))
    return path


def test_rule():
    assert wet_sub_for(1 / 30) == 10 and wet_sub_for(0.0625) == 10 and wet_sub_for(0.125) == 40


def test_default_and_override(tmp_path):
    assert Bundle(str(_tiny_bundle(tmp_path / "old")), "PREM").wet_sub == WET_SUB == 10
    assert Bundle(str(_tiny_bundle(tmp_path / "new", wet_sub=40)), "PREM").wet_sub == 40


def test_owncell_on_coarse_lattice(tmp_path):
    """P off-centre in a 1/8 deg cell: its own cell fails the fit-range rule and is point loads.  10 -> 40
    changes the vertical kernel of the 3x3 cells around P by ~0.02 mm per m of water (the 10x10 error)."""
    b10 = Bundle(str(_tiny_bundle(tmp_path / "a")), "PREM")
    b40 = Bundle(str(_tiny_bundle(tmp_path / "b", wet_sub=40)), "PREM")
    lon, lat = 10.06, 45.06
    ci, cj = int(round(lon / 0.125)), int(round((90 - lat) / 0.125))
    di, dj = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    II, JJ = (ci + di).ravel(), (cj + dj).ravel()
    pos = np.zeros(len(II), int)                          # every cell -> stored cell 0 (sptr -1: whole, wet)
    d = (b10._fine(lon, lat, II, JJ, pos)[0].sum() - b40._fine(lon, lat, II, JJ, pos)[0].sum()) * 1e3
    assert 0.01 < abs(d) < 0.05, d
