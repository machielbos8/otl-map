"""FFT map = brute-force summation (vertical), and compute -> stations -> BLQ end to end, on Schwiderski
resampled to 0.5 deg (no coastline file needed)."""
import json
import os
import subprocess
import sys

import netCDF4 as nc
import numpy as np
import pytest

from otl_map.compute import compute
from otl_map.paths import greenfnc_dir
from otl_map.resample import resample_tidemodel
from otl_map import stations as S

MESH = [200.0, 201.0, 0.0, 0.5, 0.5, 0.5]         # 3 x 2 nodes in the open equatorial Pacific (ocean all around)


@pytest.fixture(scope="module")
def schw2(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    mp = pytest.MonkeyPatch()
    mp.setenv("OTL_MAP_DATA", str(d))
    resample_tidemodel("SCHW", "SCHW_2", spacing=0.5, harmonics=["m2"])
    run = d / "run"
    compute("SCHW_2", "PREM", str(run), components=[0], harmonics=["m2"], mesh=MESH, prepared=False, workers=2)
    mp.undo()
    return d, run


def test_fft_equals_brute_force(schw2, monkeypatch):
    d, run = schw2
    monkeypatch.setenv("OTL_MAP_DATA", str(d))
    C = json.load(open(os.path.join(greenfnc_dir(), "greensfunction_approx.json")))["PREM.dat"]["0"]
    cfg = dict(tidemodel="SCHW_2", greensfunction="PREM.dat", component=0, harmonics=["m2"], mesh=MESH,
               rho=1030.0, verbose=False, label="brute", constants=C)
    json.dump(cfg, open(run / "slow.json", "w"))
    r = subprocess.run([sys.executable, "-c", "from otl_map.compute_otl_slow import main; main()", "-i", "slow.json"],
                       cwd=run, capture_output=True, text=True, env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-1500:]
    fa = nc.Dataset(run / "amp_fft0_fft.m2")
    fp = nc.Dataset(run / "pha_fft0_fft.m2")
    ba = nc.Dataset(run / "amp_brute.nc")
    bp = nc.Dataset(run / "pha_brute.nc")
    F = np.array(fa["z"][:], float) * np.exp(1j * np.radians(np.array(fp["z"][:], float)))
    B = np.array(ba["z"][:], float) * np.exp(1j * np.radians(np.array(bp["z"][:], float)))
    assert F.shape == B.shape == (2, 3)
    assert np.abs(B).min() > 1.0                       # real loading signal (mm), ocean all around the nodes
    # FFT grids are stored to 0.01 mm / 0.01 deg; signals here are up to ~10 mm
    assert np.abs(F - B).max() < 0.01 + 2e-4 * np.abs(B).max()


def test_stations_end_to_end(schw2, tmp_path, monkeypatch):
    d, run = schw2
    monkeypatch.setenv("OTL_MAP_DATA", str(d))
    st = [dict(name="NODE", lon=201.0, lat=0.0, h=0.0), dict(name="MID", lon=200.25, lat=0.25, h=0.0)]
    res = S.compute_stations(st, "SCHW_2", "PREM", str(run), ["m2"], nearfield=False, components=[0])
    g = nc.Dataset(run / "amp_fft0_fft.m2")
    j = int(np.argmin(np.abs(g["y"][:] - 0.0)))
    i = int(np.argmin(np.abs(g["x"][:] - 201.0)))
    assert abs(abs(res["NODE"]["m2"][0]) - float(g["z"][j, i])) < 1e-6   # a station on a node returns the node
    blq = S.to_blq_convention({k: {h: np.array([v[0], v[0], v[0]]) for h, v in r.items()} for k, r in res.items()})
    f = S.write_blq(str(tmp_path / "t.blq"), st, blq, "SCHW_2", "PREM")
    txt = open(f).read()
    assert "$$ END TABLE" in txt and "  NODE" in txt and "South and West" in txt
