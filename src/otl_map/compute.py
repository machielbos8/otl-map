"""OTL maps with the row-wise FFT (paper, Sect. 2): a wrapper around otl_map.compute_otl_fft without a hand-written
configuration file.

compute(model, earth, ...) writes, for every component, the configuration the FFT program expects into the run
directory and runs it there; the grids appear as amp/pha_fft{comp}_{label}.<harm> (mm and degrees, int16 with a
0.01 scale factor, rows from south to north). By default the mesh is the global lattice of the tide model and
the tide model is the prepared <model>_mod_ext (`otl-map prepare`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from otl_map.paths import greenfnc_dir, tidemodel_dir
from otl_map.resample import read_grid

EARTH_MODELS = ("PREM", "PREM_disp", "STW105", "STW105_disp")
COMPONENTS = {0: "vertical", 1: "north", 2: "east"}


def global_mesh(model_dir):
    """[lon0, lon1, lat0, lat1, dlon, dlat] of the cell centres of a global tide model."""
    g = read_grid(os.path.join(model_dir, sorted(f for f in os.listdir(model_dir) if f.startswith("amp."))[0]))
    sp = float(g["sp"][0])
    return [float(g["x"][0]), float(g["x"][-1]), float(g["y"][-1]), float(g["y"][0]), sp, float(g["sp"][1])]


def compute(model, earth="PREM", rundir=".", components=(0, 1, 2), harmonics=None, mesh=None, label="fft",
            prepared=True, workers=None, frame="CE", verbose=False):
    """Run the FFT for each component; returns the list of grids written.

    model: tide model name (its prepared version <model>_mod_ext is used unless prepared=False);
    earth: PREM, PREM_disp, STW105 or STW105_disp; frame: CE (default) or CM Green's functions;
    mesh: [lon0, lon1, lat0, lat1, dlon, dlat] of the output nodes (default: the global tide-model lattice)."""
    if earth not in EARTH_MODELS:
        raise ValueError(f"earth model {earth!r} not in {EARTH_MODELS}")
    tm = f"{model}_mod_ext" if prepared else model
    tdir = tidemodel_dir(tm)
    if harmonics is None:
        harmonics = sorted(f.split(".", 1)[1] for f in os.listdir(tdir) if f.startswith("amp."))
    gf = f"{earth}.dat" if frame == "CE" else f"{earth}_CM.dat"
    consts = json.load(open(os.path.join(greenfnc_dir(), "greensfunction_approx.json")))[gf]
    mesh = mesh or global_mesh(tdir)
    os.makedirs(rundir, exist_ok=True)
    env = dict(os.environ)
    if workers:
        env["OTL_MAP_WORKERS"] = str(workers)
    written = []
    for c in components:
        cfg = dict(greensfunction=gf, component=int(c), verbose=bool(verbose), label=label, tidemodel=tm,
                   harmonics=list(harmonics), mesh=[float(x) for x in mesh], rho=1030.0,
                   constants=[float(x) for x in consts[str(c)]])
        fname = os.path.join(rundir, f"otl_map_c{c}.json")
        json.dump(cfg, open(fname, "w"), indent=1)
        cmd = [sys.executable, "-c", "from otl_map.compute_otl_fft import main; main()", "-i", os.path.basename(fname)]
        r = subprocess.run(cmd, cwd=rundir, env=env, capture_output=not verbose, text=True)
        if r.returncode:
            raise RuntimeError(f"compute_otl_fft failed for component {c}:\n{(r.stderr or '')[-2000:]}")
        for h in harmonics:
            for t in ("amp", "pha"):
                f = os.path.join(rundir, f"{t}_fft{c}_{label}.{h}")
                if not os.path.isfile(f):
                    raise RuntimeError(f"compute_otl_fft did not write {f}")
                written.append(f)
    return written
