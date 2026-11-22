#!/usr/bin/env python3
"""Bundle v2 for the unified coastal refinement (docs/code/unified_refinement_plan.md).

    refine_bundle_v2/<model>/
      meta.json            grid (nx, ny, x0/y0 = first cell centre, sp), ncell N, cf, M = N + 1, harmonics
      idx.npy    int64     sorted linear cell index j*nx + i of every stored cell
      w0.npy     float32   cell water fraction (mask_<model>.nc, quadtree)
      tide.npy   complex64 (n, nharm) u + i v of the modified tide model, exactly as compute_otl_fft
                           reads it (amp*cos(pha), amp*sin(pha); land 999 -> 0)
      sptr.npy   int32     row into sub.npy for a coastal cell (1e-6 < w0 < 1-1e-6), else -1
      sub.npy    uint8     (n_coastal, cf*cf) quadtree sub-cell water fraction * 255 (cpp/create_subcells),
                           rows N->S, columns W->E
      coarse_<GF>.npy float64 (ny, 3, 2M+1, 2M+1)  FFT kernel element for a node in tide row j and the
                           cell at row offset t, column offset di (coarse_kernel.table_for_row)

Stored cells: every cell within 2N columns/rows (Chebyshev, wrapping in longitude) of a coastal cell, so
the +-N patch of any station whose patch contains a coastal cell is complete.

Run from the repo root:
  OTL_MAP=$PWD env/bin/python tools/build_refine_bundle_v2.py FES2014b PREM [STW105 ...]
"""
import json
import math
import os
import sys
from glob import glob
from multiprocessing import Pool

import netCDF4 as nc
import numpy as np

from otl_map.subcells_io import read_subcells  # noqa: E402
from otl_map._runlog import Run  # noqa: E402

HARMS = ["m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1", "mf", "mm", "ssa"]
GF_FILE = {"PREM": "PREM.dat", "PREM_disp": "PREM_disp.dat", "STW105": "STW105.dat", "STW105_disp": "STW105_disp.dat"}


def read_grd(path):
    d = nc.Dataset(path)
    nx, ny = (int(v) for v in d["dimension"][:])
    z = np.array(d["z"][:], np.float32).reshape(ny, nx)
    xr, yr, sp = (np.array(d[k][:], float) for k in ("x_range", "y_range", "spacing"))
    d.close()
    return z, xr, yr, sp


def dilate(a, r):
    """Box (Chebyshev) dilation by r cells, wrapping in longitude, not across the poles."""
    out = a.copy()
    for _ in range(r):
        h = out | np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1)
        v = h.copy()
        v[1:] |= h[:-1]
        v[:-1] |= h[1:]
        out = v
    return out


_G = {}


def _rows(args):
    gf_file, rows, cy1, csp, M, ny = args
    from otl_map.coarse_kernel import load_gf, table_for_row
    if gf_file not in _G:
        C = json.load(open("greenfnc/greensfunction_approx.json"))[gf_file]
        _G[gf_file] = ([load_gf(gf_file, c) for c in range(3)], [C[str(c)] for c in range(3)])
    gfs, ccs = _G[gf_file]
    out = []
    for j in rows:
        lat = math.degrees(cy1 - (j + 0.5) * csp)
        if abs(lat) > 89.5:
            out.append((j, np.zeros((3, 2 * M + 1, 2 * M + 1))))
        else:
            out.append((j, table_for_row(gfs, ccs, j, cy1, csp, csp, M, cny=ny)))
    return out


def main():
    model, earths = sys.argv[1], sys.argv[2:] or ["PREM"]
    # OTL_BUNDLE_NCELL: patch half-width in cells for a sensitivity test (default 0.25 deg); writes to
    # refine_bundle_v2_np<N>/<model> so the production bundle is untouched.
    NCELL = os.environ.get("OTL_BUNDLE_NCELL")
    out = f"refine_bundle_v2/{model}" if not NCELL else f"refine_bundle_v2_np{NCELL}/{model}"
    tide_dir = f"surfaceloads/{model}_mod_ext"
    parts = sorted(glob(f"cpp/subcells_{model}/part_*.bin"))
    inputs = [f"cpp/mask_{model}.nc", f"{tide_dir}/amp.m2", "greenfnc/greensfunction_approx.json",
              "src/otl_map/coarse_kernel.py"] + parts + [f"greenfnc/{GF_FILE[e]}" for e in earths]
    with Run(f"unified refinement: bundle v2 for {model} ({', '.join(earths)})",
             params=dict(model=model, earth_models=earths, ncell=NCELL or "default"), inputs=inputs, script=__file__) as run:
        w0, xr, yr, sp = read_grd(f"cpp/mask_{model}.nc")
        ny, nx = w0.shape
        spx = float(sp[0])
        if abs(nx * spx - 360.0) > 1e-6:      # e.g. 10801 columns with centres 0..360: see tools/cut_seam_column.py
            raise SystemExit(f"{model}: {nx} columns x {spx} deg != 360 deg (duplicate seam column?)")
        N = int(NCELL) if NCELL else max(4, int(round(0.25 / spx)))
        M = N + 1
        S = [read_subcells(p) for p in parts]
        cf = S[0]["cf"]
        ks = np.concatenate([s["k"] for s in S])
        fs = np.concatenate([s["f"] for s in S])
        o = np.argsort(ks)
        ks, fs = ks[o], fs[o]
        coastal = np.zeros(ny * nx, bool)
        coastal[ks] = True
        coastal = coastal.reshape(ny, nx)
        keep = dilate(coastal, 2 * N)
        idx = np.flatnonzero(keep.ravel()).astype(np.int64)
        print(f"{model}: {nx}x{ny}, N={N}, cf={cf}, coastal {len(ks):,}, stored {len(idx):,}", flush=True)
        sptr = np.full(len(idx), -1, np.int32)
        pos = np.searchsorted(idx, ks)
        assert np.all(idx[pos] == ks)
        sptr[pos] = np.arange(len(ks), dtype=np.int32)
        harms = [h for h in HARMS if os.path.exists(f"{tide_dir}/amp.{h}")]
        tide = np.zeros((len(idx), len(harms)), np.complex64)
        for hi, h in enumerate(harms):
            a, _, _, _ = read_grd(f"{tide_dir}/amp.{h}")
            p, _, _, _ = read_grd(f"{tide_dir}/pha.{h}")
            a, p = a.ravel()[idx].astype(float), p.ravel()[idx].astype(float)
            land = ~(a < 999.0)
            u = a * np.cos(np.radians(p))
            v = a * np.sin(np.radians(p))
            u[land] = v[land] = 0.0
            tide[:, hi] = u + 1j * v
        os.makedirs(out, exist_ok=True)
        for name, arr in (("idx", idx), ("w0", w0.ravel()[idx].astype(np.float32)), ("tide", tide),
                          ("sptr", sptr), ("sub", fs)):
            np.save(f"{out}/{name}.npy", arr)
            run.output(f"{out}/{name}.npy")
        cy1, csp = math.radians(float(yr[1])), math.radians(spx)
        for e in earths:
            T = np.zeros((ny, 3, 2 * M + 1, 2 * M + 1))
            chunks = [list(range(j, min(j + 40, ny))) for j in range(0, ny, 40)]
            with Pool(min(8, os.cpu_count() or 1)) as pool:
                for res in pool.imap_unordered(_rows, [(GF_FILE[e], c, cy1, csp, M, ny) for c in chunks]):
                    for j, t in res:
                        T[j] = t
            np.save(f"{out}/coarse_{e}.npy", T)
            run.output(f"{out}/coarse_{e}.npy")
            print(f"  coarse table {e} done", flush=True)
        meta = dict(model=model, nx=nx, ny=ny, x0=float(xr[0]) + spx / 2, y0=float(yr[1]) - spx / 2, sp=spx,
                    ncell=N, M=M, cf=cf, harmonics=harms, earth_models=earths, rho=1030.0,
                    coastal_cells=int(len(ks)), stored_cells=int(len(idx)), tide_model=f"{model}_mod_ext")
        json.dump(meta, open(f"{out}/meta.json", "w"), indent=2)
        run.output(f"{out}/meta.json")
        run.note(**{k: meta[k] for k in ("ncell", "cf", "coastal_cells", "stored_cells")})


if __name__ == "__main__":
    main()
