#!/usr/bin/env python3
"""Global refined OTL grids with the unified near field (replaces refine_otl_coast / refine_all).

At every grid node whose +-N patch holds a coastal cell:  u_ref = u_fft + NF_fine(node) - NF_coarse(node)
(otl_map.nearfield.Bundle.correction at the node; all harmonics and components at once).  Other nodes
keep u_fft.  Writes amp/pha_nfv2{comp}_{label}.{harm} next to the fft grids (int16, same encoding),
leaving the old refine grids untouched.

Run from the repo root:
  OTL_MAP=$PWD env/bin/python tools/refine_grid.py FES2014b PREM [PREM_disp STW105 STW105_disp]
Options: --workers N (default 8), --rundir DIR (default runs/<earth>_World_<model>; one Earth model only).
The note 'seconds_correction' is the time for the near-field corrections alone (paper Table 2);
'seconds' includes writing the grids.
"""
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc
import numpy as np

from build_refine_bundle_v2 import dilate  # noqa: E402
from otl_map._runlog import Run  # noqa: E402

_B = None


def _work(args):
    global _B
    bundle, earth, nodes = args
    if _B is None or _B[0] != (bundle, earth):
        from otl_map.nearfield import Bundle
        _B = ((bundle, earth), Bundle(bundle, earth))
    B = _B[1]
    out = []
    for i, j in nodes:
        c = B.correction(B.cell_lon(i), B.cell_lat(j))
        if c is not None:
            out.append((i, j, c.astype(np.complex64)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("earths", nargs="*", default=["PREM"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rundir")
    a = ap.parse_args()
    model, earths, workers = a.model, a.earths, a.workers
    if a.rundir and len(earths) != 1:
        raise SystemExit("--rundir needs exactly one Earth model")
    bundle = f"refine_bundle_v2/{model}"
    idx, sptr = np.load(f"{bundle}/idx.npy"), np.load(f"{bundle}/sptr.npy")
    import json
    meta = json.load(open(f"{bundle}/meta.json"))
    nx, ny, N, H = meta["nx"], meta["ny"], meta["ncell"], meta["harmonics"]
    c = np.zeros(nx * ny, bool)
    c[idx[sptr >= 0]] = True
    trig = dilate(c.reshape(ny, nx), N)
    jj, ii = np.nonzero(trig)
    nodes = list(zip(ii.tolist(), jj.tolist()))
    chunks = [nodes[k:k + 2000] for k in range(0, len(nodes), 2000)]
    for earth in earths:
        rd = a.rundir or f"runs/{earth}_World_{model}"
        inputs = [f"{bundle}/meta.json", f"{bundle}/coarse_{earth}.npy", "src/otl_map/nearfield.py"]
        inputs += [f"{rd}/amp_fft0_fft.m2"]
        with Run(f"unified refinement: global refined grids {model} {earth} (nfv2)",
                 params=dict(model=model, earth=earth, nodes=len(nodes), workers=workers, rundir=rd),
                 inputs=inputs, script=__file__) as run:
            t0 = time.time()
            corr = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for k, r in enumerate(ex.map(_work, [(bundle, earth, ch) for ch in chunks])):
                    corr.extend(r)
                    if k % 100 == 0:
                        print(f"  {earth}: {k}/{len(chunks)} chunks, {time.time() - t0:.0f} s", flush=True)
            t_corr = time.time() - t0
            I = np.array([x[0] for x in corr])
            J = np.array([x[1] for x in corr])
            Cc = np.stack([x[2] for x in corr])                  # (n, 3, nharm)
            JA = ny - 1 - J                                      # grid rows ascend in latitude
            for comp in range(3):
                for hi, h in enumerate(H):
                    src_a, src_p = f"{rd}/amp_fft{comp}_fft.{h}", f"{rd}/pha_fft{comp}_fft.{h}"
                    dst_a, dst_p = f"{rd}/amp_nfv2{comp}_fft.{h}", f"{rd}/pha_nfv2{comp}_fft.{h}"
                    shutil.copy(src_a, dst_a)
                    shutil.copy(src_p, dst_p)
                    da, dp = nc.Dataset(dst_a, "a"), nc.Dataset(dst_p, "a")
                    za, zp = np.array(da["z"][:], float), np.array(dp["z"][:], float)
                    z = za[JA, I] * np.exp(1j * np.radians(zp[JA, I])) + Cc[:, comp, hi]
                    za[JA, I], zp[JA, I] = np.abs(z), np.degrees(np.angle(z))
                    da["z"][:] = za
                    dp["z"][:] = zp
                    da.close()
                    dp.close()
                    run.output(dst_a)
                    run.output(dst_p)
            run.note(nodes_refined=int(len(corr)), seconds=round(time.time() - t0), seconds_correction=round(t_corr),
                     nodes_triggered=len(nodes), workers=workers, host=os.uname().nodename)
            print(f"{earth}: {len(corr):,} nodes refined ({len(nodes):,} triggered) in {t_corr:.0f} s "
                  f"({workers} workers), {time.time() - t0:.0f} s with writing", flush=True)


if __name__ == "__main__":
    main()
