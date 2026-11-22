"""Command-line interface `otl-map` (skeleton: subcommands are being ported, see docs/ROADMAP.md)."""
from __future__ import annotations

import argparse
import sys

from otl_map import __version__

SUBCOMMANDS = {
    "resample": "put a (coarse) tide model on a fine global grid, e.g. Schwiderski on 1/16 deg",
    "fetch-coastline": "download and verify the full-resolution coastline (Zenodo)",
    "masks": "water fractions of the cells (and sub-cells of coastal cells) of a tide-model grid",
    "prepare": "water fractions, scaling and one-cell extension of a tide model",
    "compute": "FFT map for given harmonics, components and region",
    "bundle": "near-field bundle for a tide model and Earth model",
    "refine": "refined grid at the coastal nodes (Eq. 6)",
    "stations": "OTL at stations (BLQ/JSON), optionally with the centre-of-mass correction",
    "fit-greens": "pole-plus-polynomial constants for a user Green's function",
}


def main(argv=None):
    p = argparse.ArgumentParser(prog="otl-map", description="Global ocean tide loading maps")
    p.add_argument("--version", action="version", version=f"otl-map {__version__}")
    sub = p.add_subparsers(dest="cmd")
    sps = {name: sub.add_parser(name, help=help_) for name, help_ in SUBCOMMANDS.items()}
    r = sps["resample"]
    r.add_argument("model", help="tide model name (e.g. SCHW)")
    r.add_argument("-o", "--out", help="name of the resampled model (default <model>_<n>, n = cells per degree)")
    r.add_argument("-s", "--spacing", type=float, default=0.0625, help="grid spacing in degrees (default 1/16)")
    r.add_argument("--harmonics", nargs="+", help="harmonics (default: all found)")
    r.add_argument("--fill-cells", type=float, default=2.0,
                   help="fill water near the coast from the nearest water node within this many source cells")
    pr = sps["prepare"]
    pr.add_argument("model")
    pr.add_argument("--harmonics", nargs="+")
    pr.add_argument("--no-extend", action="store_true", help="scale only (output <model>_mod)")
    b = sps["bundle"]
    b.add_argument("model")
    b.add_argument("earths", nargs="*", default=["PREM"], help="PREM, PREM_disp, STW105, STW105_disp")
    b.add_argument("--workers", type=int)
    rf = sps["refine"]
    rf.add_argument("model")
    rf.add_argument("earth")
    rf.add_argument("--rundir", required=True, help="directory with the FFT grids of `otl-map compute`")
    rf.add_argument("--workers", type=int)
    m = sps["masks"]
    m.add_argument("model", help="tide model name (its amp.m2 defines the grid)")
    m.add_argument("--cf", type=int, default=60, help="sub-cells per cell side for coastal cells (default 60)")
    m.add_argument("--no-subcells", action="store_true", help="water fractions only")
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        return 0
    if a.cmd == "resample":
        from otl_map.resample import resample_tidemodel
        out = a.out or f"{a.model}_{int(round(1 / a.spacing))}"
        d = resample_tidemodel(a.model, out, a.spacing, a.harmonics, a.fill_cells)
        print(f"{a.model} -> {out} ({a.spacing:g} deg): {d}")
        return 0
    if a.cmd == "prepare":
        from otl_map.prepare import prepare
        d = prepare(a.model, a.harmonics, extend=not a.no_extend)
        print(f"{a.model} -> {d}")
        return 0
    if a.cmd == "bundle":
        from otl_map.bundle import build_bundle
        print(f"bundle -> {build_bundle(a.model, a.earths, workers=a.workers)}")
        return 0
    if a.cmd == "refine":
        from otl_map.refine_grid import refine_grid
        refine_grid(a.model, a.earth, a.rundir, a.workers)
        return 0
    if a.cmd == "masks":
        import os
        from otl_map import coastline
        from otl_map.paths import cache_dir, tidemodel_dir
        g = coastline.grid_of(os.path.join(tidemodel_dir(a.model), "amp.m2"))
        w = coastline.water_fraction(g)
        d = cache_dir() / "masks"
        d.mkdir(parents=True, exist_ok=True)
        coastline.write_mask(str(d / f"mask_{a.model}.nc"), g, w)
        print(f"water fractions -> {d / f'mask_{a.model}.nc'}")
        if not a.no_subcells:
            files, n = coastline.write_subcells(str(cache_dir() / f"subcells_{a.model}"), g, w, cf=a.cf)
            print(f"{n} coastal cells, {a.cf}x{a.cf} sub-cells -> {cache_dir() / f'subcells_{a.model}'}")
        return 0
    print(f"otl-map {a.cmd}: not implemented yet in {__version__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
