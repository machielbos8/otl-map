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
    b.add_argument("--wet-sub", type=int, help="point-load sub-cells per side for whole wet cells beyond the "
                   "0.15 deg fit range (default 10 up to 1/16 deg, 40 for coarser grids)")
    rf = sps["refine"]
    rf.add_argument("model")
    rf.add_argument("earth")
    rf.add_argument("--rundir", required=True, help="directory with the FFT grids of `otl-map compute`")
    rf.add_argument("--workers", type=int)
    c = sps["compute"]
    c.add_argument("model", help="tide model (its prepared <model>_mod_ext is used)")
    c.add_argument("earth", nargs="?", default="PREM", help="PREM, PREM_disp, STW105, STW105_disp")
    c.add_argument("--rundir", default=".", help="output directory (default .)")
    c.add_argument("--components", nargs="+", type=int, default=[0, 1, 2], help="0 up, 1 north, 2 east")
    c.add_argument("--harmonics", nargs="+")
    c.add_argument("--region", nargs=4, type=float, metavar=("LON0", "LON1", "LAT0", "LAT1"),
                   help="output nodes of the tide-model lattice inside this box (default: global)")
    c.add_argument("--frame", choices=["CE", "CM"], default="CE")
    c.add_argument("--raw", action="store_true", help="use the tide model as it is (not <model>_mod_ext)")
    c.add_argument("--workers", type=int)
    st = sps["stations"]
    st.add_argument("model")
    st.add_argument("earth", nargs="?", default="PREM")
    st.add_argument("stationfile", help="lines: name lon lat [height]")
    st.add_argument("--rundir", default=".", help="directory with the FFT grids (default .)")
    st.add_argument("-o", "--out", required=True, help="output file (.blq or .json)")
    st.add_argument("--cmc", action="store_true", help="add the centre-of-mass correction (BLQ output)")
    st.add_argument("--no-nearfield", action="store_true", help="interpolated FFT maps only")
    st.add_argument("--harmonics", nargs="+")
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
    if a.cmd == "compute":
        from otl_map.compute import compute, global_mesh
        from otl_map.paths import tidemodel_dir
        mesh = None
        if a.region:
            g = global_mesh(tidemodel_dir(a.model if a.raw else f"{a.model}_mod_ext"))
            sp = g[4]
            snap = lambda v, lo: lo + round((v - lo) / sp) * sp  # noqa: E731
            mesh = [snap(a.region[0], g[0]), snap(a.region[1], g[0]), snap(a.region[2], g[2]),
                    snap(a.region[3], g[2]), sp, g[5]]
        files = compute(a.model, a.earth, a.rundir, a.components, a.harmonics, mesh, prepared=not a.raw,
                        workers=a.workers, frame=a.frame, verbose=True)
        print(f"{len(files)} grids -> {a.rundir}")
        return 0
    if a.cmd == "stations":
        from otl_map import stations as S
        sta = S.read_stations(a.stationfile)
        res = S.compute_stations(sta, a.model, a.earth, a.rundir, a.harmonics, nearfield=not a.no_nearfield)
        if a.out.lower().endswith(".json"):
            if a.cmc:
                raise SystemExit("--cmc is only supported for BLQ output")
            S.write_json(a.out, sta, res, a.model, a.earth)
        else:
            blq = S.to_blq_convention(res)
            cmc = None
            if a.cmc:
                cmc = S.cmc_coefficients(a.model, list(next(iter(res.values())).keys()))
                S.apply_cmc_blq(blq, sta, cmc)
            S.write_blq(a.out, sta, blq, a.model, a.earth, cmc)
        print(f"{len(sta)} stations -> {a.out}")
        return 0
    if a.cmd == "prepare":
        from otl_map.prepare import prepare
        d = prepare(a.model, a.harmonics, extend=not a.no_extend)
        print(f"{a.model} -> {d}")
        return 0
    if a.cmd == "bundle":
        from otl_map.bundle import build_bundle
        print(f"bundle -> {build_bundle(a.model, a.earths, workers=a.workers, wet_sub=a.wet_sub)}")
        return 0
    if a.cmd == "refine":
        from otl_map.refine_grid import refine_grid
        refine_grid(a.model, a.earth, a.rundir, a.workers)
        return 0
    if a.cmd == "masks":
        import os
        from otl_map import coastline
        from otl_map.paths import mask_file, masks_dir, subcells_dir, tidemodel_dir
        g = coastline.grid_of(os.path.join(tidemodel_dir(a.model), "amp.m2"))
        w = coastline.water_fraction(g)
        masks_dir().mkdir(parents=True, exist_ok=True)
        coastline.write_mask(mask_file(a.model), g, w)
        print(f"water fractions -> {mask_file(a.model)}")
        if not a.no_subcells:
            files, n = coastline.write_subcells(subcells_dir(a.model), g, w, cf=a.cf)
            print(f"{n} coastal cells, {a.cf}x{a.cf} sub-cells -> {subcells_dir(a.model)}")
        return 0
    print(f"otl-map {a.cmd}: not implemented yet in {__version__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
