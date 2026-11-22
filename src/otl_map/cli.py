"""Command-line interface `otl-map` (skeleton: subcommands are being ported, see docs/ROADMAP.md)."""
from __future__ import annotations

import argparse
import sys

from otl_map import __version__

SUBCOMMANDS = {
    "resample": "put a (coarse) tide model on a fine global grid, e.g. Schwiderski on 1/16 deg",
    "fetch-coastline": "download and verify the full-resolution coastline (Zenodo)",
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
    print(f"otl-map {a.cmd}: not implemented yet in {__version__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
