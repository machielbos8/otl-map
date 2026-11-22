"""Command-line interface `otl-map` (skeleton: subcommands are being ported, see docs/ROADMAP.md)."""
from __future__ import annotations

import argparse
import sys

from otl_map import __version__

SUBCOMMANDS = {
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
    for name, help_ in SUBCOMMANDS.items():
        sub.add_parser(name, help=help_)
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help()
        return 0
    print(f"otl-map {a.cmd}: not implemented yet in {__version__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
