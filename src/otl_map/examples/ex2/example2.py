"""Example 2 of the otl-map manual: the complete chain.

The same station as Example 1, now with the coastline: water fractions and coastal sub-cells (Step 1), the
scaled and extended tide model (Step 2), the FFT (Step 3) and the near-field correction (Step 4). The coastline
(9.7 MB download, 180 MB unpacked) is fetched once into the cache directory. A few minutes, most of it for the
near-field bundle.

    python example2.py
"""
import sys

from otl_map.cli import main

COMMANDS = [
    "fetch-coastline",
    "resample SCHW -s 0.0625",
    "masks SCHW_16",
    "prepare SCHW_16",
    "compute SCHW_16 PREM --region -6 -3 47.5 49.5 --rundir brest2",
    "bundle SCHW_16 PREM",
    "stations SCHW_16 PREM brest.txt --rundir brest2 -o brest.blq",
]

if __name__ == "__main__":
    for c in COMMANDS:
        print(f"$ otl-map {c}", flush=True)
        if main(c.split()):
            sys.exit(1)
    print(open("brest.blq").read())
