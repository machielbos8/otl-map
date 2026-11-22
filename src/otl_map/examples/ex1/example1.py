"""Example 1 of the otl-map manual: a first map without the coastline.

Schwiderski's M2 model (shipped with otl-map) is put on the 1/16 deg grid of FES2014b, the M2 loading
(up, north, east) is computed around Brest with the FFT, and the OTL at the GNSS station BRST is written to
brest.json. No coastline is needed (--raw: no water fractions). About a minute.

    python example1.py
"""
import sys

from otl_map.cli import main

COMMANDS = [
    "resample SCHW -s 0.0625",
    "compute SCHW_16 PREM --raw --harmonics m2 --region -6 -3 47.5 49.5 --rundir brest",
    "stations SCHW_16 PREM brest.txt --rundir brest --harmonics m2 --no-nearfield -o brest.json",
]

if __name__ == "__main__":
    for c in COMMANDS:
        print(f"$ otl-map {c}", flush=True)
        if main(c.split()):
            sys.exit(1)
    print(open("brest.json").read())
