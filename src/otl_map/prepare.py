"""Prepare a tide model for the FFT (paper, Sect. 3, steps 1-2): extend it by one cell over the coast and scale the
tide of every cell by its water fraction.

Extension: each land cell with at least one ocean cell among its eight neighbours (wrapping in longitude, not
across the poles) receives the mean in-phase and quadrature tide of those ocean neighbours (single pass).
Scaling: every cell is multiplied by its water fraction (otl_map.coastline, `otl-map masks`), so a cell carries
the tidal water mass that the coastline places in it. Output: <model>_mod_ext (or _mod), same layout as the
input; cells without tide are written with amplitude 0 and phase 0, as the modify_tidemodel tool did.
"""
from __future__ import annotations

import os

import netCDF4 as nc
import numpy as np

from otl_map.paths import cache_dir, tidemodel_dir
from otl_map.resample import read_grid, write_grid


def extend_one_cell(u, v, land):
    """Single-pass extension of (ny, nx) u, v over land cells next to ocean cells (mean of the ocean neighbours)."""
    wet = (~land).astype(float)
    su = np.zeros_like(u)
    sv = np.zeros_like(v)
    sw = np.zeros_like(u)
    uu, vv = np.where(land, 0.0, u), np.where(land, 0.0, v)
    ny = u.shape[0]
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            a = np.roll(uu * 1.0, -di, axis=1)          # neighbour at column i + di (wrapping)
            b = np.roll(vv * 1.0, -di, axis=1)
            c = np.roll(wet, -di, axis=1)
            # neighbour at row j + dj, no wrap across the poles
            ta, tb, tc = np.zeros_like(a), np.zeros_like(b), np.zeros_like(c)
            if dj == 0:
                ta, tb, tc = a, b, c
            elif dj == 1:
                ta[:ny - 1], tb[:ny - 1], tc[:ny - 1] = a[1:], b[1:], c[1:]
            else:
                ta[1:], tb[1:], tc[1:] = a[:ny - 1], b[:ny - 1], c[:ny - 1]
            su += ta
            sv += tb
            sw += tc
    fill = land & (sw > 0)
    u2, v2 = u.copy(), v.copy()
    u2[fill] = su[fill] / sw[fill]
    v2[fill] = sv[fill] / sw[fill]
    return u2, v2, land & ~fill


def read_mask(model):
    f = os.environ.get("OTL_MAP_CPP")
    f = os.path.join(f, f"mask_{model}.nc") if f else str(cache_dir() / "masks" / f"mask_{model}.nc")
    if not os.path.isfile(f):
        raise FileNotFoundError(f"{f} not found: run `otl-map masks {model}` first")
    return read_grid(f)["z"]


def prepare(model, harmonics=None, extend=True, out=None):
    """Write <model>_mod_ext (or _mod without extension); returns the output directory."""
    src = tidemodel_dir(model)
    if harmonics is None:
        harmonics = sorted(f.split(".", 1)[1] for f in os.listdir(src) if f.startswith("amp."))
    w = read_mask(model)
    if out is None:
        out = f"{model}_mod_ext" if extend else f"{model}_mod"
    odir = tidemodel_dir(out, create=True)
    for h in harmonics:
        A = read_grid(os.path.join(src, f"amp.{h}"))
        P = read_grid(os.path.join(src, f"pha.{h}"))
        if A["z"].shape != w.shape:
            raise ValueError(f"mask of {model} does not match amp.{h}")
        land = ~(A["z"] < 999.0)
        u = np.where(land, 0.0, A["z"] * np.cos(np.radians(P["z"])))
        v = np.where(land, 0.0, A["z"] * np.sin(np.radians(P["z"])))
        if extend:
            u, v, land = extend_one_cell(u, v, land)
        u, v = np.where(land, 0.0, w * u), np.where(land, 0.0, w * v)
        amp = np.where(land, 0.0, np.hypot(u, v))
        pha = np.where(land, 0.0, np.degrees(np.arctan2(v, u)))
        for z, kind, f in ((amp, "amplitude", "amp"), (pha, "phase-lag", "pha")):
            write_grid(os.path.join(odir, f"{f}.{h}"), z, A["xr"], A["yr"], A["sp"], kind,
                       f"{kind} of {model}, {'extended and ' if extend else ''}scaled by water fraction, {h}")
    return odir
