"""Water fractions of tide-model cells and of their sub-cells from the full-resolution quadtree coastline.

Replaces the C++ tools create_mask and create_subcells of the development repository with the same
definitions (QuadtreeMask::exact): the land area of a cell is the area, in longitude-latitude radians^2, of
the land leaves of the quadtree that overlap it, clipped to |lat| <= 90 deg, and the water fraction is
w = 1 - land / (dx dy). Coastal cells (1e-6 < w < 1 - 1e-6) are divided into cf x cf sub-cells with their own
water fraction, written in the binary format read by otl_map.subcells_io.

The coastline file (quadtree_0.00137_v2.dat, GSHHG-derived, finest cell 0.00137 deg) is found in
$OTL_MAP_COASTLINE, or in the cache directory (`otl-map fetch-coastline`).
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import netCDF4 as nc
import numpy as np

from otl_map._quadtree import land_area, read_land_leaves, subcell_land
from otl_map.paths import coastline_file

RAD = math.pi / 180.0


@lru_cache(maxsize=2)
def land_leaves(path: str | None = None):
    """(n_levels, xc, yc, level) of the land leaves of the quadtree (read once per process)."""
    return read_land_leaves(path or coastline_file())


def grid_of(amp_file):
    """Grid of a tide-model file (GMT classic layout): degrees, pixel registered; a duplicate 360-deg column is
    dropped as in otl_map.resample.read_grid."""
    from otl_map.resample import read_grid
    g = read_grid(amp_file)
    if abs(g["nx"] * g["sp"][0] - 360.0) > 1e-6:
        raise ValueError(f"{amp_file}: {g['nx']} columns x {g['sp'][0]} deg != 360 deg (a global grid is required)")
    return dict(xr=g["xr"], yr=g["yr"], sp=g["sp"], nx=g["nx"], ny=g["ny"])


def water_fraction(grid, path=None, row0=0, row1=-1):
    """Water fraction (ny, nx), rows north to south, of the cells of `grid` (dict from grid_of)."""
    n, xc, yc, lv = land_leaves(path)
    dx, dy = grid["sp"][0] * RAD, grid["sp"][1] * RAD
    a = land_area(n, xc, yc, lv, grid["xr"][0] * RAD, dx, grid["nx"], grid["yr"][1] * RAD, dy, grid["ny"],
                  row0, row1)
    return 1.0 - a / (dx * dy)


def write_mask(path, grid, w):
    """Water fractions in the GMT classic layout of the tide models (as mask_<model>.nc of create_mask)."""
    ny, nx = w.shape
    d = nc.Dataset(path, "w", format="NETCDF3_CLASSIC")
    d.createDimension("side", 2)
    d.createDimension("xysize", nx * ny)
    v = d.createVariable("dimension", "i4", ("side",))
    v[:] = [nx, ny]
    for name in ("x_range", "y_range", "spacing"):
        v = d.createVariable(name, "f8", ("side",))
        v[:] = grid[{"x_range": "xr", "y_range": "yr", "spacing": "sp"}[name]]
    v = d.createVariable("z_range", "f8", ("side",))
    v[:] = [float(w.min()), float(w.max())]
    v = d.createVariable("z", "f8", ("xysize",))
    v.long_name = "water fraction"
    v.node_offset = 1
    v[:] = w.ravel()
    d.close()


def default_cf(sp_deg):
    """Sub-cells per side of a coastal cell for sub-cells of about 100 m: 60 at 1/16 deg (FES2014b), 32 at 1/30 deg
    (FES2022b, TPXO10-atlas), 120 at 1/8 deg (GOT5.5); at least 8. Used by `otl-map masks` and
    tools/produce_model.py, so that both make the sub-cells of the online service."""
    return max(8, int(round(sp_deg / 0.0625 * 60)))


def make_masks(model, cf=None, subcells=True):
    """Water fractions (and coastal sub-cells) of tide model `model`, written to masks_dir(); returns
    (mask file, sub-cell directory or None, number of coastal cells, cf)."""
    from otl_map.paths import mask_file, masks_dir, subcells_dir, tidemodel_dir
    g = grid_of(os.path.join(tidemodel_dir(model), "amp.m2"))
    w = water_fraction(g)
    masks_dir().mkdir(parents=True, exist_ok=True)
    write_mask(mask_file(model), g, w)
    if not subcells:
        return mask_file(model), None, int(((w > 1e-6) & (w < 1 - 1e-6)).sum()), None
    cf = cf or default_cf(float(g["sp"][0]))
    _, n = write_subcells(subcells_dir(model), g, w, cf=cf)
    return mask_file(model), subcells_dir(model), n, cf


def write_subcells(out_dir, grid, w, cf=60, rows_per_part=120, path=None, tol=1e-6):
    """Sub-cell water fractions of the coastal cells, in parts of `rows_per_part` rows (create_subcells format):
    header int32 nx, ny, cf, row0, row1; float64 xr0, yr1, spx, spy (deg); records int64 k, float64 w,
    uint8[cf*cf] sub-cell water fraction * 255 (rounded), rows N->S, columns W->E."""
    n, xc, yc, lv = land_leaves(path)
    nx, ny = grid["nx"], grid["ny"]
    dx, dy = grid["sp"][0] * RAD, grid["sp"][1] * RAD
    os.makedirs(out_dir, exist_ok=True)
    rec = np.dtype([("k", "<i8"), ("w", "<f8"), ("f", np.uint8, (cf * cf,))])
    files, ncoast = [], 0
    for part, row0 in enumerate(range(0, ny, rows_per_part)):
        row1 = min(row0 + rows_per_part, ny)
        wr = w[row0:row1].ravel()
        loc = np.flatnonzero((wr > tol) & (wr < 1.0 - tol))
        cells = (loc + row0 * nx).astype(np.int64)
        land = subcell_land(n, xc, yc, lv, grid["xr"][0] * RAD, dx, nx, grid["yr"][1] * RAD, dy, ny, cf,
                            cells, row0, row1)
        ww = np.clip(1.0 - land / ((dx / cf) * (dy / cf)), 0.0, 1.0)
        # sub-cells entirely beyond a pole count as land (QuadtreeMask::exact returns 1 = land there)
        jj = np.arange(cf)
        top = grid["yr"][1] - ((cells // nx)[:, None] * cf + jj[None, :]) * grid["sp"][1] / cf     # deg
        eps = 1e-9                                         # an edge exactly on a pole counts as beyond it
        beyond = (top - grid["sp"][1] / cf >= 90.0 - eps) | (top <= -90.0 + eps)                         # (ncell, cf)
        if beyond.any():
            ww = ww.reshape(len(cells), cf, cf)
            ww[np.broadcast_to(beyond[:, :, None], ww.shape)] = 0.0
            ww = ww.reshape(len(cells), cf * cf)
        body = np.empty(len(cells), rec)
        body["k"] = cells
        body["w"] = wr[loc]
        body["f"] = np.floor(255.0 * ww + 0.5).astype(np.uint8)        # lround for non-negative values
        f = os.path.join(out_dir, f"part_{part:02d}.bin")
        with open(f, "wb") as fo:
            fo.write(np.array([nx, ny, cf, row0, row1], "<i4").tobytes())
            fo.write(np.array([grid["xr"][0], grid["yr"][1], grid["sp"][0], grid["sp"][1]], "<f8").tobytes())
            fo.write(body.tobytes())
        files.append(f)
        ncoast += len(cells)
    return files, ncoast
