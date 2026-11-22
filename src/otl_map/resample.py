"""Resample a tide model onto a finer regular grid (e.g. the 1/16 deg FES2014b lattice).

The FFT method assumes cells that are small compared to the 0.15 deg range of the pole-plus-polynomial fit and
a near-field patch of about +-0.25 deg (paper, Sects. 2-3), so a coarse model such as Schwiderski (1 deg) must
first be put on a fine grid. The in-phase and quadrature components are interpolated bilinearly where all four
surrounding source nodes are water; elsewhere, within `fill_cells` source cells of water, the value of the
nearest water node is taken, so that the coarse model also covers the water near the coast that its own
coarse land mask leaves out. Nodes farther from water stay land (999.9). The usual water-fraction scaling and
one-cell extension (`otl-map prepare`) follow on the fine grid.

Grid files use the GMT classic layout of the tide models (x_range, y_range, z_range, spacing, dimension and
z, pixel registered, rows from north to south, amplitude in m, Greenwich phase lag in degrees, land >= 999).
"""
from __future__ import annotations

import os

import netCDF4 as nc
import numpy as np
from scipy.ndimage import distance_transform_edt

from otl_map.paths import derived_dir, tidemodel_dir

LAND = 999.9


def read_grid(path):
    """GMT classic grid: dict with z (ny, nx), cell-centre coordinates x (lon), y (lat, north to south)."""
    d = nc.Dataset(path)
    xr = np.array(d["x_range"][:], float)
    yr = np.array(d["y_range"][:], float)
    sp = np.array(d["spacing"][:], float)
    nx, ny = (int(v) for v in d["dimension"][:])
    z = np.array(d["z"][:], float).reshape(ny, nx)
    d.close()
    return dict(z=z, x=xr[0] + sp[0] * (np.arange(nx) + 0.5), y=yr[1] - sp[1] * (np.arange(ny) + 0.5),
                xr=xr, yr=yr, sp=sp, nx=nx, ny=ny)


def write_grid(path, z, xr, yr, sp, long_name, title):
    ny, nx = z.shape
    d = nc.Dataset(path, "w", format="NETCDF3_CLASSIC")
    d.createDimension("side", 2)
    d.createDimension("xysize", nx * ny)
    v = d.createVariable("dimension", "i4", ("side",))
    v[:] = [nx, ny]
    for name, val, unit in (("x_range", xr, "degrees"), ("y_range", yr, "degrees"), ("spacing", sp, None)):
        v = d.createVariable(name, "f8", ("side",))
        v[:] = val
        if unit:
            v.units = unit
    water = z < 999.0
    v = d.createVariable("z_range", "f8", ("side",))
    v[:] = [z[water].min(), z[water].max()] if water.any() else [0.0, 0.0]
    v.units = "meters" if long_name == "amplitude" else "degrees"
    v = d.createVariable("z", "f4", ("xysize",))
    v.long_name = long_name
    v.scale_factor = 1.0
    v.add_offset = 0.0
    v.node_offset = 1
    v[:] = z.astype(np.float32).ravel()
    d.title = title
    d.close()


def target_lattice(spacing):
    """Global pixel-registered lattice with cell centres at lon = i*sp and lat = 90 - j*sp (as FES2014b)."""
    nx = int(round(360.0 / spacing))
    ny = int(round(180.0 / spacing)) + 1
    xr = np.array([-spacing / 2, 360.0 - spacing / 2])
    yr = np.array([-90.0 - spacing / 2, 90.0 + spacing / 2])
    return dict(nx=nx, ny=ny, x=np.arange(nx) * spacing, y=90.0 - np.arange(ny) * spacing, xr=xr, yr=yr,
                sp=np.array([spacing, spacing]))


def resample_uv(src, tgt, u, v, fill_cells=2.0):
    """Interpolate u, v (ny, nx; nan = land) of the source lattice to the target lattice; nan where land."""
    ny, nx = u.shape
    wet = np.isfinite(u)
    sx, sy = src["sp"][0], src["sp"][1]
    x0, y0 = src["x"][0], src["y"][0]                         # first centre (west), first row (north)
    global_x = abs(nx * sx - 360.0) < 1e-6
    # fractional source indices of every target node
    fi = (tgt["x"][None, :] - x0) / sx
    if global_x:
        fi = np.mod(fi, nx)
    fj = (y0 - tgt["y"][:, None]) / sy
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    tx = fi - i0
    ty = fj - j0
    J0 = np.broadcast_to(j0, (len(tgt["y"]), len(tgt["x"])))
    I0 = np.broadcast_to(i0, J0.shape)
    TX = np.broadcast_to(tx, J0.shape)
    TY = np.broadcast_to(ty, J0.shape)
    inside = (J0 >= 0) & (J0 + 1 < ny) & (global_x | ((I0 >= 0) & (I0 + 1 < nx)))
    jc = np.clip(J0, 0, ny - 2)
    ic = np.clip(I0, 0, nx - 1)
    ic1 = (ic + 1) % nx if global_x else np.clip(ic + 1, 0, nx - 1)
    out = []
    # nearest water node of every source node, for the fill near the coast
    dist, (nj, ni) = distance_transform_edt(~wet, return_indices=True)
    for f in (u, v):
        f0 = np.where(wet, f, 0.0)
        c = [f0[jc, ic], f0[jc, ic1], f0[jc + 1, ic], f0[jc + 1, ic1]]
        w = [wet[jc, ic], wet[jc, ic1], wet[jc + 1, ic], wet[jc + 1, ic1]]
        all_wet = inside & w[0] & w[1] & w[2] & w[3]
        bil = (1 - TX) * (1 - TY) * c[0] + TX * (1 - TY) * c[1] + (1 - TX) * TY * c[2] + TX * TY * c[3]
        # nearest source node of the target node, then its nearest water node
        jn = np.clip(np.rint(fj).astype(int), 0, ny - 1)
        inn = np.rint(fi).astype(int)
        inn = np.mod(inn, nx) if global_x else np.clip(inn, 0, nx - 1)
        JN = np.broadcast_to(jn, J0.shape)
        IN = np.broadcast_to(inn, J0.shape)
        FJ = np.broadcast_to(fj, J0.shape)
        within = (FJ >= -0.5) & (FJ <= ny - 0.5)               # not beyond the latitude range of the source
        if not global_x:
            FI = np.broadcast_to(fi, J0.shape)
            within &= (FI >= -0.5) & (FI <= nx - 0.5)
        near_ok = within & (dist[JN, IN] <= fill_cells)
        near_val = f0[nj[JN, IN], ni[JN, IN]]
        res = np.where(all_wet, bil, np.where(near_ok & ~all_wet, near_val, np.nan))
        out.append(res)
    return out[0], out[1]


def resample_tidemodel(name, out_name, spacing=0.0625, harmonics=None, fill_cells=2.0):
    """Resample tide model `name` (all harmonics found, or `harmonics`) to a global grid; writes `out_name`."""
    sdir = tidemodel_dir(name)
    if harmonics is None:
        harmonics = sorted(f.split(".", 1)[1] for f in os.listdir(sdir) if f.startswith("amp."))
    tgt = target_lattice(spacing)
    odir = derived_dir(out_name)
    for h in harmonics:
        A = read_grid(os.path.join(sdir, f"amp.{h}"))
        P = read_grid(os.path.join(sdir, f"pha.{h}"))
        land = ~(A["z"] < 999.0)
        u = np.where(land, np.nan, A["z"] * np.cos(np.radians(P["z"])))
        v = np.where(land, np.nan, A["z"] * np.sin(np.radians(P["z"])))
        U, V = resample_uv(A, tgt, u, v, fill_cells)
        water = np.isfinite(U)
        amp = np.where(water, np.hypot(U, V), LAND)
        pha = np.where(water, np.degrees(np.arctan2(V, U)), LAND)
        for z, kind, f in ((amp, "amplitude", "amp"), (pha, "phase lag", "pha")):
            write_grid(os.path.join(odir, f"{f}.{h}"), z, tgt["xr"], tgt["yr"], tgt["sp"], kind,
                       f"{kind} of {name} resampled to {spacing:g} deg, harmonic {h}")
    return odir
