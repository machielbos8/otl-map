"""The FFT near-field kernel, element by element (offline; needs the Cython extensions).

compute_otl_fft convolves every latitude row of the tide model with a kernel G[k] (one value per
longitude offset k).  For the unified coastal refinement (docs/code/unified_refinement_plan.md) we
need, for a grid node, the contribution of the cells of a small patch around it EXACTLY as the FFT
added it, so that it can be removed again and replaced by a finer near field.  `kernel_row` repeats
the per-(mesh row, tide row) kernel construction of compute_otl_fft._process_row_batch for a few
offsets only, calling the same Cython functions (integrate_cell, numerical_integral, Greensfunction).
The delta test (paper_JoG/paper_scripts/coarse_kernel_delta_test.py) checks it against compute_otl_fft.

Convention (as in the FFT output): the node at tide-grid column m receives from the cell at column
m + di the value  G[(-di) mod nx] * (u + i v)  with u + i v = amp * exp(i * phase) of the (modified)
tide model; G[nx - k] = G[k] (vertical, north) and -G[k] (east).  Result in metres per metre of tide.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile

import numpy as np

from otl_map.cell_integration import integrate_cell, numerical_integral
from otl_map.greensfunction import Greensfunction

R = 6.371e6
RAD = math.pi / 180.0


def load_gf(gf_file, component):
    """Greensfunction instance for greenfnc/<gf_file> and component (via a temporary json)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
        json.dump({"greensfunction": gf_file, "component": int(component)}, fp)
        name = fp.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gf = Greensfunction(name)
    finally:
        os.unlink(name)
    return gf


def kernel_row(gf, cc, lat_m, lat_c, kmax, csp0, csp1, rho=1030.0, near_cells=3.0):
    """G[k], k = 0..kmax, for mesh latitude lat_m and tide-row latitude lat_c (radians).

    Literal copy of the kernel construction in compute_otl_fft._process_row_batch.
    """
    comp = gf.component
    cc = np.asarray(cc, dtype=np.double)
    hpi = math.pi / 2
    csm, snm = math.cos(lat_m), math.sin(lat_m)
    csc, snc = math.cos(lat_c), math.sin(lat_c)
    yy0, yy1 = lat_c - 0.5 * csp1, lat_c + 0.5 * csp1
    if yy0 < -hpi:
        yy0 = -hpi
    elif yy1 > hpi:
        yy1 = hpi
    A = R * R * csp0 * (math.sin(yy1) - math.sin(yy0))
    dlon = np.arange(kmax + 1, dtype=np.double) * csp0
    csp_k = np.clip(snm * snc + csm * csc * np.cos(dlon), -1.0, 1.0)
    theta_k = np.arccos(csp_k)
    G = np.zeros(kmax + 1)
    near = theta_k < near_cells * csp0
    far = ~near
    if far.any():
        gfar = rho * A * np.asarray(gf.interpolate_array(theta_k[far]))
        if comp != 0:
            snp = np.sqrt(np.maximum(1.0 - csp_k[far] ** 2, 0.0))
            safe = np.where(snp > 0.0, snp, 1.0)
            if abs(csm) < 1e-7:
                alpha = (dlon[far] - math.pi) if snm > 0.0 else -dlon[far]
            else:
                sna = csc * (-np.sin(dlon[far])) / safe
                csa = (snc - csp_k[far] * snm) / (safe * csm)
                alpha = np.arctan2(sna, csa)
            gfar = gfar * (np.cos(alpha - math.pi) if comp == 1 else np.sin(alpha - math.pi))
        G[far] = gfar
    for k in np.where(near)[0]:
        # compute_distance(lon_m, ..., lon_c, ...): theta = acos(csp) (0 if csp > 0.999999999)
        c = snm * snc + csm * csc * math.cos(-k * csp0)
        theta = 0.0 if c > 0.999999999 else math.acos(max(-1.0, c))
        xp = R * csm * (-k * csp0)
        yp = R * (lat_m - lat_c)
        dx = R * csm * csp0
        dy = R * csp1
        if theta + csp0 < 0.15 * RAD:
            G[k] = rho * integrate_cell(comp, cc, xp, yp, dx, dy)
        else:
            G[k] = rho * numerical_integral(comp, gf, xp, yp, dx, dy)
    return G


def table_for_row(gfs, ccs, j, cy1, csp0, csp1, M, rho=1030.0, near_cells=3.0, cny=None):
    """Coarse table for a node in tide row j (centre latitude cy1 - (j+0.5) csp1, radians).

    Returns T[comp, t + M, di + M] for tide-row offset t (cell row j + t) and column offset di
    (cell column m + di), |t|, |di| <= M: the FFT's contribution of that cell to the node, per
    unit (u + i v) of the cell, in metres.  Rows outside the grid are left 0.
    """
    lat_m = cy1 - (j + 0.5) * csp1
    T = np.zeros((3, 2 * M + 1, 2 * M + 1))
    for comp in range(3):
        for t in range(-M, M + 1):
            jc = j + t
            if jc < 0 or (cny is not None and jc >= cny):
                continue
            lat_c = cy1 - (jc + 0.5) * csp1
            G = kernel_row(gfs[comp], ccs[comp], lat_m, lat_c, M, csp0, csp1, rho, near_cells)
            for di in range(-M, M + 1):
                v = G[abs(di)]
                if di > 0 and comp == 2:
                    v = -v
                T[comp, t + M, di + M] = v
    return T
