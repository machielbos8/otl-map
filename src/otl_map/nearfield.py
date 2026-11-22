"""Near-field loading for the unified coastal refinement (numpy only, runs on the API server).

See docs/code/unified_refinement_plan.md.  This module holds the pieces that do not need Cython:

  integrate_cell(component, c, xp, yp, dx, dy)
      vectorised port of otl_map.cell_integration.integrate_cell (Appendix A of the paper):
      the pole+polynomial Green's function (Eq. 3) integrated analytically over a planar
      rectangle dx x dy, station at (xp, yp) from the cell centre.  Same sign conventions as
      the Cython version (component 2 returns the east integral with the FFT sign), same
      branches for x or y near 0, so the two agree to rounding.
"""
from __future__ import annotations

import numpy as np

_EPS = 1.0e-4


def _ind_ver(c, x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x2, y2 = x * x, y * y
    x3, y3 = x2 * x, y2 * y
    r = np.sqrt(x2 + y2)
    ax, ay = np.abs(x) < _EPS, np.abs(y) < _EPS
    with np.errstate(all="ignore"):
        lyr = np.log(y + r)
        lxr = np.log(r + x)
        p0_gen = 0.5 * (y * np.log((1 + x / r) / (1 - x / r)) + 2 * x * lyr)
        p0 = np.where(ax, 0.0, np.where(ay, x * lyr, p0_gen))
        p2_gen = (1 / 12) * ((4 * x * y3 + x3 * x * np.sqrt(y2 / x2 + 1) * np.arcsinh(y / x) + 4 * x3 * y) / r
                             - 2 * x3 * np.arctanh(y / r) + 2 * y3 * lxr + 3 * x3 * lyr - 2 * y3 / 3)
        p2 = np.where(ax & ay, 0.0,
                      np.where(ax, (1 / 12) * (2 * y3 * lxr - 2 * y3 / 3),
                               np.where(ay, (1 / 12) * (3 * x3 * lyr), p2_gen)))
    p1 = x * y
    p3 = x3 * y / 3.0 + x * y3 / 3.0
    return c[0] * p0 + c[1] * p1 + c[2] * p2 + c[3] * p3


def _ind_hor(c, x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x2, y2 = x * x, y * y
    x3, y3 = x2 * x, y2 * y
    r = np.sqrt(x2 + y2)
    r2 = r * r
    ax, ay = np.abs(x) < _EPS, np.abs(y) < _EPS
    with np.errstate(all="ignore"):
        lr2 = np.log(r2)
        p0 = np.where(ax, -0.5 * y * (lr2 - 2.0), -0.5 * (y * (lr2 - 2.0) + 2 * x * np.arctan(y / x)))
        p1 = np.where(ax, -0.5 * y * r, -0.5 * (y * r + x2 * np.log(y + r)))
        p3 = np.where(ax, 0.0, -r / 24 * (5 * x2 * y + 3 * x3 / np.sqrt(y2 / x2 + 1) * np.arcsinh(y / x) + 2 * y3))
    p2 = -0.5 * x2 * y
    out = c[0] * p0 + c[1] * p1 + c[2] * p2 + c[3] * p3
    return np.where(ax & ay, 0.0, out)


def _rect(ind, c, xp, yp, dx, dy):
    f1 = ind(c, dx / 2 - xp, dy / 2 - yp) - ind(c, -dx / 2 - xp, dy / 2 - yp)
    f2 = ind(c, dx / 2 - xp, -dy / 2 - yp) - ind(c, -dx / 2 - xp, -dy / 2 - yp)
    return f1 - f2


def integrate_cell(component, c, xp, yp, dx, dy):
    """Vectorised otl_map.cell_integration.integrate_cell (arrays broadcast)."""
    c = np.asarray(c, float)
    xp, yp, dx, dy = (np.asarray(a, float) for a in (xp, yp, dx, dy))
    if component == 0:
        return _rect(_ind_ver, c, -xp, yp, dx, dy)
    if component == 1:
        return _rect(_ind_hor, c, yp, xp, dy, dx)
    return _rect(_ind_hor, c, -xp, yp, dx, dy)


# =====================================================================================
# Unified coastal refinement:  u(P) = I[u_fft](P) + NF_fine(P) - I[NF_coarse](P)
# =====================================================================================
import json as _json
import math as _math
import os as _os

R = 6.371e6
RAD = _math.pi / 180.0
ANALYTIC_M = 1000.0          # sub-cells closer than this to P are integrated analytically
WET_SUB = 10                 # default point-load sub-cells per side for a whole wet cell beyond the 0.15 deg fit
                             # range; a bundle may set its own (meta.json "wet_sub", see bundle.wet_sub_for)
_GF_FILE = {"PREM": "PREM.dat", "PREM_disp": "PREM_disp.dat", "STW105": "STW105.dat",
            "STW105_disp": "STW105_disp.dat"}


class Bundle:
    """refine_bundle_v2/<model> (tools/build_refine_bundle_v2.py) for one Earth model."""

    def __init__(self, path, earth_model, otl_map=None):
        from otl_map.gf_spline import GF as _GF
        from otl_map.paths import greenfnc_dir
        gdir = _os.path.join(otl_map, "greenfnc") if otl_map else greenfnc_dir()
        m = _json.load(open(_os.path.join(path, "meta.json")))
        self.meta = m
        self.nx, self.ny, self.sp = m["nx"], m["ny"], m["sp"]
        self.x0, self.y0 = m["x0"], m["y0"]                      # first cell centre (lon), north row (lat)
        self.N, self.M, self.cf = m["ncell"], m["M"], m["cf"]
        self.wet_sub = int(m.get("wet_sub", WET_SUB))
        self.harmonics = m["harmonics"]
        self.rho = m.get("rho", 1030.0)
        ld = lambda n: np.load(_os.path.join(path, n + ".npy"), mmap_mode="r")  # noqa: E731
        self.idx, self.w0, self.tide, self.sptr, self.sub = (ld(n) for n in ("idx", "w0", "tide", "sptr", "sub"))
        self.T = np.load(_os.path.join(path, f"coarse_{earth_model}.npy"))
        gfn = _GF_FILE[earth_model]
        C = _json.load(open(_os.path.join(gdir, "greensfunction_approx.json")))[gfn]
        self.cc = [np.asarray(C[str(c)], float) for c in range(3)]
        gpath = _os.path.join(gdir, gfn)
        self.gv, self.gh = _GF(gpath, 0), _GF(gpath, 1)
        a = np.arange(self.cf)
        self._sub_dy = (a + 0.5) / self.cf                      # sub-cell row centres, fraction of sp from N edge
        self._sub_dx = (a + 0.5) / self.cf

    # ---------------------------------------------------------------- indexing
    def cell_lon(self, i):
        return self.x0 + i * self.sp

    def cell_lat(self, j):
        return self.y0 - j * self.sp

    def lookup(self, k):
        k = np.asarray(k, np.int64)
        p = np.searchsorted(self.idx, k)
        p = np.minimum(p, len(self.idx) - 1)
        return np.where(np.asarray(self.idx[p]) == k, p, -1)

    def bracket(self, lon, lat):
        """Four nodes (i, j) and bilinear weights, exactly as the station interpolation (wrap in longitude)."""
        lon = lon % 360.0
        fi = (lon - self.x0) / self.sp
        i0 = int(_math.floor(fi + 1e-7)) % self.nx
        x = fi - _math.floor(fi + 1e-7)
        latmin = self.y0 - (self.ny - 1) * self.sp
        fj = (lat - latmin) / self.sp
        js = min(int(_math.floor(fj + 1e-7)), self.ny - 2)
        y = fj - js
        jS, jN = self.ny - 1 - js, self.ny - 2 - js                # north-indexed rows of the S and N nodes
        i1 = (i0 + 1) % self.nx
        return [(i0, jS, (1 - x) * (1 - y)), (i1, jS, x * (1 - y)), (i0, jN, (1 - x) * y), (i1, jN, x * y)]

    # ---------------------------------------------------------------- the correction
    def triggered(self, lon, lat):
        """True if the +-N patch around the station's cell holds a coastal cell (cheap lookup)."""
        lon = lon % 360.0
        N = self.N
        ci = int(round((lon - self.x0) / self.sp)) % self.nx
        cj = int(round((self.y0 - lat) / self.sp))
        di, dj = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
        II, JJ = ((ci + di) % self.nx).ravel(), (cj + dj).ravel()
        ok = (JJ >= 0) & (JJ < self.ny)
        pos = self.lookup(JJ[ok].astype(np.int64) * self.nx + II[ok])
        pos = pos[pos >= 0]
        return bool(len(pos)) and bool((np.asarray(self.sptr[pos]) >= 0).any())

    def correction(self, lon, lat, detail=False):
        """NF_fine(P) - I[NF_coarse](P), complex (3, nharm) in mm, FFT/grid convention.

        None when the +-N patch around the station's cell holds no coastal cell (no refinement needed).
        """
        lon = lon % 360.0
        nx, ny, sp, N, M = self.nx, self.ny, self.sp, self.N, self.M
        ci = int(round((lon - self.x0) / sp)) % nx
        cj = int(round((self.y0 - lat) / sp))
        di, dj = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
        II, JJ = ((ci + di) % nx).ravel(), (cj + dj).ravel()
        ok = (JJ >= 0) & (JJ < ny)
        II, JJ = II[ok], JJ[ok]
        pos = self.lookup(JJ.astype(np.int64) * nx + II)
        if not (pos >= 0).any():
            return None
        found = pos >= 0
        if not (np.asarray(self.sptr[pos[found]]) >= 0).any():
            return None
        if not found.all():
            raise RuntimeError(f"patch of ({lon:.5f},{lat:.5f}) incomplete in the bundle")
        tide = np.asarray(self.tide[pos])                          # (ncell, nharm)
        wet = np.abs(tide).max(axis=1) > 0
        II, JJ, pos, tide = II[wet], JJ[wet], pos[wet], tide[wet]
        if len(II) == 0:                                           # coastal cells without tide (lakes)
            return None
        # ---- coarse: what the FFT put in at the four nodes, interpolated to P
        Kc = np.zeros((3, len(II)))
        for (ni, nj, w) in self.bracket(lon, lat):
            t = JJ - nj
            d = (II - ni + nx // 2) % nx - nx // 2
            if np.abs(t).max() > M or np.abs(d).max() > M:
                raise RuntimeError("node offset beyond the coarse table")
            Kc += w * self.T[nj][:, t + M, d + M]
        # ---- fine: the same cells at P
        Kf = self._fine(lon, lat, II, JJ, pos)
        corr = ((Kf - Kc) @ tide) * 1e3                          # (3, nharm), mm
        if detail:
            return corr, Kf, Kc, II, JJ, tide
        return corr

    def _fine(self, lon, lat, II, JJ, pos):
        """Kernel of every patch cell at P (rho * integral of G over the cell's water), (3, ncell).

        Whole wet cells inside the 0.15 deg fit range: analytic over the cell (as the FFT near field);
        other wet cells: wet_sub^2 point-load sub-cells; coastal cells: cf^2 quadtree sub-cells with
        weights f / mean(f) (the cell's FFT mass), analytic within ANALYTIC_M of P, point loads beyond.
        Mirrored-x convention of the FFT kernel (xp = R cos(lat_P) (lon_cell - lon_P)).
        """
        sp, cf, rho = self.sp, self.cf, self.rho
        csP, snP = _math.cos(lat * RAD), _math.sin(lat * RAD)
        nc_ = len(II)
        K = np.zeros((3, nc_))
        lonc, latc = self.cell_lon(II), self.cell_lat(JJ)
        dlon_c = (lonc - lon + 180.0) % 360.0 - 180.0
        cth = snP * np.sin(latc * RAD) + csP * np.cos(latc * RAD) * np.cos(dlon_c * RAD)
        thc = np.degrees(np.arccos(np.clip(cth, -1, 1)))
        sptr = np.asarray(self.sptr[pos])
        coast = sptr >= 0
        ana = ~coast & (thc + sp < 0.15)
        wet = ~coast & ~ana
        if ana.any():
            xa, ya = R * csP * dlon_c[ana] * RAD, R * (lat - latc[ana]) * RAD
            for comp in range(3):
                K[comp, ana] = rho * integrate_cell(comp, self.cc[comp], xa, ya, R * csP * sp * RAD, R * sp * RAD)
        parts = []                                                  # (cell id, lat, dlon, dsp, ws)
        for sel, n in ((wet, self.wet_sub), (coast, cf)):
            ids = np.flatnonzero(sel)
            if not len(ids):
                continue
            a = (np.arange(n) + 0.5) / n
            oy = (sp / 2 - a * sp)[:, None] * np.ones(n)[None, :]   # rows N -> S
            ox = np.ones(n)[:, None] * (-sp / 2 + a * sp)[None, :]  # columns W -> E
            SLAT = latc[ids][:, None] + oy.ravel()[None, :]
            SLON = dlon_c[ids][:, None] + ox.ravel()[None, :]
            if n == cf:
                f = np.asarray(self.sub[sptr[ids]], float)
                m = f.mean(axis=1, keepdims=True)
                ws = np.where(m > 0, f / np.where(m > 0, m, 1.0), 1.0)
            else:
                ws = np.ones_like(SLAT)
            CID = np.repeat(ids, n * n).reshape(len(ids), n * n)
            k = ws > 0
            parts.append((CID[k], SLAT[k], SLON[k], np.full(int(k.sum()), sp / n), ws[k]))
        if not parts:
            return K
        cid, SLAT, SLON, dsp, ws = (np.concatenate([p[i] for p in parts]) for i in range(5))
        x = R * csP * SLON * RAD
        y = R * (lat - SLAT) * RAD
        near = np.hypot(x, y) < ANALYTIC_M
        far = ~near
        if far.any():
            top, bot = (SLAT[far] + dsp[far] / 2) * RAD, (SLAT[far] - dsp[far] / 2) * RAD
            area = R * R * dsp[far] * RAD * (np.sin(top) - np.sin(bot)) * ws[far]
            cs, sn = np.cos(SLAT[far] * RAD), np.sin(SLAT[far] * RAD)
            csp_ = np.clip(snP * sn + csP * cs * np.cos(SLON[far] * RAD), -1, 1)
            th = np.arccos(csp_)
            gv, gh = self.gv.interp(th), self.gh.interp(th)
            snp = np.sqrt(np.maximum(1 - csp_ ** 2, 0.0))
            safe = np.where(snp > 0, snp, 1.0)
            al = np.arctan2(cs * np.sin(SLON[far] * RAD) / safe, (sn - csp_ * snP) / (safe * csP))
            cf_ = cid[far]
            K[0] += rho * np.bincount(cf_, area * gv, nc_)
            K[1] += rho * np.bincount(cf_, area * gh * np.cos(al - _math.pi), nc_)
            K[2] += rho * np.bincount(cf_, area * gh * np.sin(al - _math.pi), nc_)
        if near.any():
            dx, dy = R * csP * dsp[near] * RAD, R * dsp[near] * RAD
            for comp in range(3):
                v = integrate_cell(comp, self.cc[comp], x[near], y[near], dx, dy)
                K[comp] += rho * np.bincount(cid[near], ws[near] * v, nc_)
        return K
