"""Log-spaced natural cubic spline of a tabulated load Green's function (numpy only).

Same interpolation as otl_map.greensfunction (spline of G*theta in log(theta)), with a dense lookup table for
fast vectorised evaluation. Used by the near-field correction.
"""
from __future__ import annotations

import math

import numpy as np

RAD = math.pi / 180.0


class GF:
    """Log-spaced natural cubic spline of the load Green's function (component-selectable).

    Mirrors otl_map.greensfunction: interpolate(theta_rad) with the y[i]*=x[i] scaling
    trick, so no Cython dependency on the server.  component 0 -> G_ver, 1/2 -> G_hor.
    """
    def __init__(self, path, component):
        col = 1 if component == 0 else 2
        t, g = [], []
        with open(path) as fp:
            fp.readline()                              # header
            for line in fp:
                c = line.split()
                t.append(RAD * float(c[0])); g.append(float(c[col]))
        self.x = np.array(t); self.y = np.array(g) * self.x    # scale
        self.lnx = np.log(self.x); self.n = len(self.x)
        self.y2 = self._spline2(self.lnx, self.y)
        # Dense equi-log-spaced lookup table of G, evaluated once from the accurate
        # cubic spline.  Runtime interp is then a vectorised index + linear blend
        # (no per-point searchsorted / spline), which is the dominant refine cost.
        self._lo = float(self.lnx[0]); hi = float(self.lnx[-1]); self._M = 200_000
        self._inv_d = (self._M - 1) / (hi - self._lo)
        self._lut = self._spline(np.exp(np.linspace(self._lo, hi, self._M)))

    @staticmethod
    def _spline2(x, y):
        n = len(x); y2 = np.zeros(n); u = np.zeros(n)
        y2[0] = -0.5; u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]))
        for i in range(1, n - 1):
            sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]); p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            ui = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            u[i] = (6.0 * ui / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
        qn = 0.5; un = (3.0 / (x[n - 1] - x[n - 2])) * (-(y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
        y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)
        for i in range(n - 2, -1, -1):
            y2[i] = y2[i] * y2[i + 1] + u[i]
        return y2

    def _spline(self, theta):
        """Accurate cubic-spline G(theta) — used once to fill the lookup table."""
        th = np.clip(np.asarray(theta), self.x[1], self.x[-1])
        lnth = np.log(th)
        il = np.clip(np.searchsorted(self.lnx, lnth, side="right") - 1, 0, self.n - 2)
        iu = il + 1
        h = self.lnx[iu] - self.lnx[il]
        a = (self.lnx[iu] - lnth) / h; b = (lnth - self.lnx[il]) / h
        yy = a * self.y[il] + b * self.y[iu] + ((a**3 - a) * self.y2[il] + (b**3 - b) * self.y2[iu]) * h * h / 6.0
        return yy / th                                  # undo scaling

    def interp(self, theta):
        """Fast G(theta): index into the dense LUT and linearly blend (no searchsorted)."""
        th = np.clip(np.asarray(theta, dtype=float), self.x[1], self.x[-1])
        pos = (np.log(th) - self._lo) * self._inv_d
        i0 = np.clip(pos.astype(np.intp), 0, self._M - 2)
        frac = pos - i0
        g0 = self._lut[i0]
        return g0 + frac * (self._lut[i0 + 1] - g0)
