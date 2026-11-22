"""Table 4 of the paper: analytical cell integrals (Appendix A/B) for a 1/16 x 1/16 deg cell at 45 N, PREM, 1-m
tide, rho = 1030 kg/m3, in mm, for eight station positions; and against brute-force quadrature."""
import json
import math
import os

import numpy as np
import pytest

from otl_map.cell_integration import integrate_cell
from otl_map.paths import greenfnc_dir

R, RHO, RAD = 6.371e6, 1030.0, math.pi / 180
SP, LAT = 0.0625, 45.0
DX, DY = R * math.cos(LAT * RAD) * SP * RAD, R * SP * RAD
PHYS = {0: 1.0, 1: 1.0, 2: -1.0}          # integrate_cell returns east with the opposite sign
# position, xp, yp, up, east, north (mm), from the paper (results/paper/cell_integral_test_values.txt)
TABLE4 = [("centre", 0.0, 0.0, -0.819971, 0.0, 0.0),
          ("inside", DX / 4, DY / 4, -0.730811, -0.056521, -0.053374),
          ("E edge", DX / 2, 0.0, -0.561123, -0.127970, 0.0),
          ("NE corner", DX / 2, DY / 2, -0.378955, -0.078284, -0.088863),
          ("1 east", DX, 0.0, -0.236326, -0.078715, 0.0),
          ("1 north", 0.0, DY, -0.165997, 0.0, -0.060898),
          ("2 NE", 2 * DX, 2 * DY, -0.040487, -0.009602, -0.013336),
          ("3 north", 0.0, 3 * DY, -0.030371, 0.0, -0.011903)]


@pytest.fixture(scope="module")
def prem():
    C = json.load(open(os.path.join(greenfnc_dir(), "greensfunction_approx.json")))["PREM.dat"]
    return {k: np.array(C[str(k)], float) for k in range(3)}


@pytest.mark.parametrize("name,xp,yp,up,east,north", TABLE4)
def test_table4(prem, name, xp, yp, up, east, north):
    for comp, ref in ((0, up), (2, east), (1, north)):
        v = 1e3 * RHO * PHYS[comp] * integrate_cell(comp, prem[comp], xp, yp, DX, DY)
        assert v == pytest.approx(ref, abs=1.5e-6), (name, comp)


def quad(c, comp, xp, yp, n=800):
    s = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(s * DX, s * DY)
    dx, dy = xp - X, yp - Y
    r = np.hypot(dx, dy)
    g = c[0] / r + c[1] + c[2] * r + c[3] * r ** 2
    g = g * (dy / r if comp == 1 else dx / r if comp == 2 else 1.0)
    return float(np.sum(g) * (DX / n) * (DY / n))


@pytest.mark.parametrize("name,xp,yp", [(t[0], t[1], t[2]) for t in TABLE4 if t[0] in ("1 east", "1 north", "2 NE")])
def test_against_quadrature(prem, name, xp, yp):
    for comp in (0, 1, 2):
        a = integrate_cell(comp, prem[comp], xp, yp, DX, DY) * PHYS[comp]
        q = quad(prem[comp], comp, xp, yp)
        if abs(q) > 1e-12:
            assert a == pytest.approx(q, rel=2e-3), (name, comp)
