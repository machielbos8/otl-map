"""otl_map.coastline on a small synthetic quadtree, against brute-force sampling of the same tree."""
import math

import numpy as np
import pytest

from otl_map._quadtree import land_area, read_land_leaves, subcell_land

FLTMAX = np.finfo(np.float32).max
LAND, WATER = 999.9, 0.0


@pytest.fixture()
def tree(tmp_path):
    """n_levels = 4 (root level 3, finest leaf 2 pi / 8 = 45 deg). Root children: (0,0) subdivided further,
    (1,0) land, (0,1) water, (1,1) subdivided. Children order (0,0), (1,0), (0,1), (1,1)."""
    vals = [FLTMAX,                          # root child (0,0): subdivided
            LAND, WATER, FLTMAX, WATER,      #   its children; (0,1) subdivided again
            LAND, LAND, WATER, LAND,         #     finest leaves
            LAND,                            # root child (1,0): land
            WATER,                           # root child (0,1): water
            FLTMAX,                          # root child (1,1): subdivided
            WATER, LAND, LAND, WATER]
    f = tmp_path / "tree.dat"
    with open(f, "wb") as fo:
        fo.write(np.array([4], "<i4").tobytes())
        fo.write(np.array(vals, "<f4").tobytes())
    return str(f)


def brute_land(path, lon, lat):
    """Land (True) at points (rad) by walking the tree as QuadtreeMask::land_or_water."""
    n, xc, yc, lv = read_land_leaves(path)
    s = 2 * math.pi / 2 ** (n - 1)
    x0 = xc * s
    x1 = x0 + (2.0 ** lv) * s
    y1 = 1.5 * math.pi - yc * s
    y0 = y1 - (2.0 ** lv) * s
    out = np.zeros(lon.shape, bool)
    for a, b, c, d in zip(x0, x1, y0, y1):
        out |= (lon >= a) & (lon < b) & (lat >= c) & (lat < d)
    return out


def test_read(tree):
    n, xc, yc, lv = read_land_leaves(tree)
    assert n == 4 and len(xc) == 7          # 7 land leaves


def test_land_area_matches_sampling(tree):
    n, xc, yc, lv = read_land_leaves(tree)
    nx, ny = 24, 13                         # 15 deg cells, pixel registered, poles at row centres
    dx = dy = math.radians(15.0)
    xedge, ytop = -dx / 2, math.pi / 2 + dy / 2
    a = land_area(n, xc, yc, lv, xedge, dx, nx, ytop, dy, ny)
    m = 60                                  # sample points per cell side
    for j in (1, 4, 6, 9, 12):
        for i in (0, 3, 7, 11, 18, 23):
            u = (np.arange(m) + 0.5) / m
            lon = (xedge + (i + u) * dx) % (2 * math.pi)
            lat = ytop - (j + u) * dy
            LON, LAT = np.meshgrid(lon, lat)
            inside = np.abs(LAT) <= math.pi / 2
            frac = (brute_land(tree, LON, LAT) & inside).mean()
            assert a[j, i] / (dx * dy) == pytest.approx(frac, abs=1e-9), (j, i)


def test_subcells_sum_to_cell(tree):
    n, xc, yc, lv = read_land_leaves(tree)
    nx, ny, cf = 24, 13, 6
    dx = dy = math.radians(15.0)
    xedge, ytop = -dx / 2, math.pi / 2 + dy / 2
    a = land_area(n, xc, yc, lv, xedge, dx, nx, ytop, dy, ny)
    cells = np.arange(ny * nx, dtype=np.int64)
    sub = subcell_land(n, xc, yc, lv, xedge, dx, nx, ytop, dy, ny, cf, cells, 0, ny)
    assert np.allclose(sub.sum(1).reshape(ny, nx), a, atol=1e-12)
