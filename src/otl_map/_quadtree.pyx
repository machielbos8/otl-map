# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Quadtree coastline (GSHHG-derived, Frisken & Perry layout) and land areas on a regular grid.

File layout (little endian): int32 n_levels, then float32 values in depth-first order: for every subdivided
cell its four children (x, y) = (0,0), (1,0), (0,1), (1,1); FLT_MAX = subdivided (its children follow),
otherwise a leaf with value >= 999 for land. The root covers longitude [0, 2 pi] and latitude [-pi/2, 3 pi/2]
(the upper half is unused); a leaf with location codes (xc, yc) at level L spans
  lon [xc s, (xc + 2^L) s],   lat [3 pi/2 - (yc + 2^L) s, 3 pi/2 - yc s],   s = 2 pi / 2^(n_levels - 1).
Land areas are computed in plain longitude-latitude radians^2 and clipped to |lat| <= pi/2, exactly as
QuadtreeMask::exact of the C++ tools (cpp/create_mask, cpp/create_subcells of the development repository).
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport floor, fmin, fmax, M_PI

cdef float FLTMAX = np.finfo(np.float32).max


def read_land_leaves(path):
    """Return (n_levels, xc, yc, level) of all land leaves (uint32, uint32, uint8 arrays)."""
    raw = np.fromfile(path, dtype="<i4", count=1)
    cdef int n_levels = int(raw[0])
    cdef cnp.float32_t[:] v = np.fromfile(path, dtype="<f4", offset=4)
    cdef Py_ssize_t nv = v.shape[0], p = 0, n = 0
    cdef int root = n_levels - 1
    # explicit stack: per level the parent codes and the index of the next child to read
    cdef cnp.uint32_t[:] sx = np.zeros(n_levels + 1, np.uint32)
    cdef cnp.uint32_t[:] sy = np.zeros(n_levels + 1, np.uint32)
    cdef cnp.int32_t[:] si = np.zeros(n_levels + 1, np.int32)
    xs = np.empty(nv, np.uint32)
    ys = np.empty(nv, np.uint32)
    ls = np.empty(nv, np.uint8)
    cdef cnp.uint32_t[:] ox = xs
    cdef cnp.uint32_t[:] oy = ys
    cdef cnp.uint8_t[:] ol = ls
    cdef int lev = root                     # level of the cell whose children are being read
    cdef unsigned int half, cx, cy
    cdef int i
    cdef float val
    sx[lev] = 0; sy[lev] = 0; si[lev] = 0
    while lev <= root:
        i = si[lev]
        if i == 4:                          # all children done: back to the parent
            lev += 1
            if lev <= root:
                si[lev] += 1
            continue
        if p >= nv:
            raise ValueError("quadtree file ends early")
        val = v[p]
        p += 1
        half = 1u << (lev - 1)
        cx = sx[lev] + (half if (i == 1 or i == 3) else 0)
        cy = sy[lev] + (half if (i == 2 or i == 3) else 0)
        if val == FLTMAX:                   # subdivided: descend
            lev -= 1
            sx[lev] = cx; sy[lev] = cy; si[lev] = 0
        else:
            if val > 999.0:
                ox[n] = cx; oy[n] = cy; ol[n] = lev - 1
                n += 1
            si[lev] += 1
    return n_levels, xs[:n].copy(), ys[:n].copy(), ls[:n].copy()


def land_area(int n_levels, cnp.uint32_t[:] xc, cnp.uint32_t[:] yc, cnp.uint8_t[:] lv,
              double xedge, double dx, int nx, double ytop, double dy, int ny, int row0=0, int row1=-1):
    """Land area (rad^2) of the cells of a global regular grid, rows row0 <= j < row1 (north to south).

    xedge: western edge of column 0 (rad), dx: spacing (rad), nx columns covering 2 pi (wrap);
    ytop: northern edge of row 0 (rad), dy: spacing (rad)."""
    if row1 < 0:
        row1 = ny
    out_np = np.zeros((row1 - row0, nx))
    cdef double[:, :] out = out_np
    cdef double s = 2.0 * M_PI / (1u << (n_levels - 1))
    cdef double y3 = 1.5 * M_PI, hp = 0.5 * M_PI
    cdef Py_ssize_t k, n = xc.shape[0]
    cdef double x0, x1, y0, y1, ax, ay, cx0, cy1
    cdef long i, i0, i1, j, j0, j1, ii
    for k in range(n):
        x0 = xc[k] * s
        x1 = x0 + (1u << lv[k]) * s
        y1 = y3 - yc[k] * s
        y0 = y1 - (1u << lv[k]) * s
        if y1 > hp:
            y1 = hp
        if y0 < -hp:
            y0 = -hp
        if y1 <= y0:
            continue
        j0 = <long>floor((ytop - y1) / dy)
        j1 = <long>floor((ytop - y0) / dy)
        if j1 < row0 or j0 >= row1:
            continue
        i0 = <long>floor((x0 - xedge) / dx)
        i1 = <long>floor((x1 - xedge) / dx)
        for j in range(max(j0, row0), min(j1, row1 - 1) + 1):
            cy1 = ytop - j * dy
            ay = fmin(y1, cy1) - fmax(y0, cy1 - dy)
            if ay <= 0.0:
                continue
            for i in range(i0, i1 + 1):
                cx0 = xedge + i * dx
                ax = fmin(x1, cx0 + dx) - fmax(x0, cx0)
                if ax <= 0.0:
                    continue
                ii = i % nx
                if ii < 0:
                    ii += nx
                out[j - row0, ii] += ax * ay
    return out_np


def subcell_land(int n_levels, cnp.uint32_t[:] xc, cnp.uint32_t[:] yc, cnp.uint8_t[:] lv,
                 double xedge, double dx, int nx, double ytop, double dy, int ny, int cf,
                 cnp.int64_t[:] cells, int row0, int row1):
    """Land area (rad^2) of the cf x cf sub-cells of the given cells (linear index j*nx + i, rows row0..row1-1).

    Returns float64 (len(cells), cf*cf), sub-cells rows north to south, columns west to east."""
    cdef Py_ssize_t nc = cells.shape[0]
    pos_np = np.full((row1 - row0) * nx, -1, np.int64)
    cdef cnp.int64_t[:] pos = pos_np
    cdef Py_ssize_t q
    for q in range(nc):
        pos[cells[q] - <long>row0 * nx] = q
    out_np = np.zeros((nc, cf * cf))
    cdef double[:, :] out = out_np
    cdef double s = 2.0 * M_PI / (1u << (n_levels - 1))
    cdef double y3 = 1.5 * M_PI, hp = 0.5 * M_PI
    cdef double ddx = dx / cf, ddy = dy / cf
    cdef Py_ssize_t k, n = xc.shape[0]
    cdef double x0, x1, y0, y1, ax, ay, cx0, cy1
    cdef long I, I0, I1, J, J0, J1, i, j, c, ii, jj
    cdef long R0 = <long>row0 * cf, R1 = <long>row1 * cf
    for k in range(n):
        x0 = xc[k] * s
        x1 = x0 + (1u << lv[k]) * s
        y1 = y3 - yc[k] * s
        y0 = y1 - (1u << lv[k]) * s
        if y1 > hp:
            y1 = hp
        if y0 < -hp:
            y0 = -hp
        if y1 <= y0:
            continue
        J0 = <long>floor((ytop - y1) / ddy)
        J1 = <long>floor((ytop - y0) / ddy)
        if J1 < R0 or J0 >= R1:
            continue
        I0 = <long>floor((x0 - xedge) / ddx)
        I1 = <long>floor((x1 - xedge) / ddx)
        for J in range(max(J0, R0), min(J1, R1 - 1) + 1):
            cy1 = ytop - J * ddy
            ay = fmin(y1, cy1) - fmax(y0, cy1 - ddy)
            if ay <= 0.0:
                continue
            j = J // cf
            jj = J - j * cf
            for I in range(I0, I1 + 1):
                cx0 = xedge + I * ddx
                ax = fmin(x1, cx0 + ddx) - fmax(x0, cx0)
                if ax <= 0.0:
                    continue
                i = I // cf
                ii = I - i * cf
                i = i % nx
                if i < 0:
                    i += nx
                c = pos[(j - row0) * nx + i]
                if c >= 0:
                    out[c, jj * cf + ii] += ax * ay
    return out_np
