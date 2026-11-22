"""Reader for the binary output of cpp/create_subcells (quadtree sub-cell water fractions)."""
from __future__ import annotations

import numpy as np


def read_subcells(path):
    """Return dict(nx, ny, cf, row0, row1, xr0, yr1, spx, spy, k (int64), w (float64), f (uint8 [n, cf*cf]))."""
    raw = np.fromfile(path, dtype=np.uint8)
    nx, ny, cf, row0, row1 = np.frombuffer(raw[:20], "<i4")
    xr0, yr1, spx, spy = np.frombuffer(raw[20:52], "<f8")
    rec = np.dtype([("k", "<i8"), ("w", "<f8"), ("f", np.uint8, (int(cf) * int(cf),))])
    body = np.frombuffer(raw[52:], dtype=rec)
    return dict(nx=int(nx), ny=int(ny), cf=int(cf), row0=int(row0), row1=int(row1), xr0=xr0, yr1=yr1,
                spx=spx, spy=spy, k=body["k"].copy(), w=body["w"].copy(), f=body["f"].copy())
