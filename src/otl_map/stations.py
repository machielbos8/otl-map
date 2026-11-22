"""OTL at stations: bilinear interpolation of the FFT maps plus the near-field correction (paper, Eq. 6), and output
as BLQ or JSON, optionally with the centre-of-mass correction (CMC).

Conventions
- Internally (grids, near-field correction): components 0 up, 1 north, 2 east, positive in those directions;
  amplitudes in mm, Greenwich phase lag in degrees, phasor u + i v = A exp(i phase).
- BLQ: rows radial (up), tangential EW (positive WEST), tangential NS (positive SOUTH); amplitudes in m;
  Greenwich phase lag in degrees, lags positive. North and east are negated (phase + 180 deg) for BLQ.
- CMC (FES-style coefficients Gx, Gy, Gz in m): the displacement of the station relative to the centre of mass
  of the solid Earth plus oceans, added in the BLQ convention (radial, south, west) as in the online service.
"""
from __future__ import annotations

import datetime
import json
import math
import os

import netCDF4 as nc
import numpy as np

from otl_map import __version__
from otl_map.paths import bundle_root, tidemodel_dir
from otl_map.resample import read_grid

BLQ_HARMS = ["m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1", "mf", "mm", "ssa"]
HARM_DISPLAY = {"m2": "M2", "s2": "S2", "n2": "N2", "k2": "K2", "k1": "K1", "o1": "O1", "p1": "P1", "q1": "Q1",
                "mf": "Mf", "mm": "Mm", "ssa": "Ssa"}
RAD = math.pi / 180.0
RHO_W, A_EARTH, M_EARTH = 1030.0, 6.371e6, 5.972e24


# ------------------------------------------------------------------ input
def read_stations(path):
    """Station file: one station per line, `name lon lat [height]` (degrees, m); '#' starts a comment."""
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        out.append(dict(name=f[0], lon=float(f[1]), lat=float(f[2]), h=float(f[3]) if len(f) > 3 else 0.0))
    return out


# ------------------------------------------------------------------ interpolation of the FFT maps
def _grid(path):
    d = nc.Dataset(path)
    x, y = np.array(d["x"][:], float), np.array(d["y"][:], float)       # y ascending (south to north)
    z = np.ma.filled(d["z"][:].astype(float), np.nan)
    d.close()
    return x, y, z


def bilinear(x, y, U, lon, lat):
    """Bilinear interpolation of complex U (ny, nx; y ascending, x global and periodic) at (lon, lat)."""
    nx = len(x)
    sp = x[1] - x[0]
    periodic = abs(nx * sp - 360.0) < 1e-6
    fi = ((lon - x[0]) % 360.0) / sp if periodic else (lon - x[0]) / sp
    fj = (lat - y[0]) / (y[1] - y[0])
    i0 = int(math.floor(fi + 1e-9))
    j0 = min(max(int(math.floor(fj + 1e-9)), 0), len(y) - 2)
    if not periodic:
        i0 = min(max(i0, 0), nx - 2)            # regional grid: last column -> last cell with tx = 1
    tx, ty = fi - i0, fj - j0
    if periodic:
        i0 %= nx
        i1 = (i0 + 1) % nx
    else:
        i1 = i0 + 1
    return ((1 - tx) * (1 - ty) * U[j0, i0] + tx * (1 - ty) * U[j0, i1]
            + (1 - tx) * ty * U[j0 + 1, i0] + tx * ty * U[j0 + 1, i1])


def compute_stations(stations, model, earth="PREM", rundir=".", harmonics=None, label="fft", nearfield=True,
                     bundle=None, components=(0, 1, 2)):
    """OTL at stations (internal convention): dict name -> {harm: complex array (3,) in mm}; components not
    computed stay zero."""
    if harmonics is None:
        harmonics = [h for h in BLQ_HARMS if os.path.isfile(os.path.join(rundir, f"amp_fft0_{label}.{h}"))]
    res = {s["name"]: {} for s in stations}
    for h in harmonics:
        for c in components:
            x, y, A = _grid(os.path.join(rundir, f"amp_fft{c}_{label}.{h}"))
            _, _, P = _grid(os.path.join(rundir, f"pha_fft{c}_{label}.{h}"))
            U = A * np.exp(1j * np.radians(P))
            for s in stations:
                res[s["name"]].setdefault(h, np.zeros(3, complex))[c] = bilinear(x, y, U, s["lon"], s["lat"])
    if nearfield:
        from otl_map.nearfield import Bundle
        bdir = bundle or str(bundle_root() / model)
        if not os.path.isfile(os.path.join(bdir, "meta.json")):
            raise FileNotFoundError(f"near-field bundle {bdir} not found: run `otl-map bundle {model} {earth}` "
                                    "or use nearfield=False")
        B = Bundle(bdir, earth)
        hidx = {h: i for i, h in enumerate(B.harmonics)}
        for s in stations:
            corr = B.correction(s["lon"] % 360.0, s["lat"])
            if corr is None:
                continue
            for h in harmonics:
                if h in hidx:
                    for c in components:
                        res[s["name"]][h][c] += corr[c, hidx[h]]
    return res


# ------------------------------------------------------------------ centre-of-mass correction
def cmc_coefficients(model, harmonics=None):
    """CMC coefficients (m) of a tide model from its raw grids: G = -rho a^3 / M  sum h Y1 dA (U in-phase, V
    quadrature), for Y1 = (cos lat cos lon, cos lat sin lon, sin lat) -> Gx, Gy, Gz."""
    d = tidemodel_dir(model)
    if harmonics is None:
        harmonics = [h for h in BLQ_HARMS if os.path.isfile(os.path.join(d, f"amp.{h}"))]
    C = RHO_W * A_EARTH ** 3 / M_EARTH
    out = {}
    for h in harmonics:
        A = read_grid(os.path.join(d, f"amp.{h}"))
        P = read_grid(os.path.join(d, f"pha.{h}"))
        a = np.where(A["z"] < 999.0, A["z"], 0.0)
        p = np.radians(np.where(A["z"] < 999.0, P["z"], 0.0))
        lat, lon = np.radians(A["y"])[:, None], np.radians(A["x"])[None, :]
        dA = np.cos(lat) * np.radians(A["sp"][0]) * np.radians(A["sp"][1])
        g = {}
        for part, f in (("U", a * np.cos(p)), ("V", a * np.sin(p))):
            g[f"Gx_{part}"] = -C * float(np.sum(f * np.cos(lat) * np.cos(lon) * dA))
            g[f"Gy_{part}"] = -C * float(np.sum(f * np.cos(lat) * np.sin(lon) * dA))
            g[f"Gz_{part}"] = -C * float(np.sum(f * np.sin(lat) * dA))
        out[h] = g
    return out


def apply_cmc_blq(blq, stations, cmc):
    """Add the CMC to BLQ-convention values blq[name][h] = complex (3,) in mm (radial, south, west)."""
    for s in stations:
        phi, lam = s["lat"] * RAD, s["lon"] * RAD
        cp, sp_, cl, sl = math.cos(phi), math.sin(phi), math.cos(lam), math.sin(lam)
        for h, g in cmc.items():
            if h not in blq[s["name"]]:
                continue
            d = np.zeros(3, complex)
            for part, k in (("U", 1.0), ("V", 1j)):
                gx, gy, gz = g[f"Gx_{part}"], g[f"Gy_{part}"], g[f"Gz_{part}"]
                d[0] += k * (gx * cp * cl + gy * cp * sl + gz * sp_)          # radial
                d[1] += k * (gx * sp_ * cl + gy * sp_ * sl - gz * cp)         # south
                d[2] += k * (gx * sl - gy * cl)                               # west
            blq[s["name"]][h] = blq[s["name"]][h] + 1000.0 * d
    return blq


# ------------------------------------------------------------------ output
def to_blq_convention(res):
    """Internal (up, north, east) -> BLQ (radial, west, south) order is handled at writing; here: (up, south,
    west) = (up, -north, -east)."""
    return {n: {h: np.array([v[0], -v[1], -v[2]]) for h, v in d.items()} for n, d in res.items()}


def write_blq(path, stations, blq, model, earth, cmc=None):
    """BLQ file (amplitudes m; rows radial, tangential EW (west), tangential NS (south); phase lags in degrees)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    L = ["$$ Ocean loading displacement", "$$", f"$$ otl-map {__version__}", f"$$ {now:%Y-%m-%d %H:%M:%S} UTC", "$$",
         "$$ COLUMN ORDER:  M2  S2  N2  K2  K1  O1  P1  Q1  MF  MM SSA", "$$", "$$ ROW ORDER:", "$$ AMPLITUDES (m)",
         "$$   RADIAL", "$$   TANGENTL    EW", "$$   TANGENTL    NS", "$$ PHASES (degrees)", "$$   RADIAL",
         "$$   TANGENTL    EW", "$$   TANGENTL    NS", "$$",
         "$$ Displacement is defined positive in upwards, South and West direction.",
         f"$$ The phase lag is relative to Greenwich and lags positive. The {earth}",
         "$$ Green's function is used.", "$$", f"$$ CMC:  {'YES' if cmc else 'NO'} (corr.tide centre of mass)", "$$",
         "$$ A constant seawater density of 1030 kg/m^3 is used.", "$$", "$$ END HEADER"]
    for s in stations:
        d = blq[s["name"]]
        L += ["$$", f"  {s['name']}", f"$$ {model} {earth}",
              f"$$ {s['name']:<24s}{'RADI TANG':<9s}  lon/lat:{s['lon']:9.4f} {s['lat']:9.4f} {s['h']:9.3f}"]
        for c in (0, 2, 1):                       # radial, EW (west), NS (south)
            L.append(" " + "".join(" " + f"{abs(d[h][c]) / 1000.0 if h in d else 0.0:7.5f}"[1:] for h in BLQ_HARMS))
        for c in (0, 2, 1):
            L.append(" " + "".join(f"{math.degrees(np.angle(d[h][c])) if h in d else 0.0:7.1f}" for h in BLQ_HARMS))
    L.append("$$ END TABLE")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def write_json(path, stations, res, model, earth):
    """JSON in the internal convention (up, north, east positive; mm; phase lag in degrees)."""
    out = dict(otl_map=__version__, tide_model=model, earth_model=earth, convention="up, north, east positive; "
               "amplitude mm; Greenwich phase lag deg", stations=[])
    for s in stations:
        out["stations"].append(dict(name=s["name"], lon=s["lon"], lat=s["lat"], h=s["h"], otl={
            h: dict(amp=[round(float(abs(v[c])), 4) for c in range(3)],
                    pha=[round(float(math.degrees(np.angle(v[c]))), 3) for c in range(3)])
            for h, v in res[s["name"]].items()}))
    json.dump(out, open(path, "w"), indent=1)
    return path
