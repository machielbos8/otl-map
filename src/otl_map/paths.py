"""Where otl-map finds and writes its data.

Environment variables (all optional):
  OTL_MAP              working tree of a production setup (the development repository): greenfnc/, surfaceloads/
                       (derived tide models), cpp/ (water fractions and sub-cells), refine_bundle_v2/
  OTL_MAP_TIDEMODELS   directory with the raw tide models as distributed (e.g. hecuba/surfaceload); read only
  OTL_MAP_CPP          directory with the water fractions mask_<model>.nc and sub-cells subcells_<model>/
  OTL_MAP_COASTLINE    the quadtree coastline file (quadtree_0.00137_v2.dat)
  OTL_MAP_DATA         user data directory (default ~/.local/share/otl-map), used when OTL_MAP is not set
  OTL_MAP_CACHE        cache directory (default ~/.cache/otl-map), used when OTL_MAP is not set
  CARGAHOME            hecuba; its coast/quadtree/ is searched for the coastline

Rules: raw tide models are read from OTL_MAP_TIDEMODELS, OTL_MAP/surfaceloads, the user data directory or the
package (SCHW); everything derived (resampled or prepared models, water fractions, sub-cells, bundles) is written
under OTL_MAP when it is set, otherwise under the user data and cache directories, and never into
OTL_MAP_TIDEMODELS.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DATA = Path(__file__).resolve().parent / "data"
COASTLINE = "quadtree_0.00137_v2.dat"


def _env(var, *sub):
    v = os.environ.get(var)
    return Path(v, *sub) if v else None


def user_data_dir() -> Path:
    return Path(os.environ.get("OTL_MAP_DATA", Path.home() / ".local" / "share" / "otl-map"))


def cache_dir() -> Path:
    return Path(os.environ.get("OTL_MAP_CACHE", Path.home() / ".cache" / "otl-map"))


def greenfnc_dir() -> str:
    d = _env("OTL_MAP", "greenfnc")
    if d is not None and d.is_dir():
        return str(d)
    return str(PACKAGE_DATA / "greenfnc")


def tidemodel_dir(name: str) -> str:
    """Directory of tide model <name> (raw or derived), for reading."""
    for d in (_env("OTL_MAP_TIDEMODELS", name), _env("OTL_MAP", "surfaceloads", name),
              user_data_dir() / "tidemodels" / name, PACKAGE_DATA / "tidemodels" / name):
        if d is not None and d.is_dir():
            return str(d)
    raise FileNotFoundError(f"tide model {name!r} not found (OTL_MAP_TIDEMODELS, OTL_MAP/surfaceloads, "
                            f"{user_data_dir() / 'tidemodels'})")


def derived_dir(name: str) -> str:
    """Directory for a derived tide model <name> (written; created if needed): $OTL_MAP/surfaceloads/<name>,
    else the user data directory. Never inside OTL_MAP_TIDEMODELS."""
    root = _env("OTL_MAP", "surfaceloads") or user_data_dir() / "tidemodels"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def masks_dir() -> Path:
    """Water fractions mask_<model>.nc and sub-cells subcells_<model>/."""
    return _env("OTL_MAP_CPP") or _env("OTL_MAP", "cpp") or cache_dir() / "masks"


def mask_file(model: str) -> str:
    return str(masks_dir() / f"mask_{model}.nc")


def subcells_dir(model: str) -> str:
    return str(masks_dir() / f"subcells_{model}")


def bundle_root() -> Path:
    """Near-field bundles <root>/<model>/."""
    return _env("OTL_MAP", "refine_bundle_v2") or cache_dir() / "refine_bundle_v2"


def coastline_file() -> str:
    for f in (_env("OTL_MAP_COASTLINE"), _env("CARGAHOME", "coast", "quadtree", COASTLINE), cache_dir() / COASTLINE):
        if f is not None and f.is_file():
            return str(f)
    raise FileNotFoundError("coastline not found: set OTL_MAP_COASTLINE, or run `otl-map fetch-coastline`")
