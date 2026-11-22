"""Where otl-map finds its data.

Green's functions: $OTL_MAP/greenfnc if it exists, else the tables shipped with the package.
Tide models: the first of $OTL_MAP_TIDEMODELS/<name>, $OTL_MAP/surfaceloads/<name>, the user data directory
(~/.local/share/otl-map/tidemodels/<name>) and the models shipped with the package (SCHW).
Coastline and other downloads: the cache directory (~/.cache/otl-map, or $OTL_MAP_CACHE).
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DATA = Path(__file__).resolve().parent / "data"


def _env_dir(var, *sub):
    v = os.environ.get(var)
    return Path(v, *sub) if v else None


def greenfnc_dir() -> str:
    d = _env_dir("OTL_MAP", "greenfnc")
    if d is not None and d.is_dir():
        return str(d)
    return str(PACKAGE_DATA / "greenfnc")


def user_data_dir() -> Path:
    return Path(os.environ.get("OTL_MAP_DATA", Path.home() / ".local" / "share" / "otl-map"))


def cache_dir() -> Path:
    return Path(os.environ.get("OTL_MAP_CACHE", Path.home() / ".cache" / "otl-map"))


def tidemodel_dir(name: str, create: bool = False) -> str:
    """Directory of tide model <name>; with create=True the user data directory is returned (and made) when the
    model does not exist yet (output of `otl-map prepare`)."""
    for d in (_env_dir("OTL_MAP_TIDEMODELS", name), _env_dir("OTL_MAP", "surfaceloads", name),
              user_data_dir() / "tidemodels" / name, PACKAGE_DATA / "tidemodels" / name):
        if d is not None and d.is_dir():
            return str(d)
    if create:
        d = (_env_dir("OTL_MAP_TIDEMODELS", name) or user_data_dir() / "tidemodels" / name)
        d.mkdir(parents=True, exist_ok=True)
        return str(d)
    raise FileNotFoundError(f"tide model {name!r} not found (set OTL_MAP_TIDEMODELS or put it in "
                            f"{user_data_dir() / 'tidemodels'})")
