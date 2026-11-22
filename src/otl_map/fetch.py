"""Download the full-resolution coastline (`otl-map fetch-coastline`).

The coastline is the quadtree `quadtree_0.00137_v2.dat` (finest cell 0.00137 deg, about 150 m), derived from the
full-resolution GSHHG shorelines (Wessel and Smith 1996; GNU LGPL), with the grounding line around Antarctica.
It is published, gzipped (9.7 MB), as an asset of the GitHub release `coastline-v2` of otl-map. The download is
unpacked into the cache directory (otl_map.paths.cache_dir) and accepted only if its SHA-256 is the one below.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from otl_map.paths import COASTLINE, cache_dir

URL = "https://github.com/machielbos8/otl-map/releases/download/coastline-v2/quadtree_0.00137_v2.dat.gz"
SHA256 = "29f9b4a8a900b32c0b33474278765e71dabae8f9ee3012407b5b07e8a4a8b801"     # of the unpacked file
SIZE = 179597540


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch_coastline(dest: str | os.PathLike | None = None, url: str = URL, force: bool = False,
                    quiet: bool = False) -> Path:
    """Download and unpack the coastline to `dest` (default: <cache>/quadtree_0.00137_v2.dat); returns its path.

    An existing file with the right checksum is kept unless `force`. `url` may also be a local .gz file (file path
    or file:// URL), e.g. a copy received by other means."""
    dest = Path(dest) if dest else cache_dir() / COASTLINE
    if dest.is_file() and not force:
        if dest.stat().st_size == SIZE and sha256(dest) == SHA256:
            if not quiet:
                print(f"coastline present and verified: {dest}")
            return dest
        if not quiet:
            print(f"{dest} exists but its checksum differs: downloading again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(url):
        url = Path(url).resolve().as_uri()
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        gz = Path(tmp) / "coastline.dat.gz"
        if not quiet:
            print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r, open(gz, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            bar = None
            if not quiet and total:
                from tqdm import tqdm
                bar = tqdm(total=total, unit="B", unit_scale=True)
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                f.write(b)
                if bar:
                    bar.update(len(b))
            if bar:
                bar.close()
        part = Path(tmp) / "coastline.dat"
        with gzip.open(gz, "rb") as fi, open(part, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        got = sha256(part)
        if got != SHA256:
            raise RuntimeError(f"coastline checksum mismatch ({got} != {SHA256}); the download is not used")
        os.replace(part, dest)
    if not quiet:
        print(f"coastline -> {dest} (checksum verified)")
    return dest
