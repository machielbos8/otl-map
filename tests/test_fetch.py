"""otl-map fetch-coastline: a file with the wrong checksum is refused; the real coastline (from a local copy, when
one is available) is unpacked and verified, and a second call keeps it."""
import gzip
import os
from pathlib import Path

import pytest

from otl_map.fetch import SHA256, fetch_coastline, sha256

LOCAL = [os.environ.get("OTL_MAP_COASTLINE_GZ", ""),
         str(Path.home() / "projects/hecuba/coast/quadtree/quadtree_0.00137_v2.dat")]


def test_wrong_file_is_refused(tmp_path):
    gz = tmp_path / "fake.dat.gz"
    with gzip.open(gz, "wb") as f:
        f.write(b"not a coastline")
    dest = tmp_path / "out" / "quadtree.dat"
    with pytest.raises(RuntimeError, match="checksum"):
        fetch_coastline(dest, url=str(gz), quiet=True)
    assert not dest.exists()


def test_real_coastline_from_local_copy(tmp_path):
    src = next((p for p in LOCAL if p and Path(p).is_file()), None)
    if src is None:
        pytest.skip("no local copy of the coastline")
    gz = Path(src)
    if gz.suffix != ".gz":
        gz = tmp_path / "q.dat.gz"
        with open(src, "rb") as fi, gzip.open(gz, "wb", compresslevel=1) as fo:
            fo.write(fi.read())
    dest = fetch_coastline(tmp_path / "c" / "quadtree.dat", url=str(gz), quiet=True)
    assert sha256(dest) == SHA256
    mtime = dest.stat().st_mtime_ns
    fetch_coastline(dest, url="file:///nonexistent.gz", quiet=True)        # verified copy kept, no download
    assert dest.stat().st_mtime_ns == mtime
