"""The otl-map manual cannot go stale: its Python examples run, and its `otl-map <cmd> --help` texts are those of
this version (tools/manual_examples.py; the tutorial outputs are checked with `--only ex1 --check`, about a
minute, and ex2 needs the coastline)."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual" / "otl-map_manual_v0.1.md"
# otl-map (public) does not carry the manual source yet (it is generated from otlmap-dev without docs/manual)
pytestmark = pytest.mark.skipif(not MANUAL.is_file() or not (ROOT / "tools" / "manual_examples.py").is_file(),
                                reason="manual source not in this repository")


def test_manual_python_blocks_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OTL_MAP_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("OTL_MAP_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"```python\n(.*?)```", MANUAL.read_text(), re.S)
    assert blocks
    ns = {}
    for k, code in enumerate(blocks):
        try:
            exec(compile(code, f"manual python block {k}", "exec"), ns)
        except Exception as e:
            raise AssertionError(f"manual python block {k} failed: {e!r}\n{code}") from e


def test_manual_help_texts_current():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "manual_examples.py"), "--only", "help", "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
