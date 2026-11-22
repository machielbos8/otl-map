"""otl-map-examples copies the examples (and the manual when it is installed); the example scripts run exactly the
otl-map commands of the manual's tutorial, and Example 3 runs."""
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from otl_map import examples_cmd

EX = Path(examples_cmd.__file__).parent / "examples"
ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual" / "otl-map_manual_v0.1.md"


def test_copy(tmp_path):
    dest = tmp_path / "ex"
    assert examples_cmd.main([str(dest)]) == 0
    for f in ("README.txt", "ex1/example1.py", "ex1/brest.txt", "ex2/example2.py", "ex2/brest.txt", "ex3/example3.py"):
        assert (dest / f).is_file(), f
    with pytest.raises(SystemExit):                       # never overwrites
        examples_cmd.main([str(dest)])


def _commands(script):
    return runpy.run_path(str(script))["COMMANDS"]


@pytest.mark.skipif(not MANUAL.is_file(), reason="manual source not in this repository")
def test_scripts_match_manual():
    md = MANUAL.read_text()
    cmds = [" ".join(c.split()) for c in re.findall(r"^otl-map ([^\n#]+?)\s*(?:#.*)?$", md, re.M)]

    def section(title, end):
        i = md.index(title)
        j = md.index(end, i)
        return [" ".join(c.split()) for c in re.findall(r"^otl-map ([^\n#]+?)\s*(?:#.*)?$", md[i:j], re.M)]
    ex1 = section("## Example 1", "## Example 2")
    ex2 = section("## Example 2", "## Example 3")
    assert ex1 and ex2 and cmds
    assert ex1 == _commands(EX / "ex1" / "example1.py")
    s2 = _commands(EX / "ex2" / "example2.py")
    assert [c for c in s2 if c not in ("fetch-coastline", "resample SCHW -s 0.0625")] == ex2
    code = re.findall(r"```python\n(.*?)```", md, re.S)[0]
    assert code.strip() in (EX / "ex3" / "example3.py").read_text()


def test_example3_runs(tmp_path):
    env = dict(os.environ, OTL_MAP_DATA=str(tmp_path / "data"), OTL_MAP_CACHE=str(tmp_path / "cache"))
    env.pop("OTL_MAP", None)
    r = subprocess.run([sys.executable, str(EX / "ex3" / "example3.py")], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert re.search(r"M2 up: \d+\.\d\d mm", r.stdout)
