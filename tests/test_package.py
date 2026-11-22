"""Package data and imports."""
import os

from otl_map import __version__
from otl_map.gf_spline import GF
from otl_map.paths import greenfnc_dir, tidemodel_dir


def test_version():
    assert __version__


def test_greens_functions_shipped():
    for m in ("PREM", "PREM_disp", "STW105", "STW105_disp"):
        f = os.path.join(greenfnc_dir(), f"{m}.dat")
        assert os.path.isfile(f)
        g = GF(f, 0)
        assert g.interp(1e-3 * 3.141592653589793 / 180) < 0          # vertical displacement is downward


def test_schwiderski_shipped():
    d = tidemodel_dir("SCHW")
    assert os.path.isfile(os.path.join(d, "amp.m2")) and os.path.isfile(os.path.join(d, "pha.m2"))


def test_cli_help():
    from otl_map.cli import main
    assert main([]) == 0
