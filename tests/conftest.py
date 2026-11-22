"""Package tests run without any production setup: the OTL_MAP* variables and CARGAHOME of the developer's shell
are removed for the whole session (before any fixture, also module-scoped ones), and data and cache go to a
temporary directory."""
import pytest

VARS = ("OTL_MAP", "OTL_MAP_TIDEMODELS", "OTL_MAP_CPP", "OTL_MAP_COASTLINE", "OTL_MAP_DATA", "OTL_MAP_CACHE",
        "CARGAHOME")


@pytest.fixture(autouse=True, scope="session")
def _clean_env(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for v in VARS:
        mp.delenv(v, raising=False)
    base = tmp_path_factory.mktemp("otlmap_home")
    mp.setenv("OTL_MAP_DATA", str(base / "data"))
    mp.setenv("OTL_MAP_CACHE", str(base / "cache"))
    yield
    mp.undo()
