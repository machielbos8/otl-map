"""Cython extensions of otl-map (metadata is in pyproject.toml)."""
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

NP = [numpy.get_include()]
MODULES = {"geometry": [], "tidemodel": [], "greensfunction": NP, "cell_integration": NP,
           "compute_otl_slow": NP, "_quadtree": NP, "compute_otl_fft": NP, "modify_tidemodel": NP}
extensions = [Extension(f"otl_map.{m}", [f"src/otl_map/{m}.pyx"], include_dirs=inc) for m, inc in MODULES.items()]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
