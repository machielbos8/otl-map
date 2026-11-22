# Roadmap to 0.1.0

Follows the release plan in the private otl_map repository (docs/release_plan_otl-map.md).

1. [x] Skeleton: pyproject/setup, licence, Cython modules copied, package data (Green's functions, SCHW),
       `otl_map.paths` instead of OTL_MAP, CLI stub, tests (Table 4, package data).
2. [ ] `otl_map.coastline`: water fractions and sub-cell fractions from the quadtree (replaces cpp/create_mask and
       cpp/create_subcells); validate against the C++ output; then port `modify_tidemodel`, `bundle.py` and
       `refine_grid.py` to the package layout.
3. [ ] Station output: bilinear u/v interpolation, near-field correction, BLQ/JSON, centre-of-mass correction.
4. [ ] CLI subcommands wired to the modules.
5. [ ] Tests: FFT = brute force on a small SCHW grid; kernel table = FFT; end-to-end SCHW.
6. [ ] CI (GitHub Actions), wheels (cibuildwheel), trusted publishing.
7. [ ] Coastline on Zenodo + `fetch-coastline`.
8. [ ] Documentation: tide-model format, Green's-function format, tutorial.
9. [ ] 0.1.0 on TestPyPI, then PyPI; Zenodo DOI.
