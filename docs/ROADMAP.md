# Roadmap to 0.1.0

Follows the release plan in the private otl_map repository (docs/release_plan_otl-map.md).

1. [x] Skeleton: pyproject/setup, licence, Cython modules copied, package data (Green's functions, SCHW),
       `otl_map.paths` instead of OTL_MAP, CLI stub, tests (Table 4, package data).
2. [x] `otl-map resample`: coarse tide models (Schwiderski 1 deg) onto the 1/16 deg FES2014b lattice
       (bilinear u/v where all four nodes are water, nearest water node within 2 source cells near the coast).
3. [ ] `otl_map.coastline`: water fractions and sub-cell fractions from the quadtree (replaces cpp/create_mask and
       cpp/create_subcells); validate against the C++ output; then port `modify_tidemodel`, `bundle.py` and
       `refine_grid.py` to the package layout.
4. [ ] Station output: bilinear u/v interpolation, near-field correction, BLQ/JSON, centre-of-mass correction.
5. [ ] CLI subcommands wired to the modules.
6. [ ] Tests: FFT = brute force on a small SCHW grid; kernel table = FFT; end-to-end SCHW.
7. [ ] CI (GitHub Actions), wheels (cibuildwheel), trusted publishing.
8. [ ] Coastline on Zenodo + `fetch-coastline`.
9. [ ] Documentation: tide-model format, Green's-function format, tutorial.
10. [ ] 0.1.0 on TestPyPI, then PyPI; Zenodo DOI.
