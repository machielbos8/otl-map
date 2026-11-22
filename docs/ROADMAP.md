# Roadmap

## 0.1.0 (released November 2026)

Follows the release plan in the private otl_map repository (docs/release_plan_otl-map.md).

1. [x] Skeleton: pyproject/setup, licence, Cython modules copied, package data (Green's functions, SCHW),
       `otl_map.paths` instead of OTL_MAP, CLI stub, tests (Table 4, package data).
2. [x] `otl-map resample`: coarse tide models (Schwiderski 1 deg) onto the 1/16 deg FES2014b lattice
       (bilinear u/v where all four nodes are water, nearest water node within 2 source cells near the coast).
3. [x] `otl_map.coastline` (+ Cython `_quadtree`): water fractions and sub-cell fractions from the quadtree,
       replacing cpp/create_mask and cpp/create_subcells (`otl-map masks`). Validated for FES2014b: water fractions
       equal to 2e-12 (same 245 376 coastal cells), sub-cells identical except 13 247 of 883 million values that
       differ by 1/255 (rounding); 0.5 s and 26 s against hours. Sub-cells beyond a pole count as land, as in C++.
3b. [x] `otl-map prepare` (NumPy rewrite of modify_tidemodel: one-cell extension + water-fraction scaling),
       `otl-map bundle` and `otl-map refine` on the package layout (cache directory, no repo paths).
       Validated for FES2014b against the production files of the development repository: prepared M2 equal
       to 2e-12; complete PREM bundle rebuilt from scratch (masks -> prepare -> bundle, 2.5 min on 8 cores):
       idx, sptr and the coarse kernel table identical, w0 to 2e-12, tide to 1.5e-8 m (float32), sub-cells as in 3.
4. [x] `otl-map compute` (FFT without a config file, per component) and `otl-map stations` (bilinear u/v + near
       field, BLQ with south/west positive or JSON, optional CMC computed from the raw tide model). Validated: the
       13 EPN stations reproduce the supplement table (FFT + near field, 429 values) to 0.0035 mm (table rounding);
       FES2014b CMC agrees with the server coefficients to 1-2 % with the right signs. Fixed in the public copy:
       compute_otl_slow applied rho and the cell area twice for cells integrated over the cell.
5. [x] CLI: resample, fetch-coastline, masks, prepare, compute, bundle, refine, stations; otl-map-examples.
6. [~] Tests: FFT = brute force (vertical) on SCHW at 0.5 deg; compute -> stations -> BLQ. Pending: kernel table = FFT.
7. [x] CI (GitHub Actions): sdist + wheels (Linux x86_64/aarch64, macOS arm64, Windows; Python 3.10-3.14), each
       tested; TestPyPI for rc tags, PyPI for release tags (trusted publishing).
8. [x] `fetch-coastline` from the GitHub release coastline-v2 (checksum verified). Zenodo: later.
9. [x] User manual (PDF, in the package): installation, conventions, method, tutorial with three examples,
       commands, Python API, file formats.
10. [x] 0.1.0rc1 on TestPyPI (tested from a clean install), 0.1.0 on PyPI.

## Next

- Kernel-table test (coarse table = FFT kernel).
- `otl-map fit-greens`: pole-plus-polynomial constants for a user Green's function.
- Zenodo: DOI for the software and the coastline.
