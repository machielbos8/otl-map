otl-map examples (made by `otl-map-examples`)

  otl-map_manual_v0.1.pdf   the user manual; the tutorial (Section 5) explains these examples step by step
  ex1/   a first map without the coastline: Schwiderski M2 around Brest, OTL at BRST   (about a minute)
  ex2/   the complete chain with the coastline: water fractions, tide preparation, FFT,
         near-field correction, BLQ file for BRST                                      (a few minutes)
  ex3/   the same from Python, on a coarse grid                                         (seconds)

Run an example from its folder, with the virtual environment of otl-map activated:

  cd ex1
  python example1.py

Each script prints the otl-map commands it runs; you can also type them yourself (`otl-map --help`).
Derived tide models go to ~/.local/share/otl-map, water fractions, bundles and the coastline to
~/.cache/otl-map (Section 2 of the manual).
