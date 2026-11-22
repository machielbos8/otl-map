# otl-map

Global ocean tide loading (OTL) maps computed with an FFT along latitude rows, with a coastal near-field
correction that also gives the OTL at any station.

**Status: under development (0.1.0.dev0). Not yet released.**

The method is described in Scherneck and Bos (in preparation), *Fast Computation of Ocean Tide Loading
Maps* (Journal of Geodesy). otl-map is for off-line computation (maps, many stations, other tide or Earth
models). For the OTL corrections of some stations, use the Onsala service at
<https://elder.oso.chalmers.se/loading/l.php>; for automatic computations from your own software, the OTL
API at <https://otl.teromovigo.com> (FES2014b, FES2022b, GOT5.5, TPXO10-atlas).

## What it does

- prepares a tide model: water fraction of every cell from a full-resolution coastline, scaling of the tide by
  this fraction, extension by one cell over the coast;
- computes global or regional OTL maps (vertical, north, east) with the row-wise FFT, with the Green's function
  integrated analytically over the nearby cells;
- applies the near-field correction to refine the map at coastal grid nodes, or to compute the OTL at a station;
- writes station values in BLQ or JSON format, with or without the centre-of-mass correction.

Shipped with the package: load Green's functions for PREM and STW105, elastic and with anelastic dispersion
(CE and CM frames), and the Schwiderski (1980) M2 tide model as an example. The full-resolution coastline
(derived from GSHHG, Wessel and Smith 1996) is downloaded once on first use. Other tide models are not
distributed; see the documentation for the expected format.

## Installation

In a virtual environment of its own (Linux and macOS; on Windows `py -m venv` and `Scripts\Activate.ps1`):

```bash
python3 -m venv ~/venvs/otl-map
source ~/venvs/otl-map/bin/activate
pip install otl-map
otl-map --version
```

pip also installs the dependencies (NumPy, SciPy, netCDF4, tqdm); no `requirements.txt` is needed. Then

```bash
otl-map-examples
```

copies the user manual (PDF) and the tutorial examples into a new folder `otl-map-examples`. Open the manual
first; `cd otl-map-examples/ex1 && python example1.py` runs the first example.

From source, for development:

```bash
git clone https://github.com/machielbos8/otl-map.git && cd otl-map
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## Licence

Free for non-commercial use (research, academic, educational). Commercial use requires a separate licence from
TeroMovigo – Earth Innovation Lda (info@teromovigo.com). See [LICENSE](LICENSE).
