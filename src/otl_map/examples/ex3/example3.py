"""Example 3 of the otl-map manual: otl-map from Python.

Schwiderski's M2 model on a coarse 0.5 deg grid (so that it runs in seconds), the vertical loading on 3 x 2
nodes in the open Pacific, and the value at a point between them. No coastline is needed.

    python example3.py
"""
from otl_map.resample import resample_tidemodel
from otl_map.compute import compute
from otl_map import stations as S

resample_tidemodel("SCHW", "SCHW_2", spacing=0.5, harmonics=["m2"])
mesh = [200.0, 201.0, 0.0, 0.5, 0.5, 0.5]        # lon0, lon1, lat0, lat1, dlon, dlat
compute("SCHW_2", "PREM", "pacific", components=[0], harmonics=["m2"], mesh=mesh,
        prepared=False)
st = [dict(name="MID", lon=200.25, lat=0.25, h=0.0)]
res = S.compute_stations(st, "SCHW_2", "PREM", "pacific", ["m2"], nearfield=False,
                         components=[0])
u = res["MID"]["m2"][0]                           # complex, mm, up
print(f"M2 up: {abs(u):.2f} mm")
