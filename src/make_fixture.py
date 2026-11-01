"""Generate fixture_data.json following SCHEMA.md (random plausible values) for developing the map page."""
import json
import math
import random
from pathlib import Path

random.seed(7)
HERE = Path(__file__).parent
LAT0, LON0 = 59.915, 10.75
NCOL, NROW = 6, 5
DLAT = 0.0045   # ~500 m
DLON = 0.0090   # ~500 m at 60 N
BUDGETS_RUN = [40, 20]

features = []
for z in range(NCOL * NROW):
    r, c = divmod(z, NCOL)
    lat_c = LAT0 + (r - (NROW - 1) / 2) * DLAT
    lon_c = LON0 + (c - (NCOL - 1) / 2) * DLON
    j = lambda: random.uniform(-0.0004, 0.0004)
    corners = [
        [lon_c - DLON / 2, lat_c - DLAT / 2],
        [lon_c + DLON / 2, lat_c - DLAT / 2],
        [lon_c + DLON / 2, lat_c + DLAT / 2],
        [lon_c - DLON / 2, lat_c + DLAT / 2],
    ]
    corners = [[round(x + j(), 6), round(y + j() / 2, 6)] for x, y in corners]
    corners.append(corners[0])
    features.append({
        "type": "Feature",
        "properties": {"zone": z, "cap": random.randint(40, 220), "n_stations": random.randint(2, 9),
                       "lat": round(lat_c, 6), "lon": round(lon_c, 6)},
        "geometry": {"type": "Polygon", "coordinates": [corners]},
    })

N = len(features)


def profile(h, kind):
    am = math.exp(-((h - 8) ** 2) / 4.5)
    pm = math.exp(-((h - 17) ** 2) / 6.0)
    mid = math.exp(-((h - 13) ** 2) / 30.0)
    if kind == "wd":
        return 0.15 + 1.0 * am + 1.1 * pm + 0.35 * mid
    return 0.15 + 0.2 * am + 0.35 * pm + 0.9 * math.exp(-((h - 14) ** 2) / 18.0)


def make(scale, noise=0.15, sign=1, sparse=False):
    out = {}
    for dt in ("wd", "we"):
        rows = []
        for z in range(N):
            zs = zones_scale[z] * scale
            row = []
            for h in range(24):
                v = zs * profile(h, dt) * (1 + random.uniform(-noise, noise))
                if sparse and random.random() < 0.5:
                    v *= 0.1
                if sign < 0:
                    v = v * 0.4 - zs * 0.1 * random.random() * 0.5 + zs * 0.05 * random.uniform(0, 1)
                row.append(round(v, 3))
            rows.append(row)
        out[dt] = rows
    return out


zones_scale = [random.uniform(0.4, 2.2) for _ in range(N)]
layers = {
    "dep": make(5.0),
    "arr": make(5.0),
    "dep_gru": make(5.0, 0.2),
    "arr_gru": make(5.0, 0.2),
}
for b in BUDGETS_RUN:
    f = 1.0 if b == 40 else 1.8
    layers["lost_none_b%d" % b] = make(1.2, 0.2, sparse=True)
    layers["lost_heur_b%d" % b] = make(0.3 * f, 0.3, sparse=True)
    layers["lost_mpc_b%d" % b] = make(0.04 * f, 0.4, sparse=True)
    layers["lost_oracle_b%d" % b] = make(0.015 * f, 0.4, sparse=True)
    g = {}
    for dt in ("wd", "we"):
        g[dt] = [[round(layers["lost_mpc_b%d" % b][dt][z][h] - layers["lost_oracle_b%d" % b][dt][z][h]
                        + random.uniform(-0.02, 0.02), 3) for h in range(24)] for z in range(N)]
    layers["oracle_gain_b%d" % b] = g
    layers["shortage_b%d" % b] = make(0.8 * f, 0.3, sparse=True)
    layers["overflow_b%d" % b] = make(0.6 * f, 0.3, sparse=True)

weekly = {"none": 6124.0}
for b, f in ((40, 1.0), (20, 1.8)):
    weekly["heur_b%d" % b] = round(1016.25 * f, 2)
    weekly["mpc_b%d" % b] = round(115.25 * f, 2)
    weekly["oracle_b%d" % b] = round(33.25 * f, 2)

data = {
    "meta": {
        "title": "Oslo bike-share: demand and rebalancing (prototype)",
        "period": "4 validation weeks, Mon 2025-09-01 to Sun 2025-09-28",
        "n_zones": N,
        "budgets_run": BUDGETS_RUN,
        "horizon": 6,
        "built": "2026-10-31T23:00:00",
    },
    "zones": {"type": "FeatureCollection", "features": features},
    "daytypes": ["wd", "we"],
    "layers": layers,
    "weekly": weekly,
    "curve": {
        "budgets": [10, 20, 40, 60, 80],
        "series": {
            "heuristic": [4995.0, 3880.75, 1016.25, 422.0, 148.75],
            "lp_gru": [4600.0, 3100.0, 700.0, 300.0, 120.0],
            "mpc_gru": [3900.0, 1500.0, 115.25, 40.0, 12.0],
            "mpc_oracle": [3600.0, 1200.0, 33.25, 9.0, 3.0],
        },
        "trips_per_week": 36400.0,
    },
}

out = HERE / "fixture_data.json"
out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes")
