"""Generate fixture_data.json following SCHEMA.md (version 2, random plausible values) for developing the map page."""
import json
import math
import random
import sys
from pathlib import Path

random.seed(7)
HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data" if HERE.name == "src" else HERE
LAT0, LON0 = 59.915, 10.75
NCOL, NROW = 6, 5
DLAT = 0.0045   # ~500 m
DLON = 0.0090   # ~500 m at 60 N
BUDGETS_RUN = [40, 20]
N_STATIONS = 276
N_EDGES = 200
PERIODS = ("val", "test")
# demand and loss multipliers per period (test is busier and loses more)
DEMAND_F = {"val": 1.0, "test": 1.35}
LOST_F = {"val": 1.0, "test": 1.9}

# ---------------------------------------------------------------- zones (jittered grid cells)
features = []
centres = []
for z in range(NCOL * NROW):
    r, c = divmod(z, NCOL)
    lat_c = LAT0 + (r - (NROW - 1) / 2) * DLAT
    lon_c = LON0 + (c - (NCOL - 1) / 2) * DLON
    centres.append((lat_c, lon_c))
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
        "properties": {"zone": z, "cap": 0, "n_stations": 0,
                       "lat": round(lat_c, 6), "lon": round(lon_c, 6)},
        "geometry": {"type": "Polygon", "coordinates": [corners]},
    })
N = len(features)

# ---------------------------------------------------------------- stations
NAMES = ["Tøyenparken", "Grünerløkka", "Torggata", "Youngstorget", "Vaterland", "Skippergata", "Aker Brygge",
         "Frogner plass", "Majorstuen", "Bislett", "Sagene", "Birkelunden", "Alexander Kiellands plass",
         "Carl Berners plass", "Schous plass", "Olaf Ryes plass"]
stations = []
for i in range(N_STATIONS):
    z = i % N if i < N else random.randrange(N)
    lat_c, lon_c = centres[z]
    cap = random.choice([12, 15, 18, 20, 24, 28, 32])
    stations.append({
        "id": 300 + i,
        "name": "%s %d" % (random.choice(NAMES), i),
        "lat": round(lat_c + random.uniform(-DLAT / 2.3, DLAT / 2.3), 6),
        "lon": round(lon_c + random.uniform(-DLON / 2.3, DLON / 2.3), 6),
        "zone": z,
        "cap": cap,
    })
for f in features:
    z = f["properties"]["zone"]
    zs = [s for s in stations if s["zone"] == z]
    f["properties"]["n_stations"] = len(zs)
    f["properties"]["cap"] = sum(s["cap"] for s in zs)

# ---------------------------------------------------------------- edges: the N_EDGES closest zone pairs
pairs = []
for i in range(N):
    for j in range(i + 1, N):
        d = math.hypot((centres[i][0] - centres[j][0]) * 111.0,
                       (centres[i][1] - centres[j][1]) * 55.5)
        pairs.append((d, i, j))
pairs.sort()
edges = sorted([[i, j] for _, i, j in pairs[:N_EDGES]])


# ---------------------------------------------------------------- layers
def profile(h, kind):
    am = math.exp(-((h - 8) ** 2) / 4.5)
    pm = math.exp(-((h - 17) ** 2) / 6.0)
    mid = math.exp(-((h - 13) ** 2) / 30.0)
    if kind == "wd":
        return 0.15 + 1.0 * am + 1.1 * pm + 0.35 * mid
    return 0.15 + 0.2 * am + 0.35 * pm + 0.9 * math.exp(-((h - 14) ** 2) / 18.0)


zones_scale = [random.uniform(0.4, 2.2) for _ in range(N)]


def make(scale, noise=0.15, sparse=False):
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
                row.append(round(v, 3))
            rows.append(row)
        out[dt] = rows
    return out


layers = {}
weekly = {}
trips_per_week = {}
ladder = {"budgets": [20, 40]}
for p in PERIODS:
    df, lf = DEMAND_F[p], LOST_F[p]
    L = {
        "dep": make(5.0 * df),
        "arr": make(5.0 * df),
        "dep_gru": make(5.0 * df, 0.2),
        "arr_gru": make(5.0 * df, 0.2),
    }
    for b in BUDGETS_RUN:
        f = (1.0 if b == 40 else 1.8) * lf
        L["lost_none_b%d" % b] = make(1.2 * lf, 0.2, sparse=True)
        L["lost_heur_b%d" % b] = make(0.3 * f, 0.3, sparse=True)
        L["lost_mpc_b%d" % b] = make(0.04 * f, 0.4, sparse=True)
        L["lost_oracle_b%d" % b] = make(0.015 * f, 0.4, sparse=True)
        g = {}
        for dt in ("wd", "we"):
            g[dt] = [[round(L["lost_mpc_b%d" % b][dt][z][h] - L["lost_oracle_b%d" % b][dt][z][h]
                            + random.uniform(-0.02, 0.02), 3) for h in range(24)] for z in range(N)]
        L["oracle_gain_b%d" % b] = g
        L["shortage_b%d" % b] = make(0.8 * f, 0.3, sparse=True)
        L["overflow_b%d" % b] = make(0.6 * f, 0.3, sparse=True)
        # learned-policy layers (version 2.1): PPO-GNN slightly below MPC, PPO-MLP between MPC and PPO-GNN
        mpc = L["lost_mpc_b%d" % b]
        for key, shrink, spread in (("lost_ppo_gnn", 0.90, 0.3), ("lost_ppo_mlp", 0.95, 0.3)):
            L["%s_b%d" % (key, b)] = {
                dt: [[round(max(0.0, mpc[dt][z][h] * shrink * (1 + random.uniform(-spread, spread))), 3)
                      for h in range(24)] for z in range(N)] for dt in ("wd", "we")}
        gnn = L["lost_ppo_gnn_b%d" % b]
        L["ppo_gain_b%d" % b] = {
            dt: [[round(mpc[dt][z][h] - gnn[dt][z][h], 3) for h in range(24)] for z in range(N)]
            for dt in ("wd", "we")}
        # override share 0..0.6, higher at rush hours; policy_net -3..3 bikes per hour (sign flips by zone)
        L["override_b%d" % b] = {
            dt: [[round(min(0.6, max(0.0, 0.05 + 0.5 * zones_scale[z] / 2.2 * profile(h, dt) / 2.0
                                      * (1 + random.uniform(-0.3, 0.3)))), 3) for h in range(24)] for z in range(N)]
            for dt in ("wd", "we")}
        sign = [random.choice((-1, 1)) for _ in range(N)]
        L["policy_net_b%d" % b] = {
            dt: [[round(max(-3.0, min(3.0, sign[z] * 2.2 * profile(h, dt) / 2.2 * zones_scale[z] / 1.5
                                       + random.uniform(-0.3, 0.3))), 3) for h in range(24)] for z in range(N)]
            for dt in ("wd", "we")}
    # dock-level layers (version 2.1): per 100 docks and mean fill share
    caps = [f["properties"]["cap"] or 1 for f in features]
    L["dep_per_dock"] = {dt: [[round(L["dep"][dt][z][h] / caps[z] * 100, 3) for h in range(24)] for z in range(N)]
                         for dt in ("wd", "we")}
    L["fill"] = {dt: [[round(min(0.95, max(0.05, 0.55 - 0.18 * (profile(h, dt) - 0.5) * (1 if z % 2 else -1)
                                           + 0.2 * (zones_scale[z] - 1.3) / 1.0 * 0.5
                                           + random.uniform(-0.05, 0.05))), 3) for h in range(24)] for z in range(N)]
                 for dt in ("wd", "we")}
    layers[p] = L
    w = {"none": round(6118.0 * lf, 1)}
    for b, bf in ((40, 1.0), (20, 1.8)):
        w["heur_b%d" % b] = round(1016.25 * bf * lf, 1)
        w["mpc_b%d" % b] = round(115.25 * bf * lf, 1)
        w["oracle_b%d" % b] = round(33.25 * bf * lf, 1)
        w["ppo_gnn_b%d" % b] = round(w["mpc_b%d" % b] * 0.9, 1)
        w["ppo_mlp_b%d" % b] = round(w["mpc_b%d" % b] * 0.95, 1)
    weekly[p] = w
    trips_per_week[p] = round(32400.0 * df, 1)

none_t = weekly["test"]["none"]
ladder["test"] = [
    {"key": "none", "label": "No rebalancing", "kind": "reference", "lost": [none_t, none_t]},
    {"key": "heur", "label": "Hourly occupancy rule", "kind": "reference", "lost": [3944.0, 2597.1]},
    {"key": "mpc", "label": "Full MPC", "kind": "reference", "lost": [1603.0, 521.8]},
    {"key": "ppo_mlp", "label": "PPO, MLP policy", "kind": "learned", "lost": [1437.1, 330.9]},
    {"key": "ppo_gnn", "label": "PPO, GNN policy", "kind": "learned", "lost": [1371.3, 302.2]},
    {"key": "oracle", "label": "MPC, perfect forecast", "kind": "ceiling", "lost": [1376.4, 228.8]},
]
wv = weekly["val"]
ladder["val"] = [
    {"key": "none", "label": "No rebalancing", "kind": "reference", "lost": [wv["none"], wv["none"]]},
    {"key": "heur", "label": "Hourly occupancy rule", "kind": "reference", "lost": [wv["heur_b20"], wv["heur_b40"]]},
    {"key": "mpc", "label": "Full MPC", "kind": "reference", "lost": [wv["mpc_b20"], wv["mpc_b40"]]},
    {"key": "oracle", "label": "MPC, perfect forecast", "kind": "ceiling", "lost": [wv["oracle_b20"], wv["oracle_b40"]]},
]

data = {
    "meta": {
        "title": "Oslo bike-share: demand and rebalancing (FIXTURE, random values)",
        "source": "fixture",
        "periods": {
            "val": {"label": "Validation: 4 weeks, Sept 2025", "n_weeks": 4, "days": {"wd": 20, "we": 8}},
            "test": {"label": "Test: 21 weeks, May to Sept 2026", "n_weeks": 21, "days": {"wd": 105, "we": 42}},
        },
        "budgets_run": BUDGETS_RUN,
        "horizon": 6,
        "built": "2026-11-01T21:30:00",
        "ppo": {"gnn_checkpoints": ["fixture"], "mlp_checkpoints": ["fixture"], "seeds": [0, 1, 2], "mode": "argmax"},
    },
    "zones": {"type": "FeatureCollection", "features": features},
    "stations": stations,
    "edges": edges,
    "daytypes": ["wd", "we"],
    "layers": layers,
    "weekly": weekly,
    "trips_per_week": trips_per_week,
    "ladder": ladder,
}

# optional first argument: output folder (so the fixture can be written outside the repo)
if len(sys.argv) > 1:
    DATA_DIR = Path(sys.argv[1])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
out = DATA_DIR / "fixture_data.json"
out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes")
