"""Build map_data.json and layers.geojson for the Oslo rebalancing map prototype (see SCHEMA.md).

Reads from v2/code, v2/data_out, v2/runs (read-only). Writes only into this folder.
Run from this folder:  python build_data.py
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import sys, time, json, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Where the project's code/, data_out/ and runs/ folders live (the course project's v2 folder).
# Set OSLO_PROJECT_DIR to point at it; the default assumes this script sits inside that folder.
V2 = Path(os.environ.get("OSLO_PROJECT_DIR", HERE.parent))
sys.path.insert(0, str(V2 / "code"))
# Outputs go to ../data when this script is in the repo's src/ folder, else next to the script.
OUT = HERE.parent / "data" if HERE.name == "src" and (HERE.parent / "data").is_dir() else HERE

import numpy as np
import pandas as pd
from shapely import MultiPoint, Point, box, voronoi_polygons
from shapely.geometry import mapping

import common
from common import N_ZONES
from sim import ReplaySim, HourlyOcc, NoRebalance, run_episode
from mpc import MPCController, oracle_forecast

T_START = time.time()
Z = N_ZONES
BUDGETS = [40, 20]
HORIZON = 6
DT = ["wd", "we"]

# ----------------------------------------------------------------------------- 2. sim, forecasts
sim = ReplaySim(budget=40)
starts = sim.episode_starts("val")
assert len(starts) == 4, starts
print("episode starts:", starts, [str(sim.hours[s]) for s in starts])

profile = np.asarray(common.preds_to_array(common.load_preds("naive", 0), sim.hours))
gru_seeds = [common.preds_to_array(common.load_preds("gru", s), sim.hours) for s in (0, 1, 2)]
gru = np.nanmean(np.stack(gru_seeds), axis=0).astype(np.float32)
actual = oracle_forecast(sim, "val")
assert np.isfinite(actual[starts[0]:starts[0] + 168]).all()

# ----------------------------------------------------------------------------- 4. validation hours
val_h = np.unique(np.concatenate([np.arange(s, s + 168) for s in starts]))
assert len(val_h) == 4 * 168
hod_of = sim.hod
dt_of = np.where(sim.dow < 5, 0, 1)                  # 0 = wd, 1 = we
n_days = {}
for k, name in enumerate(DT):
    n_days[name] = int(((dt_of[val_h] == k)).sum() // 24)
print("days per daytype:", n_days)
assert n_days == {"wd": 20, "we": 8}, n_days


def cell_hours(k, hh):
    """indices h (into the hour grid, restricted to validation) with daytype k and hour of day hh"""
    return val_h[(dt_of[val_h] == k) & (hod_of[val_h] == hh)]


def mean_layer(arr_TZ, nan=False):
    """(T, Z) -> {'wd': (Z,24), 'we': (Z,24)} mean per zone x hod x daytype over validation hours."""
    out = {}
    for k, name in enumerate(DT):
        m = np.zeros((Z, 24))
        for hh in range(24):
            idx = cell_hours(k, hh)
            v = arr_TZ[idx].astype(float)
            m[:, hh] = np.nanmean(v, axis=0) if nan else v.mean(axis=0)
        out[name] = m
    return out


# ----------------------------------------------------------------------------- 5. demand layers
panel = common.load_panel()
dep_obs = panel.pivot(index="ts", columns="zone", values="dep").reindex(index=sim.hours, columns=range(Z)).values
arr_obs = panel.pivot(index="ts", columns="zone", values="arr").reindex(index=sim.hours, columns=range(Z)).values
layers = {}
layers["dep"] = mean_layer(dep_obs)
layers["arr"] = mean_layer(arr_obs)
layers["dep_gru"] = mean_layer(gru[..., 0], nan=True)
layers["arr_gru"] = mean_layer(gru[..., 1], nan=True)
for k in ("dep", "arr", "dep_gru", "arr_gru"):
    assert all(np.isfinite(layers[k][d]).all() for d in DT), k
print("demand layers done; total dep/week (obs):",
      round(sum(layers["dep"][d].sum() * n_days[d] for d in DT) / 4, 1))


# ----------------------------------------------------------------------------- 6. unmet-demand runs
def accumulate(rows, with_count=False):
    """rows: list of (h, vec(Z)). Returns {'wd','we'} -> (Z,24) of sum / days (or / count)."""
    S = {d: np.zeros((Z, 24)) for d in DT}
    C = {d: np.zeros((Z, 24)) for d in DT}
    for h, v in rows:
        d = DT[dt_of[h]]
        S[d][:, hod_of[h]] += v
        C[d][:, hod_of[h]] += 1
    if with_count:
        return {d: S[d] / np.maximum(C[d], 1) for d in DT}
    return {d: S[d] / n_days[d] for d in DT}


class LoggingMPC(MPCController):
    """MPC that records the hour-1 plan slacks (shortage, overflow) at each decision."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.slack_log = []

    def targets(self, sim, bikes, h):
        out = self.plan(sim, bikes, h)
        if out.get("target") is not None:
            self.slack_log.append((h, out["shortage"].copy(), out["overflow"].copy()))
        return out["target"], sim.budget


def run_ctrl(label, ctrl, budget):
    sim.budget = budget
    t0 = time.time()
    lost_rows, weekly = [], []
    for s in starts:
        tot, df = run_episode(sim, ctrl, s, log=True)
        weekly.append(tot["lost"])
        for h, lz in zip(df["h"].values, df["lost_zone"].values):
            lost_rows.append((int(h), np.asarray(lz, dtype=float)))
    lost = accumulate(lost_rows)
    wk = float(np.mean(weekly))
    print(f"[{time.time() - T_START:6.1f}s] {label:14s} budget {budget:2d}  lost/week {wk:8.2f}  weeks {weekly}  ({time.time() - t0:.1f}s)", flush=True)
    return lost, wk, weekly


weekly = {}
weekly_each = {}
totals4 = {}                                      # layer name -> 4-week total lost, for the consistency check
none_lost, wk, we_ = run_ctrl("none", NoRebalance(), 40)
weekly["none"] = wk; weekly_each["none"] = we_
for b in BUDGETS:
    layers[f"lost_none_b{b}"] = none_lost
    totals4[f"lost_none_b{b}"] = sum(we_)

for b in BUDGETS:
    lost, wk, we_ = run_ctrl("heuristic", HourlyOcc(), b)
    layers[f"lost_heur_b{b}"] = lost; weekly[f"heur_b{b}"] = wk; weekly_each[f"heur_b{b}"] = we_
    totals4[f"lost_heur_b{b}"] = sum(we_)

    mpc = LoggingMPC(gru, profile, horizon=HORIZON)
    lost, wk, we_ = run_ctrl("MPC(GRU)", mpc, b)
    layers[f"lost_mpc_b{b}"] = lost; weekly[f"mpc_b{b}"] = wk; weekly_each[f"mpc_b{b}"] = we_
    totals4[f"lost_mpc_b{b}"] = sum(we_)
    assert len(mpc.slack_log) == 4 * 168, len(mpc.slack_log)
    layers[f"shortage_b{b}"] = accumulate([(h, s) for h, s, o in mpc.slack_log], with_count=True)
    layers[f"overflow_b{b}"] = accumulate([(h, o) for h, s, o in mpc.slack_log], with_count=True)

    orc = MPCController(actual, profile, horizon=HORIZON)
    lost, wk, we_ = run_ctrl("MPC(oracle)", orc, b)
    layers[f"lost_oracle_b{b}"] = lost; weekly[f"oracle_b{b}"] = wk; weekly_each[f"oracle_b{b}"] = we_
    totals4[f"lost_oracle_b{b}"] = sum(we_)

    layers[f"oracle_gain_b{b}"] = {d: layers[f"lost_mpc_b{b}"][d] - layers[f"lost_oracle_b{b}"][d] for d in DT}

# ----------------------------------------------------------------------------- 8. spot checks
print("\n== spot checks ==")
checks = [("mpc_b40", 115.25, 0.01), ("heur_b40", 1016.25, 0.01), ("oracle_b40", 33.25, 0.01), ("none", 6124.0, 1.0)]
for k, expected, tol in checks:
    print(f"  weekly[{k}] = {weekly[k]:.2f}  (expected {expected})")
    assert abs(weekly[k] - expected) <= tol, (k, weekly[k], expected)
assert sorted(weekly_each["mpc_b40"]) == sorted([156, 161, 35, 109]), weekly_each["mpc_b40"]
print("  MPC b40 weekly lost per week:", weekly_each["mpc_b40"])
for name, tot in totals4.items():
    lay = layers[name]
    s = sum(lay[d].sum() * n_days[d] for d in DT)
    print(f"  layer {name:16s} weighted sum {s:9.2f}  vs run total {tot}")
    assert abs(s - tot) <= 1.0, (name, s, tot)
for b in BUDGETS:
    s = sum(layers[f"oracle_gain_b{b}"][d].sum() * n_days[d] for d in DT)
    assert abs(s - (totals4[f"lost_mpc_b{b}"] - totals4[f"lost_oracle_b{b}"])) <= 1.0

# ----------------------------------------------------------------------------- 7. curve
rm = pd.read_csv(V2 / "runs" / "regime_map_val.csv")
mean_lost = rm.groupby(["budget", "forecast", "controller"])["lost"].mean()
curve_budgets = sorted(int(b) for b in rm["budget"].unique())


def series(ctrl, fc):
    return [round(float(mean_lost.loc[(b, fc, ctrl)]), 3) for b in curve_budgets]


curve = {
    "budgets": curve_budgets,
    "series": {
        "heuristic": series("HourlyOcc", "gru"),
        "lp_gru": series("LP", "gru"),
        "mpc_gru": series("MPC", "gru"),
        "mpc_oracle": series("MPC", "oracle1h"),
    },
}
g = rm[(rm.budget == 40) & (rm.controller == "MPC") & (rm.forecast == "gru")]
curve["trips_per_week"] = round(float((g["served"] + g["lost"]).mean()), 3)
print("curve:", json.dumps(curve))
i40 = curve_budgets.index(40)
assert abs(curve["series"]["mpc_gru"][i40] - weekly["mpc_b40"]) < 0.01
assert abs(curve["series"]["heuristic"][i40] - weekly["heur_b40"]) < 0.01

# ----------------------------------------------------------------------------- 3. zone polygons
zdf = pd.read_csv(common.OUT_DIR / "zones.csv").sort_values("zone").reset_index(drop=True)
assert len(zdf) == 30 and list(zdf["zone"]) == list(range(30))
lat0, lon0 = zdf["lat"].mean(), zdf["lon"].mean()
kx = np.cos(np.radians(lat0)) * 111320.0
ky = 110540.0
px = (zdf["lon"].values - lon0) * kx
py = (zdf["lat"].values - lat0) * ky
pts = [Point(x, y) for x, y in zip(px, py)]
mp = MultiPoint(pts)
env = box(px.min() - 50000, py.min() - 50000, px.max() + 50000, py.max() + 50000)
cells = list(voronoi_polygons(mp, extend_to=env).geoms)
clip = mp.convex_hull.buffer(700.0)
feats = []
for i, p in enumerate(pts):
    cell = [c for c in cells if c.contains(p)]
    assert len(cell) == 1, (i, len(cell))
    poly = cell[0].intersection(clip)
    assert poly.geom_type == "Polygon" and poly.contains(p), (i, poly.geom_type)
    ring = [[round(float(x / kx + lon0), 6), round(float(y / ky + lat0), 6)] for x, y in poly.exterior.coords]
    feats.append({
        "type": "Feature",
        "properties": {"zone": int(zdf.zone[i]), "cap": int(round(zdf.cap[i])), "n_stations": int(zdf.n_stations[i]),
                       "lat": round(float(zdf.lat[i]), 6), "lon": round(float(zdf.lon[i]), 6)},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    })
zones_fc = {"type": "FeatureCollection", "features": feats}
assert len(feats) == 30
print("zone polygons ok; mean vertices:", np.mean([len(f["geometry"]["coordinates"][0]) for f in feats]).round(1))

# ----------------------------------------------------------------------------- 9. write
layer_order = (["dep", "arr", "dep_gru", "arr_gru"]
               + [f"{n}_b{b}" for b in BUDGETS for n in ("lost_none", "lost_heur", "lost_mpc", "lost_oracle", "oracle_gain", "shortage", "overflow")])
layers_json = {k: {d: np.round(layers[k][d], 3).tolist() for d in DT} for k in layer_order}

first = sim.hours[starts[0]]
last = sim.hours[starts[-1] + 167]
meta = {
    "title": "Oslo bike-share: demand and rebalancing (prototype)",
    "period": f"4 validation weeks, Mon {first.date()} to Sun {last.date()}",
    "n_zones": 30,
    "budgets_run": BUDGETS,
    "horizon": HORIZON,
    "built": datetime.datetime.now().replace(microsecond=0).isoformat(),
}
map_data = {"meta": meta, "zones": zones_fc, "daytypes": DT, "layers": layers_json,
            "weekly": {k: round(float(v), 3) for k, v in weekly.items()}, "curve": curve}
with open(OUT / "map_data.json", "w", encoding="utf-8") as f:
    json.dump(map_data, f, separators=(",", ":"))

gj_feats = []
for zi, feat in enumerate(feats):
    props = dict(feat["properties"])
    for k in layer_order:
        for d in DT:
            for hh in range(24):
                props[f"{k}_{d}_{hh:02d}"] = round(float(layers[k][d][zi, hh]), 3)
    gj_feats.append({"type": "Feature", "properties": props, "geometry": feat["geometry"]})
with open(OUT / "layers.geojson", "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": gj_feats}, f, separators=(",", ":"))

print("\nweekly:", json.dumps(map_data["weekly"]))
for fn in ("map_data.json", "layers.geojson"):
    print(f"{fn}: {(HERE / fn).stat().st_size / 1024:.1f} KB")
print(f"total wall time {time.time() - T_START:.1f}s")
