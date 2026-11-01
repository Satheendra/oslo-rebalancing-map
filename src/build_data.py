"""Build map_data.json and layers.geojson for the Oslo rebalancing map (v3 numbers, schema version 2, see data/SCHEMA.md).

Reads from the project's v3 folder (code/, data_out/, runs/; read-only) and the raw station list. Writes only into
this repo (data/ and .cache/). Run from the repo folder:   python src\\build_data.py   [--fresh] [--periods val,test]

Environment: OSLO_PROJECT_DIR (default: the v3 folder on this machine), OSLO_STATION_MASTER (raw station list),
ORACLE_PROFILE = actual | profile  (what the oracle MPC uses for plan hours 2..H; the paper uses 'actual').
Finished controller runs are cached in .cache/ (so a failed spot check does not cost another 15 minutes); --fresh ignores it.
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import sys, time, json, datetime, pickle
from pathlib import Path

sys.dont_write_bytecode = True                      # never write __pycache__ next to the (read-only) project code
HERE = Path(__file__).resolve().parent
REPO = HERE.parent if HERE.name == "src" else HERE
OUT = REPO / "data" if (REPO / "data").is_dir() else REPO
CACHE_DIR = REPO / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))   # numba cache must not go into the project folder

V3 = Path(os.environ.get(
    "OSLO_PROJECT_DIR",
    r"C:\Users\srl\OneDrive - Högskolan Dalarna\Documents\PhD\RL Work\project\v3"))
STATION_MASTER = Path(os.environ.get(
    "OSLO_STATION_MASTER",
    r"C:\Users\srl\OneDrive - Högskolan Dalarna\Documents\PhD\MSc Theses 2026\T4 data\oslo_station_master.csv"))
sys.path.insert(0, str(V3 / "code"))

import numpy as np
import pandas as pd
import shapely
from shapely import MultiPoint, Point, STRtree, box, voronoi_polygons
from shapely.geometry import mapping
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import common
from common import N_ZONES
from sim import HourlyOcc, NoRebalance, run_episode
from mpc import MPCController
import env_v3

T_START = time.time()
Z = N_ZONES
BUDGETS = [40, 20]
HORIZON = 6
DT = ["wd", "we"]
ORACLE_PROFILE = os.environ.get("ORACLE_PROFILE", "actual")
FRESH = "--fresh" in sys.argv
PERIODS_WANTED = ["val", "test"]
if "--periods" in sys.argv:
    PERIODS_WANTED = sys.argv[sys.argv.index("--periods") + 1].split(",")
PERIOD_LABELS = {"val": "Validation: 4 weeks, Sept 2025", "test": "Test: 21 weeks, May to Sept 2026"}
assert V3.name == "v3" and (V3 / "data_out").is_dir(), V3
assert Path(common.OUT_DIR).resolve() == (V3 / "data_out").resolve(), common.OUT_DIR   # code really reads v3 data

print(f"project dir: {V3}\nrepo: {REPO}  out: {OUT}\nshapely {shapely.__version__}, numpy {np.__version__}", flush=True)


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- 1. sim, forecasts
sim = env_v3.base_sim().clone()
hod_of = sim.hod
dt_of = np.where(sim.dow < 5, 0, 1)                  # 0 = wd, 1 = we
gru = env_v3.get_forecast("gru_mean")                # (T, Z, 2), the rung the paper's MPC used
profile = env_v3.get_profile()                       # naive seed 0
panel = common.load_panel()
dep_obs = panel.pivot(index="ts", columns="zone", values="dep").reindex(index=sim.hours, columns=range(Z)).values
arr_obs = panel.pivot(index="ts", columns="zone", values="arr").reindex(index=sim.hours, columns=range(Z)).values
actual = env_v3.get_actual()                         # (T, Z, 2) realised dep, arr (NaN -> 0)
hist_occ = np.load(common.OUT_DIR / "hist_occ.npy")  # v3 hist_occ (also the default inside HourlyOcc / MPCController)


# ----------------------------------------------------------------------------- 2. periods
periods = {}
for pname in PERIODS_WANTED:
    starts = env_v3.episode_starts(pname)
    hrs = np.unique(np.concatenate([np.arange(s, s + 168) for s in starts]))
    assert len(hrs) == 168 * len(starts), (pname, "overlapping episode windows")
    n_days = {}
    for k, d in enumerate(DT):
        n_hours = int((dt_of[hrs] == k).sum())
        assert n_hours % 24 == 0
        n_days[d] = n_hours // 24
    periods[pname] = dict(starts=starts, hours=hrs, n_days=n_days)
    log(f"period {pname}: {len(starts)} weeks, days {n_days}, first {sim.hours[starts[0]]}, last start {sim.hours[starts[-1]]}, "
        f"last hour {sim.hours[starts[-1] + 167]}")
if "val" in periods:
    assert len(periods["val"]["starts"]) == 4 and periods["val"]["n_days"] == {"wd": 20, "we": 8}, periods["val"]
if "test" in periods:
    assert len(periods["test"]["starts"]) == 21 and periods["test"]["n_days"] == {"wd": 105, "we": 42}, periods["test"]


def mean_layer(arr_TZ, P, nan=False):
    """(T, Z) -> {'wd': (Z,24), 'we': (Z,24)}: mean per zone x hour of day x day type over the period's hours."""
    out = {}
    for k, d in enumerate(DT):
        m = np.zeros((Z, 24))
        for hh in range(24):
            idx = P["hours"][(dt_of[P["hours"]] == k) & (hod_of[P["hours"]] == hh)]
            v = arr_TZ[idx].astype(float)
            m[:, hh] = np.nanmean(v, axis=0) if nan else v.mean(axis=0)
        out[d] = m
    return out


def accumulate(rows, P, with_count=False):
    """rows: list of (h, vec(Z)). Returns {'wd','we'} -> (Z,24) of sum / days of that day type (or / number of decisions)."""
    S = {d: np.zeros((Z, 24)) for d in DT}
    C = {d: np.zeros((Z, 24)) for d in DT}
    for h, v in rows:
        d = DT[dt_of[h]]
        S[d][:, hod_of[h]] += v
        C[d][:, hod_of[h]] += 1
    if with_count:
        return {d: S[d] / np.maximum(C[d], 1) for d in DT}
    return {d: S[d] / P["n_days"][d] for d in DT}


class LoggingMPC(MPCController):
    """MPC that records the hour-1 plan slacks (shortage, overflow) at each decision. Same decisions as MPCController."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.slack_log = []

    def targets(self, sim, bikes, h):
        out = self.plan(sim, bikes, h)
        if out.get("target") is not None:
            self.slack_log.append((h, out["shortage"].copy(), out["overflow"].copy()))
        return out["target"], sim.budget


# ----------------------------------------------------------------------------- 3. demand layers
layers = {p: {} for p in periods}
for p, P in periods.items():
    L = layers[p]
    for name, arr in (("dep", dep_obs), ("arr", arr_obs)):
        assert np.isfinite(arr[P["hours"]]).all(), (p, name)
        L[name] = mean_layer(arr, P)
    for name, j in (("dep_gru", 0), ("arr_gru", 1)):
        assert np.isfinite(gru[P["hours"]][..., j]).all(), (p, name, "GRU forecast has NaN inside the period")
        L[name] = mean_layer(gru[..., j], P)
    P["trips_per_week"] = float(dep_obs[P["hours"]].sum() / len(P["starts"]))
    log(f"demand layers {p}: mean observed departures/week {P['trips_per_week']:.2f}")


# ----------------------------------------------------------------------------- 4. unmet-demand runs
def cache_path(p):
    return CACHE_DIR / f"runs_{p}_oracle-{ORACLE_PROFILE}.pkl"


def run_ctrl(p, label, ctrl, budget, store):
    P = periods[p]
    key = (label, budget)
    if key in store:
        log(f"{p:4s} {label:12s} b{budget}: from cache  lost/week {store[key]['wk']:.2f}")
        return store[key]
    sim.budget = budget
    t0 = time.time()
    lost_rows, weekly_each = [], []
    for wi, s in enumerate(P["starts"]):
        t1 = time.time()
        tot, df = run_episode(sim, ctrl, s, log=True)
        weekly_each.append(int(tot["lost"]))
        for h, lz in zip(df["h"].values, df["lost_zone"].values):
            lost_rows.append((int(h), np.asarray(lz, dtype=float)))
        if len(P["starts"]) > 4:
            print(f"      week {wi:2d}  lost {tot['lost']:5d}  served {tot['served']:6d}  ({time.time() - t1:.1f}s)", flush=True)
    res = dict(lost=accumulate(lost_rows, P), wk=float(np.mean(weekly_each)), weekly_each=weekly_each,
               served_each=None, wall=time.time() - t0)
    if isinstance(ctrl, LoggingMPC):
        assert len(ctrl.slack_log) == 168 * len(P["starts"]), len(ctrl.slack_log)
        res["shortage"] = accumulate([(h, s_) for h, s_, o_ in ctrl.slack_log], P, with_count=True)
        res["overflow"] = accumulate([(h, o_) for h, s_, o_ in ctrl.slack_log], P, with_count=True)
        res["mpc_fail"] = int(sum(1 for t in ctrl.solve_times) - len(ctrl.slack_log))
    log(f"{p:4s} {label:12s} b{budget}: lost/week {res['wk']:9.2f}  wall {res['wall']:6.1f}s  weeks {weekly_each}")
    store[key] = res
    with open(cache_path(p), "wb") as f:
        pickle.dump(store, f)
    return res


weekly = {p: {} for p in periods}
weekly_each = {p: {} for p in periods}
totals = {p: {} for p in periods}
for p, P in periods.items():
    store = {}
    if cache_path(p).exists() and not FRESH:
        store = pickle.load(open(cache_path(p), "rb"))
    L = layers[p]
    r = run_ctrl(p, "none", NoRebalance(), 40, store)
    weekly[p]["none"] = r["wk"]; weekly_each[p]["none"] = r["weekly_each"]
    for b in BUDGETS:
        L[f"lost_none_b{b}"] = r["lost"]
        totals[p][f"lost_none_b{b}"] = sum(r["weekly_each"])
    oracle_prof = actual if ORACLE_PROFILE == "actual" else profile
    for b in BUDGETS:
        r = run_ctrl(p, "heur", HourlyOcc(), b, store)
        L[f"lost_heur_b{b}"] = r["lost"]; weekly[p][f"heur_b{b}"] = r["wk"]; weekly_each[p][f"heur_b{b}"] = r["weekly_each"]
        totals[p][f"lost_heur_b{b}"] = sum(r["weekly_each"])

        r = run_ctrl(p, "mpc", LoggingMPC(gru, profile, horizon=HORIZON), b, store)
        L[f"lost_mpc_b{b}"] = r["lost"]; weekly[p][f"mpc_b{b}"] = r["wk"]; weekly_each[p][f"mpc_b{b}"] = r["weekly_each"]
        totals[p][f"lost_mpc_b{b}"] = sum(r["weekly_each"])
        L[f"shortage_b{b}"] = r["shortage"]; L[f"overflow_b{b}"] = r["overflow"]
        assert r["mpc_fail"] == 0, f"{r['mpc_fail']} LP failures"

        r = run_ctrl(p, "oracle", MPCController(actual, oracle_prof, horizon=HORIZON), b, store)
        L[f"lost_oracle_b{b}"] = r["lost"]; weekly[p][f"oracle_b{b}"] = r["wk"]; weekly_each[p][f"oracle_b{b}"] = r["weekly_each"]
        totals[p][f"lost_oracle_b{b}"] = sum(r["weekly_each"])
        L[f"oracle_gain_b{b}"] = {d: L[f"lost_mpc_b{b}"][d] - L[f"lost_oracle_b{b}"][d] for d in DT}

# ----------------------------------------------------------------------------- 5. spot checks
print("\n== spot checks (tolerance 0.5 trips) ==", flush=True)
check_rows = []


def check(period, what, expected, got, tol=0.5):
    ok = abs(expected - got) <= tol
    check_rows.append((period, what, expected, got, ok))
    print(f"  {'OK  ' if ok else 'FAIL'} {period:4s} {what:34s} expected {expected:10.3f}  got {got:10.3f}  diff {got - expected:+8.3f}")


val_ref = pd.read_csv(V3 / "runs" / "val_references.csv")
test_ev = pd.read_csv(V3 / "runs" / "test_eval_v3.csv")
if "val" in periods:
    for k, e in (("none", 6112.5), ("heur_b40", 1008.0), ("mpc_b40", 121.25), ("mpc_b20", 1621.75)):
        check("val", f"hard-coded weekly[{k}]", e, weekly["val"][k])
    assert weekly_each["val"]["mpc_b40"] == [118, 176, 62, 129], weekly_each["val"]["mpc_b40"]
    assert weekly_each["val"]["mpc_b20"] == [1894, 1836, 1104, 1653], weekly_each["val"]["mpc_b20"]
    for ctrl, key in (("NO_REBALANCE", "none"), ("HEURISTIC", "heur"), ("FULL_MPC", "mpc")):
        for b in BUDGETS:
            k = "none" if key == "none" else f"{key}_b{b}"
            g = val_ref[(val_ref.controller == ctrl) & (val_ref.budget == b)].sort_values("week")["lost"].values
            check("val", f"val_references {ctrl} b{b} mean", float(g.mean()), weekly["val"][k])
            for wi, (a, c) in enumerate(zip(g, weekly_each["val"][k])):
                check_rows.append(("val", f"{ctrl} b{b} week {wi}", float(a), float(c), abs(a - c) <= 0.5))
            if not all(abs(a - c) <= 0.5 for a, c in zip(g, weekly_each["val"][k])):
                print(f"  per-week MISMATCH val {ctrl} b{b}: csv {list(g)} got {weekly_each['val'][k]}")
if "test" in periods:
    for ctrl, key in (("NO_REBALANCE", "none"), ("HEURISTIC", "heur"), ("FULL_MPC", "mpc"), ("ORACLE_MPC", "oracle")):
        for b in BUDGETS:
            k = "none" if key == "none" else f"{key}_b{b}"
            g = test_ev[(test_ev.controller == ctrl) & (test_ev.budget == b)].sort_values("week")
            assert len(g) == 21 and list(g["start_h"]) == periods["test"]["starts"], (ctrl, b)
            check("test", f"test_eval_v3 {ctrl} b{b} mean", float(g["lost"].mean()), weekly["test"][k])
            for wi, (a, c) in enumerate(zip(g["lost"].values, weekly_each["test"][k])):
                check_rows.append(("test", f"{ctrl} b{b} week {wi}", float(a), float(c), abs(a - c) <= 0.5))
            bad = [(wi, int(a), c) for wi, (a, c) in enumerate(zip(g["lost"].values, weekly_each["test"][k])) if abs(a - c) > 0.5]
            if bad:
                print(f"  per-week MISMATCH test {ctrl} b{b} (week, csv, got): {bad}")
for p in periods:
    for name, tot in totals[p].items():
        lay = layers[p][name]
        s = sum(lay[d].sum() * periods[p]["n_days"][d] for d in DT)
        ok = abs(s - tot) <= 0.5
        check_rows.append((p, f"layer sum {name}", float(tot), float(s), ok))
        print(f"  {'OK  ' if ok else 'FAIL'} {p:4s} layer {name:16s} weighted sum {s:10.2f}  vs run total {tot}")
    for b in BUDGETS:
        s = sum(layers[p][f"oracle_gain_b{b}"][d].sum() * periods[p]["n_days"][d] for d in DT)
        e = totals[p][f"lost_mpc_b{b}"] - totals[p][f"lost_oracle_b{b}"]
        check_rows.append((p, f"layer sum oracle_gain_b{b}", float(e), float(s), abs(s - e) <= 0.5))
n_fail = sum(1 for r in check_rows if not r[4])
print(f"spot checks: {len(check_rows) - n_fail} passed, {n_fail} failed", flush=True)
if n_fail:
    print("FAILED checks:", [r[:4] for r in check_rows if not r[4]])
    sys.exit(f"{n_fail} spot checks failed; not writing outputs")
if set(periods) != {"val", "test"}:
    sys.exit("debug run (not both periods): checks passed, outputs not written")

# ----------------------------------------------------------------------------- 6. zone polygons, stations, edges
zdf = pd.read_csv(common.OUT_DIR / "zones.csv").sort_values("zone").reset_index(drop=True)
assert len(zdf) == 30 and list(zdf["zone"]) == list(range(30))
sm = pd.read_csv(STATION_MASTER, encoding="utf-8")
sz = pd.read_csv(common.OUT_DIR / "station_zone.csv")
st = sz.merge(sm[["station_id", "name", "lat", "lon", "capacity", "slots"]], on="station_id", how="left", validate="1:1")
assert len(st) == 276 and st[["lat", "lon", "name"]].notna().all().all()
n_cap_fallback = int(st["capacity"].isna().sum())
st["cap_final"] = st["capacity"].fillna(st["slots"])
n_cap_missing = int(st["cap_final"].isna().sum())
st["cap_final"] = st["cap_final"].fillna(st["cap"])                 # last resort: cap column of station_zone.csv
print(f"stations: {len(st)}; capacity empty -> used slots: {n_cap_fallback - n_cap_missing}; "
      f"empty in both (used station_zone cap): {n_cap_missing}")
assert (st["cap_final"] > 0).all()
nst = st.groupby("zone").size().reindex(range(30)).values
assert (nst == zdf["n_stations"].values).all(), "station count per zone differs from zones.csv"

lat0, lon0 = st["lat"].mean(), st["lon"].mean()
kx = np.cos(np.radians(lat0)) * 111320.0
ky = 110540.0
px = (st["lon"].values - lon0) * kx
py = (st["lat"].values - lat0) * ky
pts = [Point(x, y) for x, y in zip(px, py)]
mp = MultiPoint(pts)
n_unique = len({(round(x, 3), round(y, 3)) for x, y in zip(px, py)})
env = box(px.min() - 50000, py.min() - 50000, px.max() + 50000, py.max() + 50000)
cells = list(voronoi_polygons(mp, extend_to=env).geoms)
print(f"voronoi: {len(cells)} cells for {len(pts)} stations ({n_unique} distinct positions)")
clip = mp.convex_hull.buffer(400.0)
tree = STRtree(cells)
zone_cells = {z: [] for z in range(30)}
for i, p_ in enumerate(pts):
    hit = tree.query(p_, predicate="intersects")
    assert len(hit) >= 1, i
    zone_cells[int(st["zone"].iloc[i])].append(cells[int(hit[0])].intersection(clip))


def to_lonlat(geom):
    return shapely.transform(geom, lambda c: np.column_stack([c[:, 0] / kx + lon0, c[:, 1] / ky + lat0]))


def round6(geom):
    return shapely.transform(geom, lambda c: np.round(c, 6))


feats, zone_geoms = [], []
for z in range(30):
    u = unary_union(zone_cells[z]).buffer(0)
    assert u.is_valid and u.geom_type in ("Polygon", "MultiPolygon") and not u.is_empty, (z, u.geom_type)
    g = round6(to_lonlat(u))
    g = g if g.is_valid else g.buffer(0)
    assert g.is_valid, z
    if g.geom_type == "Polygon":
        g = orient(g, 1.0)
    else:
        g = shapely.MultiPolygon([orient(q, 1.0) for q in g.geoms])
    zone_geoms.append(g)
    feats.append({
        "type": "Feature",
        "properties": {"zone": int(zdf.zone[z]), "cap": int(round(zdf.cap[z])), "n_stations": int(zdf.n_stations[z]),
                       "lat": round(float(zdf.lat[z]), 6), "lon": round(float(zdf.lon[z]), 6)},
        "geometry": mapping(g),
    })
assert len(feats) == 30
# stations vs their zone polygon (lon/lat, after rounding)
outside = []
for i in range(len(st)):
    pt = Point(float(st["lon"].iloc[i]), float(st["lat"].iloc[i]))
    if not zone_geoms[int(st["zone"].iloc[i])].contains(pt):
        outside.append((int(st["station_id"].iloc[i]), int(st["zone"].iloc[i])))
n_multi = sum(1 for g in zone_geoms if g.geom_type == "MultiPolygon")
n_holes = sum(len(q.interiors) for g in zone_geoms for q in (g.geoms if g.geom_type == "MultiPolygon" else [g]))
tot_area = sum(g.area for g in zone_geoms)
print(f"zone polygons: 30 features ({n_multi} MultiPolygon, {n_holes} interior rings); stations outside own zone polygon: "
      f"{len(outside)} {outside[:10]}")
assert len(outside) <= 0.05 * len(st), outside      # "contains most of its stations"
zones_fc = {"type": "FeatureCollection", "features": feats}

stations = [{"id": int(r.station_id), "name": str(r.name), "lat": round(float(r.lat), 6), "lon": round(float(r.lon), 6),
             "zone": int(r.zone), "cap": int(round(r.cap_final))} for r in st.itertuples()]
assert len(stations) == 276

ei = np.load(common.OUT_DIR / "edge_index.npy")
pairs = sorted({(int(min(a, b)), int(max(a, b))) for a, b in zip(ei[0], ei[1]) if a != b})
edges = [[a, b] for a, b in pairs]
print(f"edges: edge_index columns {ei.shape[1]} -> {len(edges)} undirected pairs (summary.json says "
      f"{json.load(open(common.OUT_DIR / 'summary.json'))['edges']})")

# ----------------------------------------------------------------------------- 7. ladder
ta = pd.read_csv(V3 / "runs" / "test_eval_v3_table_a.csv").set_index("method")


def row_lost(method):
    return [float(ta.loc[method, "lost_B20"]), float(ta.loc[method, "lost_B40"])]


def ppo_mean(arch):
    rows = [f"PPO-{arch} s{s} [argmax]" for s in range(3)]
    assert all(r in ta.index for r in rows), rows
    v = np.array([row_lost(r) for r in rows])
    return [float(x) for x in v.mean(axis=0)]


def r3(v):
    return [round(float(x), 1) for x in v]


ladder_test = [
    {"key": "none", "label": "No rebalancing", "kind": "reference", "lost": r3(row_lost("NO_REBALANCE"))},
    {"key": "heur", "label": "Hourly occupancy rule", "kind": "reference", "lost": r3(row_lost("HEURISTIC"))},
    {"key": "mpc", "label": "Full MPC", "kind": "reference", "lost": r3(row_lost("FULL_MPC"))},
    {"key": "ppo_mlp", "label": "PPO, MLP policy", "kind": "learned", "lost": r3(ppo_mean("MLP"))},
    {"key": "ppo_gnn", "label": "PPO, GNN policy", "kind": "learned", "lost": r3(ppo_mean("GNN"))},
    {"key": "oracle", "label": "MPC, perfect forecast", "kind": "ceiling", "lost": r3(row_lost("ORACLE_MPC"))},
]
vr = val_ref.groupby(["controller", "budget"])["lost"].mean()
val_rows = [("none", "No rebalancing", "NO_REBALANCE"), ("heur", "Hourly occupancy rule", "HEURISTIC"), ("mpc", "Full MPC", "FULL_MPC")]
ladder_val = [{"key": k, "label": lab, "kind": "reference", "lost": r3([vr.loc[(c, 20)], vr.loc[(c, 40)]])} for k, lab, c in val_rows]
if "ORACLE_MPC" in set(val_ref.controller):
    ladder_val.append({"key": "oracle", "label": "MPC, perfect forecast", "kind": "ceiling",
                       "lost": r3([vr.loc[("ORACLE_MPC", 20)], vr.loc[("ORACLE_MPC", 40)]])})
ladder = {"budgets": [20, 40], "test": ladder_test, "val": ladder_val}
for row in ladder_test:
    print("  ladder test", row["key"], row["lost"])
# ladder vs own runs (reference rows)
for key, k20, k40 in (("none", "none", "none"), ("heur", "heur_b20", "heur_b40"), ("mpc", "mpc_b20", "mpc_b40"), ("oracle", "oracle_b20", "oracle_b40")):
    row = next(r for r in ladder_test if r["key"] == key)
    assert abs(row["lost"][0] - weekly["test"][k20]) < 0.06 and abs(row["lost"][1] - weekly["test"][k40]) < 0.06, (key, row, weekly["test"][k20], weekly["test"][k40])

# ----------------------------------------------------------------------------- 8. write
layer_order = (["dep", "arr", "dep_gru", "arr_gru"]
               + [f"{n}_b{b}" for b in BUDGETS for n in ("lost_none", "lost_heur", "lost_mpc", "lost_oracle", "oracle_gain", "shortage", "overflow")])
layers_json = {p: {k: {d: np.round(layers[p][k][d], 3).tolist() for d in DT} for k in layer_order} for p in ("val", "test")}

meta = {
    "title": "Oslo bike-share: demand and rebalancing",
    "source": "v3 simulator and MPC (the submitted study)",
    "periods": {p: {"label": PERIOD_LABELS[p], "n_weeks": len(periods[p]["starts"]), "days": periods[p]["n_days"]} for p in ("val", "test")},
    "budgets_run": BUDGETS,
    "horizon": HORIZON,
    "built": datetime.datetime.now().replace(microsecond=0).isoformat(),
}
map_data = {
    "meta": meta, "zones": zones_fc, "stations": stations, "edges": edges, "daytypes": DT, "layers": layers_json,
    "weekly": {p: {k: round(float(v), 3) for k, v in weekly[p].items()} for p in ("val", "test")},
    "trips_per_week": {p: round(periods[p]["trips_per_week"], 3) for p in ("val", "test")},
    "ladder": ladder,
}
with open(OUT / "map_data.json", "w", encoding="utf-8") as f:
    json.dump(map_data, f, separators=(",", ":"), ensure_ascii=False)

gj_feats = []
for zi, feat in enumerate(feats):
    props = dict(feat["properties"])
    for p in ("val", "test"):
        for k in layer_order:
            for d in DT:
                for hh in range(24):
                    props[f"{p}_{k}_{d}_{hh:02d}"] = round(float(layers[p][k][d][zi, hh]), 3)
    gj_feats.append({"type": "Feature", "properties": props, "geometry": feat["geometry"]})
with open(OUT / "layers.geojson", "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": gj_feats}, f, separators=(",", ":"), ensure_ascii=False)

print("\nweekly:", json.dumps(map_data["weekly"]))
print("trips_per_week:", json.dumps(map_data["trips_per_week"]))
for fn in ("map_data.json", "layers.geojson"):
    print(f"{fn}: {(OUT / fn).stat().st_size / 1024:.1f} KB")
print(f"total wall time {time.time() - T_START:.1f}s")
