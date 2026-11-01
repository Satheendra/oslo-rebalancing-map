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
PPO_ARCHS = ["gnn", "mlp"]                          # --ppo-archs gnn   (GNN only)   |   --ppo-archs gnn,mlp   (default)
if "--ppo-archs" in sys.argv:
    PPO_ARCHS = [a for a in sys.argv[sys.argv.index("--ppo-archs") + 1].split(",") if a]
assert set(PPO_ARCHS) <= {"gnn", "mlp"} and (not PPO_ARCHS or PPO_ARCHS[0] == "gnn"), PPO_ARCHS
PPO_SEEDS = [0, 1, 2]
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
bikes_obs = panel.pivot(index="ts", columns="zone", values="bikes").reindex(index=sim.hours, columns=range(Z)).values.astype(float)
cap_zone = panel.groupby("zone")["cap"].first().reindex(range(Z)).values.astype(float)   # constant per zone in the panel
assert (panel.groupby("zone")["cap"].nunique() == 1).all() and (cap_zone > 0).all()
actual = env_v3.get_actual()                        # (T, Z, 2) realised dep, arr (NaN -> 0)
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
    # dock-level layers (version 2.1): dep / cap x 100 per zone-hour, and fill = bikes / cap (panel), same aggregation as dep
    assert np.isfinite(bikes_obs[P["hours"]]).all(), (p, "bikes")
    L["dep_per_dock"] = mean_layer(dep_obs / cap_zone[None, :] * 100.0, P)
    L["fill"] = mean_layer(bikes_obs / cap_zone[None, :], P)
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

# ----------------------------------------------------------------------------- 4b. learned-policy (PPO) runs (version 2.1)
# Each (period, budget, arch, seed) is one rollout of the selected checkpoint (argmax) through env_v3.CandidateEnv over the
# period's weeks. Same construction as ppo_v3.evaluate_checkpoint / nb 34: CandidateEnv(split, budget_cycle=[b], forecast and
# blind_zones from the checkpoint args, seed 999), one env for all weeks, action = argmax of the masked actor logits.
# Per decision hour we log (before the env steps): the policy slot's own moves (cand["moves"][slot]: the part of the budget the
# policy controls; the first 0.75 B "common" moves are the same for every slot) and the CONTINUE slot's moves (the leftover of the
# full-MPC plan the policy could have continued). override and policy_net are built from those two lists:
#   policy_net[z]  = bikes into z minus bikes out of z in the slot's own moves
#   override[z]    = 1 if merged slot moves != merged CONTINUE moves AND (z is a source/destination of a slot move OR the net
#                    flow at z differs from the CONTINUE net flow; the second part catches a policy that cancels MPC moves)
_ppo_cache = {}
PPO_SEL = {}


def ppo_selected(arch, seed):
    if (arch, seed) not in PPO_SEL:
        sel = json.load(open(V3 / "runs" / f"selected_main_{arch}_s{seed}.json"))
        ck = V3 / "runs" / Path(sel["checkpoint"]).name
        assert ck.exists() and f"_u{sel['selected_update']}.pt" in ck.name, (arch, seed, ck)
        PPO_SEL[(arch, seed)] = (sel, ck)
    return PPO_SEL[(arch, seed)]


def _net_flow(moves):
    v = np.zeros(Z)
    for i, j, n in moves:
        v[j] += n
        v[i] -= n
    return v


def run_ppo(p, arch, seed, b):
    P = periods[p]
    sel, ck_path = ppo_selected(arch, seed)
    cp = CACHE_DIR / f"ppo_{p}_{arch}_s{seed}_b{b}.pkl"
    if cp.exists() and not FRESH:
        r = pickle.load(open(cp, "rb"))
        if r["ckpt"] == ck_path.name:
            log(f"{p:4s} PPO-{arch.upper()} s{seed} b{b}: from cache  lost/week {r['wk']:.2f}")
            return r
    import torch
    torch.set_num_threads(1)
    import ppo_v3 as PP
    from env_v3 import CONTINUE, merge_moves
    PP.load_env_impl("real")
    net, ck = PP.load_policy(ck_path)
    assert int(ck["update"]) == sel["selected_update"] and ck["arch"] == arch and ck["args"].get("seed") == seed
    fc, bz = ck["args"].get("forecast", "gru_mean"), ck["args"].get("blind_zones")
    assert fc == "gru_mean" and bz is None, (fc, bz)
    env = env_v3.CandidateEnv(p, budget_cycle=(b,), forecast=fc, seed=999, blind_zones=bz)
    assert list(env.starts) == list(P["starts"])
    t0 = time.time()
    lost_rows, ov_rows, pn_rows, weekly_each_, n_nonmpc, n_dec, n_cont_feas = [], [], [], [], 0, 0, 0
    for wi, s in enumerate(P["starts"]):
        t1 = time.time()
        obs, done, wlost = env.reset(s, b), False, 0
        while not done:
            bt = PP.to_batch([obs])
            with torch.no_grad():
                a = int(net.logits(bt).argmax(-1))
            c = env.cand
            pol = merge_moves(c["moves"][a])
            cont = merge_moves(c["moves"][CONTINUE])
            net_pol = _net_flow(pol)
            ov = np.zeros(Z)
            if pol != cont:
                ov[net_pol != _net_flow(cont)] = 1.0
                for i, j, n in pol:
                    ov[i] = ov[j] = 1.0
            n_cont_feas += int(bool(c["mask"][CONTINUE]))
            n_nonmpc += int(c["kind"][a] == 2)
            n_dec += 1
            h_dec = int(env.sim.h)
            obs, _, done, info = env.step(a)
            assert info["h"] == h_dec
            wlost += info["lost"]
            lost_rows.append((h_dec, np.asarray(info["lost_zone"], dtype=float)))
            ov_rows.append((h_dec, ov))
            pn_rows.append((h_dec, net_pol))
        assert wlost == env.totals()["lost"] == int(env.week["lost_zone"].sum()), (wlost, env.totals()["lost"])
        assert env.mpc_fail == 0
        weekly_each_.append(int(wlost))
        if len(P["starts"]) > 4 and (wi % 5 == 4 or wi == len(P["starts"]) - 1):
            print(f"      week {wi:2d}  lost {wlost:5d}  ({time.time() - t1:.1f}s)", flush=True)
    r = dict(lost=accumulate(lost_rows, P), override=accumulate(ov_rows, P, with_count=True),
             policy_net=accumulate(pn_rows, P, with_count=True), wk=float(np.mean(weekly_each_)), weekly_each=weekly_each_,
             wall=time.time() - t0, ckpt=ck_path.name, update=int(ck["update"]),
             share_transfer=n_nonmpc / n_dec, share_cont_feasible=n_cont_feas / n_dec,
             override_share_any=float(np.mean([v.max() for _, v in ov_rows])))
    log(f"{p:4s} PPO-{arch.upper()} s{seed} b{b} (u{r['update']}): lost/week {r['wk']:9.2f}  wall {r['wall']:6.1f}s  weeks {weekly_each_}")
    with open(cp, "wb") as f:
        pickle.dump(r, f)
    return r


ppo_runs = {p: {} for p in periods}                  # p -> (arch, b, seed) -> result
ppo_t0 = time.time()
for p, P in periods.items():
    for arch in PPO_ARCHS:
        for b in BUDGETS:
            for seed in PPO_SEEDS:
                ppo_runs[p][(arch, b, seed)] = run_ppo(p, arch, seed, b)
    L = layers[p]
    for arch in PPO_ARCHS:
        for b in BUDGETS:
            rs = [ppo_runs[p][(arch, b, s)] for s in PPO_SEEDS]
            L[f"lost_ppo_{arch}_b{b}"] = {d: np.mean([r["lost"][d] for r in rs], axis=0) for d in DT}
            weekly[p][f"ppo_{arch}_b{b}"] = float(np.mean([r["wk"] for r in rs]))
            weekly_each[p][f"ppo_{arch}_b{b}"] = list(np.mean([r["weekly_each"] for r in rs], axis=0))
            totals[p][f"lost_ppo_{arch}_b{b}"] = float(np.mean([sum(r["weekly_each"]) for r in rs]))
            if arch == "gnn":
                L[f"ppo_gain_b{b}"] = {d: L[f"lost_mpc_b{b}"][d] - L[f"lost_ppo_gnn_b{b}"][d] for d in DT}
                L[f"override_b{b}"] = {d: np.mean([r["override"][d] for r in rs], axis=0) for d in DT}
                L[f"policy_net_b{b}"] = {d: np.mean([r["policy_net"][d] for r in rs], axis=0) for d in DT}
if PPO_ARCHS:
    log(f"PPO runs done; wall for the PPO block {time.time() - ppo_t0:.1f}s")

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
for arch in PPO_ARCHS:                                # PPO rows (version 2.1)
    A = arch.upper()
    if "test" in periods:
        for b in BUDGETS:
            for seed in PPO_SEEDS:
                g = test_ev[(test_ev.controller == f"PPO_{A}") & (test_ev.arch == arch) & (test_ev.seed == seed)
                            & (test_ev["mode"] == "argmax") & (test_ev.budget == b)].sort_values("week")
                assert len(g) == 21 and list(g["start_h"]) == periods["test"]["starts"], (arch, seed, b)
                assert (g["update"] == ppo_selected(arch, seed)[0]["selected_update"]).all()
                got = ppo_runs["test"][(arch, b, seed)]["weekly_each"]
                check("test", f"test_eval_v3 PPO_{A} s{seed} b{b} mean", float(g["lost"].mean()), float(np.mean(got)))
                for wi, (a_, c_) in enumerate(zip(g["lost"].values, got)):
                    check_rows.append(("test", f"PPO_{A} s{seed} b{b} week {wi}", float(a_), float(c_), abs(a_ - c_) <= 0.5))
                bad = [(wi, int(a_), c_) for wi, (a_, c_) in enumerate(zip(g["lost"].values, got)) if abs(a_ - c_) > 0.5]
                if bad:
                    print(f"  per-week MISMATCH test PPO_{A} s{seed} b{b} (week, csv, got): {bad}")
    if "val" in periods:                              # eval_main_<arch>_s<seed>.csv: one row per update x mode x budget, lost = mean over the 4 val weeks
        for seed in PPO_SEEDS:
            ev = pd.read_csv(V3 / "runs" / f"eval_main_{arch}_s{seed}.csv")
            u = ppo_selected(arch, seed)[0]["selected_update"]
            for b in BUDGETS:
                row = ev[(ev["update"] == u) & (ev["mode"] == "argmax") & (ev.budget == b)]
                assert len(row) == 1, (arch, seed, b, len(row))
                got = ppo_runs["val"][(arch, b, seed)]["weekly_each"]
                check("val", f"eval_main_{arch}_s{seed} u{u} b{b} mean", float(row["lost"].iloc[0]), float(np.mean(got)))
                exp_w = [float(row[f"lost_w{k}"].iloc[0]) for k in range(1, 5)]
                check_rows.append(("val", f"eval_main_{arch}_s{seed} b{b} weeks", sum(exp_w), float(sum(got)), abs(sum(exp_w) - sum(got)) <= 2.0))
                if any(abs(a_ - c_) > 0.5 for a_, c_ in zip(exp_w, got)):
                    print(f"  per-week MISMATCH val PPO_{A} s{seed} b{b} (csv, got): {exp_w} {got}")
    if arch == "gnn" and "test" in periods:           # seed-mean vs the ladder already in map_data.json (PPO GNN [b20, b40])
        for b, e in ((20, 1371.3), (40, 302.2)):
            check("test", f"ladder PPO GNN seed-mean b{b}", e, weekly["test"][f"ppo_gnn_b{b}"], tol=0.5)
for p in periods:
    for b in BUDGETS:
        for arch in PPO_ARCHS:
            s = sum(layers[p][f"lost_ppo_{arch}_b{b}"][d].sum() * periods[p]["n_days"][d] for d in DT)
            e = totals[p][f"lost_ppo_{arch}_b{b}"]
            check_rows.append((p, f"layer sum lost_ppo_{arch}_b{b}", float(e), float(s), abs(s - e) <= 0.5))
        if "gnn" in PPO_ARCHS:
            s = sum(layers[p][f"ppo_gain_b{b}"][d].sum() * periods[p]["n_days"][d] for d in DT)
            e = totals[p][f"lost_mpc_b{b}"] - totals[p][f"lost_ppo_gnn_b{b}"]
            check_rows.append((p, f"layer sum ppo_gain_b{b}", float(e), float(s), abs(s - e) <= 0.5))
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
layer_order += ["dep_per_dock", "fill"]
for b in BUDGETS:
    layer_order += [f"lost_ppo_{a}_b{b}" for a in PPO_ARCHS]
    if "gnn" in PPO_ARCHS:
        layer_order += [f"{n}_b{b}" for n in ("ppo_gain", "override", "policy_net")]
layers_json = {p: {k: {d: np.round(layers[p][k][d], 3).tolist() for d in DT} for k in layer_order} for p in ("val", "test")}

meta = {
    "title": "Oslo bike-share: demand and rebalancing",
    "source": "v3 simulator and MPC (the submitted study)",
    "periods": {p: {"label": PERIOD_LABELS[p], "n_weeks": len(periods[p]["starts"]), "days": periods[p]["n_days"]} for p in ("val", "test")},
    "budgets_run": BUDGETS,
    "horizon": HORIZON,
    "built": datetime.datetime.now().replace(microsecond=0).isoformat(),
}
if PPO_ARCHS:
    meta["ppo"] = {f"{a}_checkpoints": [ppo_selected(a, s)[1].name for s in PPO_SEEDS] for a in PPO_ARCHS}
    meta["ppo"].update({"seeds": PPO_SEEDS, "mode": "argmax",
                        "selected_updates": {a: [ppo_selected(a, s)[0]["selected_update"] for s in PPO_SEEDS] for a in PPO_ARCHS}})
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
