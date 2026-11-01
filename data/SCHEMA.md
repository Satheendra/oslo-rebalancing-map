# Data schema (version 2, v3 numbers) shared by `src/build_data.py` (producer) and `src/template.html` (consumer)

Producer writes `data/map_data.json` and `data/layers.geojson`. `src/build_html.py` embeds `map_data.json`
into `docs/index.html`.

## `map_data.json`

```jsonc
{
  "meta": {
    "title": "Oslo bike-share: demand and rebalancing",
    "source": "v3 simulator and MPC (the submitted study)",
    "periods": {                                   // ordered: val first, test second
      "val":  {"label": "Validation: 4 weeks, Sept 2025", "n_weeks": 4,  "days": {"wd": 20,  "we": 8}},
      "test": {"label": "Test: 21 weeks, May to Sept 2026", "n_weeks": 21, "days": {"wd": 105, "we": 42}}
    },
    "budgets_run": [40, 20],
    "horizon": 6,
    "built": "2026-11-01T21:30:00"
  },
  "zones": { /* GeoJSON FeatureCollection, 30 Polygon or MultiPolygon features, WGS84.
               Built as Voronoi cells of the 276 stations, unioned per zone, clipped to the convex hull of
               the stations buffered by 400 m. properties: {"zone": int, "cap": int, "n_stations": int,
               "lat": float, "lon": float}  (lat/lon = zone centre from zones.csv). No layer values here. */ },
  "stations": [ {"id": 377, "name": "Tøyenparken", "lat": 59.9157, "lon": 10.7776, "zone": 0, "cap": 29}, ... ],  // 276
  "edges": [ [0, 5], [0, 12], ... ],              // zone graph from edge_index.npy, undirected, each pair once, i < j
  "daytypes": ["wd", "we"],
  "layers": {
    "val":  { /* layer dict, see below */ },
    "test": { /* same keys */ }
  },
  "weekly": {
    "val":  {"none": 6112.5, "heur_b40": ..., "mpc_b40": ..., "oracle_b40": ..., "heur_b20": ..., "mpc_b20": ..., "oracle_b20": ...},
    "test": { same keys }
  },
  "trips_per_week": {"val": 32400.5, "test": ...},   // mean observed departures per week in that period
  "ladder": {
    // The paper's controller ladder on the TEST weeks (from runs/test_eval_v3_table_a.csv): mean lost trips per week.
    // PPO rows = mean over seeds 0-2 of the argmax rows. Also a validation ladder from runs/val_references.csv
    // (reference controllers only, no PPO rows).
    "budgets": [20, 40],
    "test": [
      {"key": "none",    "label": "No rebalancing",        "kind": "reference", "lost": [6818.8, 6818.8]},
      {"key": "heur",    "label": "Hourly occupancy rule", "kind": "reference", "lost": [3944.0, 2597.1]},
      {"key": "mpc",     "label": "Full MPC",              "kind": "reference", "lost": [1603.0, 521.8]},
      {"key": "ppo_mlp", "label": "PPO, MLP policy",       "kind": "learned",   "lost": [1437.1, 330.9]},
      {"key": "ppo_gnn", "label": "PPO, GNN policy",       "kind": "learned",   "lost": [1371.3, 302.2]},
      {"key": "oracle",  "label": "MPC, perfect forecast", "kind": "ceiling",   "lost": [1376.4, 228.8]}
    ],
    "val": [ same keys for none, heur, mpc, oracle (from val_references.csv, mean of 4 weeks) ]
  }
}
```

### Layer dict (one per period)

Every entry: `{"wd": [[24 floats] x 30 zones], "we": [[24 floats] x 30 zones]}`; outer index = zone id 0..29,
inner = hour of day 0..23; values are MEANS PER HOUR over the days of that day type in that period.

| key | meaning |
|---|---|
| `dep`, `arr` | observed departures and arrivals per zone-hour |
| `dep_gru`, `arr_gru` | GRU forecast (mean of seeds 0-2, the `gru_mean` rung the MPC used) |
| `lost_none_b40`, `lost_none_b20` | trips lost with no rebalancing (identical values under both keys) |
| `lost_heur_bXX` | trips lost, hourly occupancy heuristic |
| `lost_mpc_bXX` | trips lost, full MPC (H = 6, GRU forecast) |
| `lost_oracle_bXX` | trips lost, MPC with the realised next-hour demand |
| `oracle_gain_bXX` | `lost_mpc - lost_oracle` |
| `shortage_bXX`, `overflow_bXX` | MPC(GRU) `plan()` hour-1 shortage and overflow slack per zone, mean over decision hours |

## `layers.geojson`

Same 30 zone features. Properties: `zone`, `cap`, `n_stations`, `lat`, `lon`, then one flat key per
period x layer x daytype x hour: `"<period>_<layer>_<daytype>_<HH>"`, e.g. `"val_dep_wd_07"`,
`"test_lost_mpc_b40_we_17"`. Values rounded to 3 decimals. Plus a second FeatureCollection is NOT
needed: stations and edges live only in `map_data.json`.
