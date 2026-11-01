# Data schema shared by `build_data.py` (producer) and `build_html.py` (consumer)

All files live in this folder (`project/v2/mdst_map/`). Nothing outside it is written.

## `map_data.json` (embedded into the HTML)

```jsonc
{
  "meta": {
    "title": "Oslo bike-share: demand and rebalancing (prototype)",
    "period": "4 validation weeks, Mon 2025-09-01 to Sun 2025-09-28",
    "n_zones": 30,
    "budgets_run": [40, 20],              // truck budgets for which unmet-demand layers exist
    "horizon": 6,
    "built": "2026-10-31T23:00:00"        // ISO timestamp
  },
  "zones": { /* GeoJSON FeatureCollection, 30 Polygon features, WGS84 lon/lat.
               properties: {"zone": int, "cap": int, "n_stations": int, "lat": float, "lon": float}
               NO layer values here (they are in "layers"). */ },
  "daytypes": ["wd", "we"],               // weekday (Mon-Fri), weekend (Sat-Sun)
  "layers": {
    // every entry: {"wd": [[24 floats] x 30 zones], "we": [[24 floats] x 30 zones]}
    // outer index = zone id 0..29, inner index = hour of day 0..23
    // values are MEANS PER HOUR over the days of that daytype in the 4 validation weeks
    "dep":            {...},   // observed departures (trips started) per zone-hour
    "arr":            {...},   // observed arrivals
    "dep_gru":        {...},   // GRU forecast of departures (mean of seeds 0,1,2)
    "arr_gru":        {...},   // GRU forecast of arrivals
    "lost_none_b40":  {...},   // trips lost, no rebalancing (budget irrelevant; same values under b20 key too)
    "lost_heur_b40":  {...},   // trips lost, hourly occupancy heuristic, budget 40
    "lost_mpc_b40":   {...},   // trips lost, MPC (H=6) with GRU forecast, budget 40
    "lost_oracle_b40":{...},   // trips lost, MPC (H=6) with oracle next-hour forecast, budget 40
    "oracle_gain_b40":{...},   // lost_mpc_b40 - lost_oracle_b40  ("where perfect information helps most")
    "shortage_b40":   {...},   // MPC(GRU) plan(): mean hour-1 predicted shortage per zone (bikes)
    "overflow_b40":   {...},   // MPC(GRU) plan(): mean hour-1 predicted overflow per zone (bikes)
    "lost_none_b20", "lost_heur_b20", "lost_mpc_b20", "lost_oracle_b20", "oracle_gain_b20",
    "shortage_b20", "overflow_b20"        // same for budget 20 (present only if meta.budgets_run contains 20)
  },
  "weekly": {
    // lost trips per WEEK (mean of the 4 weeks), computed from the same runs, for the headline numbers
    // keys: "none", "heur_b40", "mpc_b40", "oracle_b40", "heur_b20", ...
    "none": 6124.0, "heur_b40": 1016.25, "mpc_b40": 115.25, "oracle_b40": 33.25, ...
  },
  "curve": {
    // from runs/regime_map_val.csv: mean lost trips per week over 4 weeks, per budget
    "budgets": [10, 20, 40, 60, 80],
    "series": {
      "heuristic":  [4995.0, 3880.75, 1016.25, 422.0, 148.75],   // HourlyOcc (forecast-free)
      "lp_gru":     [...],                                        // old v1 LP, GRU forecast
      "mpc_gru":    [...],                                        // MPC H=6, GRU forecast
      "mpc_oracle": [...]                                         // MPC H=6, oracle next hour
    },
    "trips_per_week": 36400.0   // mean (served+lost) per week at budget 40 MPC, for a service-rate line
  }
}
```

## `layers.geojson` (for loading into Mobilitix later)

Same 30 polygons. Properties: `zone`, `cap`, `n_stations`, `lat`, `lon`, then one flat key per
layer x daytype x hour: `"<layer>_<daytype>_<HH>"`, e.g. `"dep_wd_07"`, `"lost_mpc_b40_we_17"`.
Values rounded to 3 decimals.

## Zone polygons

Voronoi cells of the 30 zone centroids (`zones.csv` lat/lon), built in a local metric frame
(UTM 32N, EPSG:32632, or a simple equirectangular scaling about the mean latitude), clipped to the
convex hull of the centroids buffered by 700 m, then converted back to lon/lat. `station_zone.csv`
has no coordinates, so station-level Voronoi is not possible from v2 data alone.
