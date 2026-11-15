"""Trace a small set of cells and points to DWR's own rasters (Milestone 4 feedback, fix 1).  NEEDS INTERNET (DWR image server).

Run from the repository root, after label_source_check.py:    python trace_label_vs_raster.py        (about 3 minutes)

Question: the earlier check (check_units_against_dwr_raster.py, Nov 2) found no cell-level agreement between the label and DWR's
rasters (rank correlation -0.07). label_source_check.py found that the point file's series restart at exactly 0 after every
gap of 96+ days, and the label's straight-line fit runs across those restarts. This script asks DWR's raster the same question
at the POINTS' OWN locations, not only at cell centres, so that three explanations can be told apart:
   1. alignment / aggregation (a 100 m pixel at a cell centre versus the median of a 500 m cell)  -> points agree, centres do not
   2. the restart artifact lives in the point file only                                        -> raster follows the reset-aware change
   3. the raster is built from the same restarting series                                        -> raster follows the raw readings
Sampling (seed 0, nothing tuned): 20 cells where no usable point restarts, 20 where every usable point restarts, 20 mixed;
cells need at least 5 usable points; up to 4 points per cell that have readings on both dates.
Dates are those of the Nov 2 check: 2020-02-01 and 2024-12-01. Raster values are in FEET (DWR's stated unit) and are converted
to millimetres with x 304.8. Writes evidence/trace_label_vs_raster_points.csv and ..._cells.csv. If a request fails the script
says so and stops without a verdict. To test the logic offline, point DWR_BASE at a local stand-in service.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from scipy.stats import spearmanr

BASE = os.environ.get("DWR_BASE", "https://gis.water.ca.gov/arcgisimg/rest/services/SAR/Vertical_Displacement_TRE_ALTAMIRA_Total_Since_20150613_{}/ImageServer")
STUDY = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/insar_study_area.csv")
OUT = Path("evidence"); OUT.mkdir(exist_ok=True)
D0, D1 = "20200201", "20241201"
FT = 304.8
START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")
rng = np.random.default_rng(0)


def raster_value(date, lon, lat, tries=3):
    url = BASE.format(date) + "/identify"
    params = {"geometry": json.dumps({"x": float(lon), "y": float(lat), "spatialReference": {"wkid": 4326}}), "geometryType": "esriGeometryPoint",
              "returnGeometry": "false", "returnCatalogItems": "false", "f": "json"}
    err = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60); r.raise_for_status(); j = r.json()
            if "error" in j:
                raise RuntimeError(j["error"])
            v = j.get("value")
            return float(v) if v not in (None, "NoData", "") else np.nan
        except Exception as e:
            err = e; time.sleep(1.5)
    raise RuntimeError(f"request failed for {url}: {err}")


# ---- points and their readings
head = pd.read_csv(STUDY, nrows=0).columns
win = [(pd.to_datetime(c[1:], format="%Y%m%d"), c) for c in head if c.startswith("D") and c[1:].isdigit() and START <= pd.to_datetime(c[1:], format="%Y%m%d") <= END]
cols = [c for _, c in win]
meta = pd.read_csv(STUDY, usecols=["CODE", "LAT", "LON"])
V = pd.read_csv(STUDY, usecols=cols, low_memory=False).apply(pd.to_numeric, errors="coerce").to_numpy(float)
t = np.array([(d - START).days / 365.25 for d, _ in win]); tdays = np.array([(d - START).days for d, _ in win])
i0, i1 = cols.index("D" + D0), cols.index("D" + D1)
M = np.isfinite(V); n = M.sum(1)
wy = (END - START).days / 365.25
slope1 = np.full(len(V), np.nan); slope2 = np.full(len(V), np.nan); span1 = np.zeros(len(V)); span2 = np.zeros(len(V)); resets = np.zeros(len(V), int)
for i in np.flatnonzero(n >= 20):
    j = np.flatnonzero(M[i]); tt, vv = t[j], V[i, j]
    span1[i] = tt[-1] - tt[0]
    slope1[i] = ((tt - tt.mean()) * (vv - vv.mean())).sum() / ((tt - tt.mean()) ** 2).sum()
    cut = np.flatnonzero(np.diff(tdays[j]) >= 30) + 1
    resets[i] = len(cut)
    sxx = sxy = sp = 0.0
    for s_ in np.split(np.arange(len(j)), cut):
        if len(s_) >= 2:
            a, b = tt[s_], vv[s_]; sxx += ((a - a.mean()) ** 2).sum(); sxy += ((a - a.mean()) * (b - b.mean())).sum(); sp += a[-1] - a[0]
    span2[i] = sp
    if sxx > 0:
        slope2[i] = sxy / sxx
usable = (n >= 20) & (span1 >= 0.6 * wy) & np.isfinite(slope1)
both = M[:, i0] & M[:, i1]
dt01 = t[i1] - t[i0]
raw = V[:, i1] - V[:, i0]                      # what the file's own readings say between the two dates (includes any restart)
segm = slope2 * dt01                           # the same change from the reset-aware fit (within-segment slope, trend carried across gaps)

# ---- cells
cells = pd.read_csv(OUT / "label_v2_cells.csv")
proc = pd.read_csv("data/subsidence_dataset_processed.csv", usecols=["grid_id", "longitude", "latitude"])
from pyproj import Transformer
from scipy.spatial import cKDTree
to = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
cx, cy = to.transform(proc.longitude.to_numpy(), proc.latitude.to_numpy()); px, py = to.transform(meta.LON.to_numpy(), meta.LAT.to_numpy())
_, cell_of = cKDTree(np.c_[cx, cy]).query(np.c_[px, py], k=1, p=np.inf, distance_upper_bound=250)
pts = pd.DataFrame({"i": np.arange(len(V)), "cell": cell_of, "usable": usable, "reset": resets > 0, "both": both})
pts = pts[(pts.cell < len(proc)) & pts.usable]
g = pts.groupby("cell").agg(n=("i", "size"), share_reset=("reset", "mean"), n_both=("both", "sum"))
g = g[(g.n >= 5) & (g.n_both >= 3)]
pick = {"no_restart": g[g.share_reset == 0].index.to_numpy(), "all_restart": g[g.share_reset == 1].index.to_numpy(), "mixed": g[(g.share_reset > 0.3) & (g.share_reset < 0.7)].index.to_numpy()}
print({k: len(v) for k, v in pick.items()}, "cells available per group")
chosen = {k: rng.choice(v, min(20, len(v)), replace=False) for k, v in pick.items()}

rows_p, rows_c = [], []
total = sum(len(v) for v in chosen.values()); done = 0
for grp, cl in chosen.items():
    for c in cl:
        sub = pts[(pts.cell == c) & pts.both]
        sub = sub.iloc[rng.permutation(len(sub))[:4]]
        for i in sub.i:
            a, b = raster_value(D0, meta.LON[i], meta.LAT[i]), raster_value(D1, meta.LON[i], meta.LAT[i])
            rows_p.append({"group": grp, "grid_id": proc.grid_id[c], "CODE": meta.CODE[i], "lon": meta.LON[i], "lat": meta.LAT[i], "restarts_in_window": int(resets[i]),
                           "file_raw_change_mm": raw[i], "file_reset_aware_change_mm": segm[i], "raster_ft_2020_02": a, "raster_ft_2024_12": b})
        ca, cb = raster_value(D0, proc.longitude[c], proc.latitude[c]), raster_value(D1, proc.longitude[c], proc.latitude[c])
        cr = cells.iloc[c]
        rows_c.append({"group": grp, "grid_id": proc.grid_id[c], "n_usable_points": int(g.n[c]), "share_of_points_that_restart": float(g.share_reset[c]),
                       "label_v1_file_sign_mm": -cr.label_v1_pos_sinking_mm, "label_v2_file_sign_mm": -cr.label_v2_pos_sinking_mm,
                       "raster_centre_ft_2020_02": ca, "raster_centre_ft_2024_12": cb})
        done += 1
        if done % 10 == 0:
            print(f"  {done} of {total} cells queried", flush=True)
P, C = pd.DataFrame(rows_p), pd.DataFrame(rows_c)
P["raster_change_mm"] = (P.raster_ft_2024_12 - P.raster_ft_2020_02) * FT
C["raster_centre_change_mm"] = (C.raster_centre_ft_2024_12 - C.raster_centre_ft_2020_02) * FT
P.to_csv(OUT / "trace_label_vs_raster_points.csv", index=False); C.to_csv(OUT / "trace_label_vs_raster_cells.csv", index=False)


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return (float(spearmanr(a[m], b[m]).statistic) if m.sum() > 5 else float("nan")), int(m.sum())


def ratio(a, b, thr=0.2 * FT):
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > thr)
    return float(np.median(a[m] / b[m])) if m.sum() else float("nan"), int(m.sum())


print(f"\nraster has no value at {P.raster_change_mm.isna().mean():.0%} of the {len(P)} points and {C.raster_centre_change_mm.isna().mean():.0%} of the {len(C)} cell centres")
if P.raster_change_mm.notna().sum() < 30:
    sys.exit("too few points with raster values to judge; see evidence/trace_label_vs_raster_points.csv")
print("\nAT THE POINTS' OWN LOCATIONS  (raster change 2020-02 to 2024-12, mm, versus what the file says at the same point)")
print("  group          points  rho(raster, raw readings)  ratio raw/raster   rho(raster, reset-aware)  ratio reset-aware/raster")
verdict = {}
for name, mask in [("all", P.group.notna()), ("no restart", P.restarts_in_window == 0), ("restarts", P.restarts_in_window > 0)]:
    s = P[mask]
    r1, n1 = sp(s.raster_change_mm.to_numpy(), s.file_raw_change_mm.to_numpy()); r2, _ = sp(s.raster_change_mm.to_numpy(), s.file_reset_aware_change_mm.to_numpy())
    q1, _ = ratio(s.file_raw_change_mm.to_numpy(), s.raster_change_mm.to_numpy()); q2, _ = ratio(s.file_reset_aware_change_mm.to_numpy(), s.raster_change_mm.to_numpy())
    verdict[name] = (r1, r2, q1, q2)
    print(f"  {name:<13} {n1:6d}  {r1:22.2f}  {q1:14.2f}  {r2:24.2f}  {q2:14.2f}")
r_raw, r_seg = verdict["restarts"][0], verdict["restarts"][1]
print("\nAT THE CELL CENTRES  (rank correlation with the raster change at the centre)")
for name in ("no_restart", "all_restart", "mixed"):
    s = C[C.group == name]
    a1, n1 = sp(s.label_v1_file_sign_mm.to_numpy(), s.raster_centre_change_mm.to_numpy()); a2, _ = sp(s.label_v2_file_sign_mm.to_numpy(), s.raster_centre_change_mm.to_numpy())
    print(f"  {name:<12} cells {n1:3d}   label v1 {a1:5.2f}   label v2 {a2:5.2f}")
print("\nREADING THE RESULT")
if np.isfinite(r_raw) and np.isfinite(r_seg):
    if r_raw > 0.6 and r_raw - r_seg > 0.2:
        print("  Points that restart: the raster follows the RAW readings, so the raster contains the same restarts (explanation 3). The label cannot be checked against this raster for those points.")
    elif r_seg > 0.4 and r_seg - r_raw > 0.3:
        print("  Points that restart: the raster follows the RESET-AWARE change, so the restarts are an artifact of the point file only (explanation 2) and the reset-aware label (v2) is the one that agrees with DWR's raster.")
    else:
        print("  Points that restart: the raster follows neither closely. Look at evidence/trace_label_vs_raster_points.csv before drawing a conclusion.")
print("  Points that do not restart: a high rank correlation and a ratio near 1.0 mean the point file and the raster agree in unit (mm), sign and timing there.")
