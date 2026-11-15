"""Offline source checks on the InSAR label (Milestone 4 feedback, fix 1).

Run from the repository root (needs data/insar_study_area.csv and data/subsidence_dataset_processed.csv; no internet):
    python label_source_check.py                      # or: python label_source_check.py path/to/insar_study_area.csv

It answers, using only the point file the label was built from:
  A. Is every stored label traceable?  Rebuilds the label from the points and compares it with the stored column.
  B. Trend versus observed.            Fitted 2020-2024 trend against the change actually read at the two ends of the window.
  C. How much do the points inside one cell disagree?  Spread, class agreement, split-half reliability, and how well ONE
     location predicts the rest of its cell (the situation of a raster read at a cell centre).
  D. Does the label agree with DWR's own velocity column VEL_V at cell level?
  E. What should DWR's interpolated raster show at a cell centre if it is built from these same points?  DWR says the rasters
     are inverse-distance-weighted from the 100 m point grid with a 500 m search radius; this applies that to the point file.
     If the real raster disagrees with this emulation, the difference is in the raster product or its query, not in how
     the label was aggregated.
Writes evidence/label_source_check.json and evidence/label_source_check_cells.csv.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

STUDY = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/insar_study_area.csv")
PROC = Path("data/subsidence_dataset_processed.csv")
LOG = Path("data/label_build_log.json")
OUT = Path("evidence"); OUT.mkdir(exist_ok=True)
START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")
MIN_OBS, MIN_SPAN_FRAC, CELL_M = 20, 0.6, 500.0
RNG = np.random.default_rng(0)
R = {}                                                   # everything printed is also stored here


def say(*a):
    print(*a, flush=True)


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() > 3 else float("nan")


# ------------------------------------------------------------------ load
head = pd.read_csv(STUDY, nrows=0).columns
dcols = sorted((pd.to_datetime(c[1:], format="%Y%m%d"), c) for c in head if c.startswith("D") and c[1:].isdigit())
win = [(d, c) for d, c in dcols if START <= d <= END]
t = np.array([(d - START).days / 365.25 for d, _ in win])
win_years = (END - START).days / 365.25
say(f"{len(win)} date columns inside {START.date()}..{END.date()}; window {win_years:.4f} years")
pre = [(d, c) for d, c in dcols if d < START]
say(f"date columns before the window: {len(pre)} ({pre[0][0].date()} first = {pre[0][1]}; next {pre[1][0].date()} .. {pre[-1][0].date()})")
meta = pd.read_csv(STUDY, usecols=["CODE", "LAT", "LON", "HEIGHT", "VEL_V", "D20150101"], low_memory=False)
V = pd.read_csv(STUDY, usecols=[c for _, c in win], low_memory=False).apply(pd.to_numeric, errors="coerce").to_numpy(float)
Vpre = pd.read_csv(STUDY, usecols=[c for _, c in pre[1:]], low_memory=False).apply(pd.to_numeric, errors="coerce").to_numpy(float)
n_pts = len(meta)
R["points"] = n_pts
say(f"{n_pts:,} points; CODE unique: {meta.CODE.is_unique}; HEIGHT median {meta.HEIGHT.median():.1f} (unit not stated)")

# ------------------------------------------------------------------ per-point trend (same arithmetic as insar_label.ipynb)
M = np.isfinite(V)
n = M.sum(1)
T = np.broadcast_to(t, V.shape)
with np.errstate(invalid="ignore", divide="ignore"):
    tbar = np.where(M, T, 0).sum(1) / n
    vbar = np.where(M, V, 0).sum(1) / n
    dt = np.where(M, T - tbar[:, None], 0)
    dv = np.where(M, V - vbar[:, None], 0)
    sxx, sxy = (dt ** 2).sum(1), (dt * dv).sum(1)
    slope = sxy / sxx
    span = np.where(M, T, -np.inf).max(1) - np.where(M, T, np.inf).min(1)
span = np.where(n >= 2, span, 0.0)
ok = (n >= MIN_OBS) & (span >= MIN_SPAN_FRAC * win_years) & np.isfinite(slope)
total = slope * win_years                                # file sign: negative = sinking
say(f"usable points: {ok.sum():,} of {n_pts:,} ({ok.mean():.1%})")
R["usable_points"] = int(ok.sum())

# ------------------------------------------------------------------ points -> cells (same rule as the notebook)
d = pd.read_csv(PROC)
to5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
cx, cy = to5070.transform(d.longitude.to_numpy(), d.latitude.to_numpy())
px, py = to5070.transform(meta.LON.to_numpy(), meta.LAT.to_numpy())
tree = cKDTree(np.c_[cx, cy])
dist, idx = tree.query(np.c_[px, py], k=1, p=np.inf, distance_upper_bound=CELL_M / 2)
inside = idx < len(d)
pt = pd.DataFrame({"cell": idx, "total": total, "ok": ok, "vel": meta.VEL_V.to_numpy()})[inside & ok]
g = pt.groupby("cell")["total"].agg(["median", "size"])

# ------------------------------------------------------------------ A. traceability
say("\nA. TRACEABILITY: rebuild the label from the points")
lab = d.insar_disp_window.to_numpy()
re_lab = np.full(len(d), np.nan); re_lab[g.index] = g["median"].to_numpy()
re_n = np.zeros(len(d), int); re_n[g.index] = g["size"].to_numpy()
both = np.isfinite(lab) & np.isfinite(re_lab)
a = {"cells_labeled_stored": int(np.isfinite(lab).sum()), "cells_labeled_rebuilt": int(np.isfinite(re_lab).sum()),
     "same_cells_labeled": bool((np.isfinite(lab) == np.isfinite(re_lab)).all()),
     "max_abs_label_difference": float(np.abs(lab[both] - re_lab[both]).max()),
     "n_points_identical_in_all_cells": bool((re_n == d.insar_n_points.to_numpy()).all())}
R["A_traceability"] = a
say(json.dumps(a, indent=2))

# ------------------------------------------------------------------ B. trend versus observed
say("\nB. TREND VS OBSERVED CHANGE (usable points)")
k = 3                                                    # mean of the first 3 and last 3 valid readings, so one noisy reading cannot decide
obs = np.full(n_pts, np.nan); fit_same = np.full(n_pts, np.nan)
for i in np.flatnonzero(ok):
    j = np.flatnonzero(M[i])
    f, l = j[:k], j[-k:]
    t0, t1 = t[f].mean(), t[l].mean()
    obs[i] = V[i, l].mean() - V[i, f].mean()
    fit_same[i] = slope[i] * (t1 - t0)
diff = obs[ok] - fit_same[ok]
B = {"points": int(ok.sum()),
     "median_abs_difference_mm": float(np.median(np.abs(diff))), "p90_abs_difference_mm": float(np.quantile(np.abs(diff), .9)),
     "median_signed_difference_mm": float(np.median(diff)),
     "spearman_observed_vs_trend": spear(obs[ok], fit_same[ok]),
     "median_R2_note": "see insar_label.ipynb Step 2 (0.90)"}
# cell label from observed change (end means minus start means), same median rule, scaled to the window length
ob_pt = pd.DataFrame({"cell": idx, "obs": obs * win_years / np.where(ok, span, np.nan), "ok": ok})[inside & ok]
ob_cell = ob_pt.groupby("cell")["obs"].median()
obs_cell = np.full(len(d), np.nan); obs_cell[ob_cell.index] = ob_cell.to_numpy()
B["cell_spearman_observed_vs_label"] = spear(obs_cell, lab)
cuts_pos = np.array(json.loads(LOG.read_text())["cutpoints_from_train_only"]) if LOG.exists() else np.array([-91.2, -21.0, 268.0, 654.5])


def klass(pos_sinking):                                  # same cut points as the label, positive = sinking
    return np.digitize(pos_sinking, cuts_pos)


m = np.isfinite(obs_cell) & np.isfinite(lab)
B["cell_class_agreement_observed_vs_label"] = float((klass(-obs_cell[m]) == klass(-lab[m])).mean())
B["cell_class_within_one"] = float((np.abs(klass(-obs_cell[m]) - klass(-lab[m])) <= 1).mean())
R["B_trend_vs_observed"] = B
say(json.dumps(B, indent=2))

# ------------------------------------------------------------------ C. disagreement inside a cell
say("\nC. HOW MUCH DO THE POINTS INSIDE ONE CELL DISAGREE?")
pt["cls"] = klass(-pt.total.to_numpy())
pt["cell_cls"] = pt.cell.map(pd.Series(klass(-re_lab[g.index]), index=g.index))
per = pt.groupby("cell")
cells = pd.DataFrame({"n": per.size(), "iqr": per.total.quantile(.75) - per.total.quantile(.25),
                      "agree": per.apply(lambda x: float((x.cls == x.cell_cls).mean()), include_groups=False),
                      "n_classes": per.cls.nunique()})
C = {}
for lo, hi, name in [(1, 1, "1 point"), (2, 2, "2 points"), (3, 5, "3-5 points"), (6, 10, "6-10 points"), (11, 99, "11+ points")]:
    s = cells[(cells.n >= lo) & (cells.n <= hi)]
    if len(s):
        C[name] = {"cells": int(len(s)), "median_share_of_points_in_the_cell_class": float(s.agree.median()),
                   "share_of_cells_where_points_span_3plus_classes": float((s.n_classes >= 3).mean()),
                   "median_IQR_mm": float(s.iqr.median()) if len(s) and (s.n >= 2).all() else None}
R["C_by_support"] = C
say(json.dumps(C, indent=2))

# split-half reliability and the leave-one-out single-location check, cells with >= 6 usable points
big = cells[cells.n >= 6].index
sub = pt[pt.cell.isin(big)]
groups = {c: x.total.to_numpy() for c, x in sub.groupby("cell")}
rs, ag = [], []
for _ in range(50):
    a_, b_ = [], []
    for c, v in groups.items():
        p = RNG.permutation(len(v)); h = len(v) // 2
        a_.append(np.median(v[p[:h]])); b_.append(np.median(v[p[h:2 * h]]))
    a_, b_ = np.array(a_), np.array(b_)
    rs.append(spear(a_, b_)); ag.append(float((klass(-a_) == klass(-b_)).mean()))
r_half = float(np.mean(rs))
C2 = {"cells": int(len(groups)), "split_half_spearman": r_half,
      "split_half_spearman_corrected_to_full_length": float(2 * r_half / (1 + r_half)),
      "split_half_class_agreement": float(np.mean(ag))}
one, rest = [], []
for c, v in groups.items():
    for i in range(len(v)):
        one.append(v[i]); rest.append(np.median(np.delete(v, i)))
C2["one_point_vs_median_of_the_rest_spearman"] = spear(one, rest)
C2["one_point_vs_median_of_the_rest_class_agreement"] = float((klass(-np.array(one)) == klass(-np.array(rest))).mean())
C2["one_point_vs_median_of_the_rest_sign_agreement"] = float((np.sign(one) == np.sign(rest)).mean())
R["C_reliability_cells_with_6plus_points"] = C2
say(json.dumps(C2, indent=2))

# ------------------------------------------------------------------ D. label vs DWR's own velocity column at cell level
say("\nD. CELL LABEL VS VEL_V (DWR's own velocity column, whole record)")
vcell = pt[pt.vel.notna()].groupby("cell")["vel"].median()
vel_cell = np.full(len(d), np.nan); vel_cell[vcell.index] = vcell.to_numpy()
mm = np.isfinite(vel_cell) & np.isfinite(lab)
D = {"cells": int(mm.sum()), "spearman_label_vs_median_VEL_V": spear(lab, vel_cell),
     "same_sign_share": float((np.sign(lab[mm]) == np.sign(vel_cell[mm])).mean()),
     "median_label_over_VEL_V_times_window_years": float(np.median(lab[mm] / (vel_cell[mm] * win_years)))}
R["D_label_vs_VEL_V"] = D
say(json.dumps(D, indent=2))

# ------------------------------------------------------------------ E. emulate DWR's raster at a cell centre
say("\nE. EMULATED RASTER (inverse-distance-weighted from the points, 500 m search radius) VS LABEL")


def nearest_col(date):
    return int(np.argmin([abs((dd - pd.Timestamp(date)).days) for dd, _ in win]))


i0, i1 = nearest_col("2020-02-01"), nearest_col("2024-12-01")
say(f"dates used: {win[i0][0].date()} and {win[i1][0].date()} (the script that queries DWR uses 2020-02-01 and 2024-12-01)")
have = np.isfinite(V[:, i0]) & np.isfinite(V[:, i1])
ptree = cKDTree(np.c_[px, py])
E = {}
sel = np.flatnonzero(np.isfinite(lab))
for power in (1, 2, 3):
    for use_all in (True, False):
        emu = np.full(len(d), np.nan)
        for c in sel:
            nb = np.array(ptree.query_ball_point([cx[c], cy[c]], r=500.0), dtype=int)
            if use_all is False:
                nb = nb[ok[nb]]
            nb = nb[have[nb]]
            if len(nb) == 0:
                continue
            dd = np.maximum(np.hypot(px[nb] - cx[c], py[nb] - cy[c]), 1.0)
            w = 1.0 / dd ** power
            emu[c] = np.sum(w * (V[nb, i1] - V[nb, i0])) / np.sum(w)
        m = np.isfinite(emu) & np.isfinite(lab)
        sub5 = m & (d.insar_n_points.to_numpy() >= 5)
        big_ = sub5 & (np.abs(emu) > 0.2 * 304.8)
        key = f"idw_power{power}_{'all_points' if use_all else 'usable_points_only'}"
        E[key] = {"cells_with_value": int(m.sum()),
                  "spearman_all_labeled_cells": spear(lab[m], emu[m]),
                  "spearman_cells_with_5plus_points": spear(lab[sub5], emu[sub5]),
                  "cells_5plus_points_and_change_over_0.2ft": int(big_.sum()),
                  "median_ratio_label_over_change": float(np.median(lab[big_] / emu[big_])),
                  "same_sign_share": float((np.sign(lab[big_]) == np.sign(emu[big_])).mean())}
        if power == 2 and use_all:
            cells_emu = emu.copy()
R["E_emulated_raster"] = E
say(json.dumps(E, indent=2))

# ------------------------------------------------------------------ F. the file's reference column
say("\nF. THE FILE'S FIRST DATE COLUMN")
r0 = pd.to_numeric(meta.D20150101, errors="coerce")
F = {"D20150101_share_empty": float(r0.isna().mean()),
     "D20150101_nonempty_points": int(r0.notna().sum()),
     "D20150101_nonempty_values_range": [float(r0.min()), float(r0.max())] if r0.notna().any() else None}
n_pre = np.isfinite(Vpre).sum(1)
F["points_with_any_reading_2019-07-07_to_2019-12-26"] = int((n_pre > 0).sum())
R["F_reference_column"] = F
say(json.dumps(F, indent=2))

# ------------------------------------------------------------------ G. series that restart at zero after a gap
say("\nG. DO THE SERIES RESTART AFTER A GAP?  (all date columns kept in the file, 2019-07-07 to 2025-06-25)")
allc = [(dd, c) for dd, c in dcols if c != "D20150101"]
Vall = pd.read_csv(STUDY, usecols=[c for _, c in allc], low_memory=False).apply(pd.to_numeric, errors="coerce").to_numpy(float)
tall = np.array([(dd - allc[0][0]).days for dd, _ in allc])
Mall = np.isfinite(Vall)
before, after, gdays = [], [], []
first_vals = []
for i in range(n_pts):
    j = np.flatnonzero(Mall[i])
    if len(j) == 0:
        continue
    first_vals.append(Vall[i, j[0]])
    if len(j) > 1:
        gp = tall[j[1:]] - tall[j[:-1]]
        for q in np.flatnonzero(gp >= 30):
            before.append(Vall[i, j[q]]); after.append(Vall[i, j[q + 1]]); gdays.append(gp[q])
before, after, gdays = map(np.array, (before, after, gdays))
first_vals = np.array(first_vals)
G = {"gaps_of_30_days_or_more": int(len(gdays)),
     "shortest_gap_days": int(gdays.min()),
     "readings_right_after_a_gap_that_are_exactly_0": float((after == 0).mean()),
     "readings_right_before_a_gap_that_are_exactly_0": float((before == 0).mean()),
     "median_value_before_gap_mm": float(np.median(before)),
     "median_gap_days": float(np.median(gdays)),
     "first_reading_of_each_point_exactly_0_share": float((first_vals == 0).mean())}
R["G_restart_after_gap"] = G
say(json.dumps(G, indent=2))

# per-point segment-aware fit: one common slope with a separate intercept for every segment (a segment starts after a gap of 30+ days)
tdays = np.array([(dd - START).days for dd, _ in win])
slope2 = np.full(n_pts, np.nan); span2 = np.zeros(n_pts); n_seg = np.zeros(n_pts, int); max_up = np.zeros(n_pts)
for i in range(n_pts):
    j = np.flatnonzero(M[i])
    if len(j) < 2:
        continue
    vv, tt = V[i, j], t[j]
    cut = np.flatnonzero(np.diff(tdays[j]) >= 30) + 1
    n_seg[i] = len(cut) + 1
    max_up[i] = max(0.0, float(np.diff(vv).max()))
    sxx2 = sxy2 = sp = 0.0
    for s_ in np.split(np.arange(len(j)), cut):
        if len(s_) >= 2:
            ts, vs = tt[s_], vv[s_]
            sxx2 += ((ts - ts.mean()) ** 2).sum(); sxy2 += ((ts - ts.mean()) * (vs - vs.mean())).sum(); sp += ts[-1] - ts[0]
    span2[i] = sp
    if sxx2 > 0:
        slope2[i] = sxy2 / sxx2
ok2 = (n >= MIN_OBS) & (span2 >= MIN_SPAN_FRAC * win_years) & np.isfinite(slope2)
total2 = slope2 * win_years
has_reset = n_seg > 1
H = {"points_with_a_reset_inside_the_window_all": float(has_reset.mean()),
     "points_with_a_reset_inside_the_window_usable_v1": float(has_reset[ok].mean()),
     "usable_v1_points_with_an_upward_step_ge_150mm": float((max_up[ok] >= 150).mean()),
     "usable_v1_points_with_an_upward_step_ge_300mm": float((max_up[ok] >= 300).mean()),
     "median_window_total_mm_usable_v1_without_reset": float(np.median(total[ok & ~has_reset])),
     "median_window_total_mm_usable_v1_with_reset": float(np.median(total[ok & has_reset])),
     "spearman_slope_vs_VEL_V_usable_v1_without_reset": spear(slope[ok & ~has_reset], meta.VEL_V.to_numpy()[ok & ~has_reset]),
     "spearman_slope_vs_VEL_V_usable_v1_with_reset": spear(slope[ok & has_reset], meta.VEL_V.to_numpy()[ok & has_reset]),
     "usable_v2_points": int(ok2.sum()), "usable_v2_share": float(ok2.mean()),
     "spearman_point_slope_v2_vs_VEL_V": spear(slope2[ok2], meta.VEL_V.to_numpy()[ok2]),
     "spearman_point_slope_v2_vs_VEL_V_points_with_reset": spear(slope2[ok2 & has_reset], meta.VEL_V.to_numpy()[ok2 & has_reset]),
     "median_slope_per_yr_v2": float(np.median(slope2[ok2]))}
R["H_reset_effect_on_points"] = H
say("\nH. EFFECT OF RESETS ON THE POINT SLOPES")
say(json.dumps(H, indent=2))

# cell label v2 (reset-aware); same cell rule, same median, same blocks and train/test split, cut points from the training cells
pt2 = pd.DataFrame({"cell": idx, "total": total2, "reset": has_reset, "vel": meta.VEL_V.to_numpy()})[inside & ok2]
g2 = pt2.groupby("cell")["total"].agg(["median", "size"])
lab2 = np.full(len(d), np.nan); lab2[g2.index] = g2["median"].to_numpy()
n2 = np.zeros(len(d), int); n2[g2.index] = g2["size"].to_numpy()
pos1, pos2 = -lab, -lab2                                  # positive = sinking, as the project's label
tr_mask = (d.split == "train").to_numpy() & np.isfinite(pos2)
cuts2 = np.quantile(pos2[tr_mask], [.2, .4, .6, .8])
cls1 = np.where(np.isfinite(pos1), np.digitize(pos1, cuts_pos), -1)
cls2 = np.where(np.isfinite(pos2), np.digitize(pos2, cuts2), -1)
both12 = np.isfinite(pos1) & np.isfinite(pos2)
vc2 = pd.DataFrame({"cell": idx, "vel": meta.VEL_V.to_numpy()})[inside & ok2].groupby("cell")["vel"].median()
vcell2 = np.full(len(d), np.nan); vcell2[vc2.index] = vc2.to_numpy()
mv1 = np.isfinite(lab) & np.isfinite(vcell2); mv2 = np.isfinite(lab2) & np.isfinite(vcell2)
reset_all = pd.DataFrame({"cell": idx, "r": has_reset})[inside & (n >= 2)].groupby("cell")["r"].mean()
reset_share_cell = np.full(len(d), np.nan); reset_share_cell[reset_all.index] = reset_all.to_numpy()
reset_use = pd.DataFrame({"cell": idx, "r": has_reset})[inside & ok].groupby("cell")["r"].mean()
reset_share_use = np.full(len(d), np.nan); reset_share_use[reset_use.index] = reset_use.to_numpy()
HH = {"cells_labeled_v1": int(np.isfinite(lab).sum()), "cells_labeled_v2": int(np.isfinite(lab2).sum()),
      "cells_labeled_in_both": int(both12.sum()),
      "cut_points_v1": [float(x) for x in cuts_pos], "cut_points_v2_from_training_cells": [float(x) for x in cuts2],
      "spearman_v1_vs_v2_label": spear(pos1[both12], pos2[both12]),
      "same_class_v1_v2": float((cls1[both12] == cls2[both12]).mean()),
      "within_one_class_v1_v2": float((np.abs(cls1[both12] - cls2[both12]) <= 1).mean()),
      "share_of_cells_sinking_v1": float((pos1[np.isfinite(pos1)] > 0).mean()),
      "share_of_cells_sinking_v2": float((pos2[np.isfinite(pos2)] > 0).mean()),
      "median_label_mm_v1": float(np.nanmedian(pos1)), "median_label_mm_v2": float(np.nanmedian(pos2)),
      "spearman_with_cell_median_VEL_V_v1": spear(lab[mv1], vcell2[mv1]), "spearman_with_cell_median_VEL_V_v2": spear(lab2[mv2], vcell2[mv2]),
      "train_cells_v2": int(((d.split == "train").to_numpy() & np.isfinite(pos2)).sum()),
      "test_cells_v2": int(((d.split == "test").to_numpy() & np.isfinite(pos2)).sum())}
by = {}
for k_, nm in enumerate(["very_low", "low", "moderate", "high", "very_high"]):
    m_ = cls1 == k_
    by[nm] = {"cells": int(m_.sum()), "mean_share_of_usable_points_with_a_reset": float(np.nanmean(reset_share_use[m_]))}
HH["reset_share_by_v1_class"] = by
HH["spearman_v1_label_vs_reset_share"] = spear(pos1, reset_share_use)
HH["spearman_v2_label_vs_reset_share"] = spear(pos2, reset_share_use)
R["H_label_v1_vs_v2"] = HH
say("\nH2. LABEL v1 (as built) VS v2 (reset-aware)")
say(json.dumps(HH, indent=2))
def loo_agree(points, cuts):
    """Per cell: share of its usable points whose own class equals the class of the median of the OTHER points (cells with 2+ points)."""
    out = {}
    for c, v in points.groupby("cell")["total"]:
        v = v.to_numpy()
        if len(v) >= 2:
            own = np.digitize(-v, cuts)
            out[c] = float(np.mean([np.digitize(-np.median(np.delete(v, i)), cuts) == own[i] for i in range(len(v))]))
    r = np.full(len(d), np.nan)
    r[list(out.keys())] = list(out.values())
    return r


loo1 = loo_agree(pd.DataFrame({"cell": idx, "total": total})[inside & ok], cuts_pos)
loo2 = loo_agree(pt2[["cell", "total"]], cuts2)
HH["leave_one_out_class_agreement_by_support_v1"] = {k_: float(np.nanmedian(loo1[(d.insar_n_points >= lo_) & (d.insar_n_points <= hi_)])) for k_, lo_, hi_ in [("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 99)]}
HH["leave_one_out_class_agreement_by_support_v2"] = {k_: float(np.nanmedian(loo2[(n2 >= lo_) & (n2 <= hi_)])) for k_, lo_, hi_ in [("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 99)]}
say(json.dumps({k_: HH[k_] for k_ in ("leave_one_out_class_agreement_by_support_v1", "leave_one_out_class_agreement_by_support_v2")}, indent=2))
pd.DataFrame({"grid_id": d.grid_id, "split": d.split, "block_id": d.block_id, "n_points_v1": d.insar_n_points, "n_points_v2": n2,
              "loo_class_agreement_v1": loo1, "loo_class_agreement_v2": loo2,
              "label_v1_pos_sinking_mm": pos1, "label_v2_pos_sinking_mm": pos2,
              "class_v1": np.where(cls1 >= 0, cls1, np.nan), "class_v2": np.where(cls2 >= 0, cls2, np.nan),
              "reset_share_of_usable_points": reset_share_use, "median_VEL_V_v2_points": vcell2}).to_csv(OUT / "label_v2_cells.csv", index=False)

# ------------------------------------------------------------------ save
out = pd.DataFrame({"grid_id": d.grid_id, "n_points": d.insar_n_points, "label_file_sign_mm": lab, "label_rebuilt": re_lab,
                    "observed_end_change_mm": obs_cell, "median_VEL_V": vel_cell,
                    "emulated_idw2_change_mm": cells_emu})
out.loc[out.label_file_sign_mm.notna()].to_csv(OUT / "label_source_check_cells.csv", index=False)
(OUT / "label_source_check.json").write_text(json.dumps(R, indent=2))
say("\nSaved evidence/label_source_check.json and evidence/label_source_check_cells.csv")
