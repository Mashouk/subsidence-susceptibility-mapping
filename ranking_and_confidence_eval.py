"""Ranking (hit rate at K) and map confidence states, on the development blocks only (Milestone 4 feedback, fixes 2 and 3).

Run from the repository root:
    python ranking_and_confidence_eval.py v1      # the label as built (what Milestone 4 reported)
    python ranking_and_confidence_eval.py v2      # the reset-aware label (see label_source_check.py)
Needs data/subsidence_dataset_processed.csv, evidence/label_v2_cells.csv (from label_source_check.py) and, for the persistence
proxy, either evidence/persistence_proxy_late2019.csv (written on first use) or data/insar_study_area.csv. No internet.
The 14 test blocks are never read: every number below comes from the 2,572 (v1) or 2,193 (v2) labeled development cells,
scored out of fold (5-fold StratifiedGroupKFold over whole 5 km blocks, seed 42, the Run 2 forest).

Part A  hit rate at K          "Of the K% of cells the model ranks highest, how many are in the top label class?"
Part B  map confidence states   what each state means, how many errors it lets through, where, and whether the 3-point rule holds
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"
assert VERSION in ("v1", "v2")
SEED = 42
OUT = Path("evidence"); OUT.mkdir(exist_ok=True)
NUM = ["n_years_cropped", "n_years_fallow", "is_perennial_crop", "land_use_changed", "clay_pct", "sand_pct", "silt_pct",
       "organic_matter_pct", "ksat_um_per_s", "available_water_capacity", "elevation_m", "slope_deg",
       "gw_decline_m_per_yr_2016_2019", "gw_mean_depth_m", "gw_latest_depth_m", "gw_depth_std_m", "gw_n_measurements",
       "dist_to_nearest_well_m", "soil_missing"]
CAT = ["drainagecl", "hydgrp", "taxclname", "land_use_name_last_year"]
R = {"label_version": VERSION}
rng = np.random.default_rng(0)


def say(*a):
    print(*a, flush=True)


def forest():
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                              ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=1))])


# ------------------------------------------------------------------ data
df = pd.read_csv("data/subsidence_dataset_processed.csv")
v2 = pd.read_csv(OUT / "label_v2_cells.csv")
df = df.merge(v2[["grid_id", "class_v2", "label_v2_pos_sinking_mm", "n_points_v2", "loo_class_agreement_v1", "loo_class_agreement_v2"]], on="grid_id", how="left")
if VERSION == "v1":
    df["cls"] = df.susceptibility_class.map({c: i for i, c in enumerate(["very_low", "low", "moderate", "high", "very_high"])})
    df["n_pts"] = df.insar_n_points; df["loo"] = df.loo_class_agreement_v1
else:
    df["cls"] = df.class_v2; df["n_pts"] = df.n_points_v2; df["loo"] = df.loo_class_agreement_v2
dev = df[df.cls.notna() & (df.split == "train")].reset_index(drop=True)
y = dev.cls.to_numpy().astype(int); groups = dev.block_id.to_numpy(); X = dev[NUM + CAT]
say(f"label {VERSION}: {len(dev)} development cells in {len(set(groups))} blocks; class counts {np.bincount(y).tolist()}")

# ------------------------------------------------------------------ out-of-fold predictions (Run 2 forest) and the two-indicator reference
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups))
fold_id = np.zeros(len(y), int)
pred = np.zeros(len(y), int); proba = np.zeros((len(y), 5)); lr_proba = np.zeros((len(y), 5))
Xi = dev[["gw_decline_m_per_yr_2016_2019", "clay_pct"]]
for f, (tr, va) in enumerate(folds):
    fold_id[va] = f
    m = forest().fit(X.iloc[tr], y[tr]); pred[va] = m.predict(X.iloc[va]); proba[va] = m.predict_proba(X.iloc[va])
    l = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]).fit(Xi.iloc[tr], y[tr])
    lr_proba[va] = l.predict_proba(Xi.iloc[va])
mf1 = float(f1_score(y, pred, average="macro"))
say(f"selected forest, out-of-fold macro-F1 {mf1:.4f}" + (" (must equal 0.3692)" if VERSION == "v1" else ""))
R["macro_F1_out_of_fold"] = mf1
exp_class = lambda p: p @ np.arange(5)
score_rf, score_lr = proba[:, 4], lr_proba[:, 4]                 # primary: probability of the top class (the "affected" class), same rule for both learned models
score_rf_ec, score_lr_ec = exp_class(proba), exp_class(lr_proba)   # secondary: expected class, reported because it ranks differently
conf = proba.max(1); wrong = pred != y

# ------------------------------------------------------------------ persistence proxy: the only pre-outcome readings in the extracted file (2019-07-07 .. 2019-12-26)
prox_path = OUT / "persistence_proxy_late2019.csv"
if not prox_path.exists():
    STUDY = Path("data/insar_study_area.csv")
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    head = pd.read_csv(STUDY, nrows=0).columns
    dc = sorted((pd.to_datetime(c[1:], format="%Y%m%d"), c) for c in head if c.startswith("D") and c[1:].isdigit() and c != "D20150101")
    pre = [(dd, c) for dd, c in dc if dd < pd.Timestamp("2020-01-01")]
    meta = pd.read_csv(STUDY, usecols=["LAT", "LON"])
    Vp = pd.read_csv(STUDY, usecols=[c for _, c in pre], low_memory=False).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    tp = np.array([(dd - pre[0][0]).days for dd, _ in pre])
    sl = np.full(len(Vp), np.nan)
    for i in range(len(Vp)):
        j = np.flatnonzero(np.isfinite(Vp[i]))
        if len(j) < 10:
            continue
        cut = np.flatnonzero(np.diff(tp[j]) >= 30) + 1               # a gap of 30+ days restarts the series at 0: fit within segments
        sxx = sxy = 0.0
        for s_ in np.split(np.arange(len(j)), cut):
            if len(s_) >= 2:
                ts, vs = tp[j[s_]] / 365.25, Vp[i, j[s_]]
                sxx += ((ts - ts.mean()) ** 2).sum(); sxy += ((ts - ts.mean()) * (vs - vs.mean())).sum()
        if sxx > 0:
            sl[i] = sxy / sxx
    to = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    cx, cy = to.transform(df.longitude.to_numpy(), df.latitude.to_numpy()); px, py = to.transform(meta.LON.to_numpy(), meta.LAT.to_numpy())
    _, ix = cKDTree(np.c_[cx, cy]).query(np.c_[px, py], k=1, p=np.inf, distance_upper_bound=250)
    q = pd.DataFrame({"cell": ix, "s": sl})[(ix < len(df)) & np.isfinite(sl)].groupby("cell")["s"].agg(["median", "size"])
    out = pd.DataFrame({"grid_id": df.grid_id}); out["proxy_slope_mm_per_yr_file_sign"] = np.nan; out["proxy_n_points"] = 0
    out.loc[q.index, "proxy_slope_mm_per_yr_file_sign"] = q["median"].to_numpy(); out.loc[q.index, "proxy_n_points"] = q["size"].to_numpy()
    out.to_csv(prox_path, index=False)
prox = pd.read_csv(prox_path)
dev = dev.merge(prox, on="grid_id", how="left")
persist = -dev.proxy_slope_mm_per_yr_file_sign.to_numpy()            # higher = sinking faster in late 2019
has_p = np.isfinite(persist)
R["persistence_proxy"] = {"definition": "median over the cell's InSAR points of the within-segment slope of readings dated 2019-07-07 to 2019-12-26 (at least 10 readings per point); only readings before the 2019-12-31 predictor cutoff; 6 months, so it carries the seasonal cycle",
                          "dev_cells_with_proxy": int(has_p.sum()), "dev_cells": int(len(dev)),
                          "spearman_proxy_vs_label": float(spearmanr(persist[has_p], dev.cls.to_numpy()[has_p]).statistic)}
say(json.dumps(R["persistence_proxy"], indent=2))


# ------------------------------------------------------------------ Part A: hit rate at K
def hit(score, aff, frac):
    k = max(1, int(round(frac * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    return float(aff[top].mean()), float(aff[top].sum() / max(aff.sum(), 1))


aff = (y == 4)
KS = [0.05, 0.10, 0.20]
methods = {"random": None, "random_forest_P_top_class": score_rf, "two_indicator_P_top_class": score_lr,
           "random_forest_expected_class": score_rf_ec, "two_indicator_expected_class": score_lr_ec, "persistence_late2019": persist}
A = {"primary_score": "probability of the top class, chosen before any ranking was computed because 'affected' is defined as the top class; the expected-class score is reported as well and is not used to pick a winner",
     "affected_definition": "development cell whose label is in the top class (very high, the top 20% of training labels)",
     "prevalence_in_evaluated_cells": float(aff.mean())}
for label, mask in [("all_development_cells", np.ones(len(y), bool)), ("cells_with_persistence_proxy", has_p)]:
    res = {}
    for name, sc in methods.items():
        res[name] = {}
        for f in KS:
            if name == "random":
                res[name][f"top{int(f*100)}%"] = {"hit_rate": float(aff[mask].mean()), "capture": f}
                continue
            if name == "persistence_late2019" and label == "all_development_cells":
                continue
            h, c = hit(sc[mask], aff[mask], f)
            res[name][f"top{int(f*100)}%"] = {"hit_rate": h, "capture": c}
    A[label] = {"cells": int(mask.sum()), "results": res}
# per fold and block-bootstrap intervals for the differences (forest minus each reference), pooled cells
fold_hits = []
for f in range(5):
    m = fold_id == f
    fold_hits.append({"fold": f, "cells": int(m.sum()), **{f"top{int(k*100)}%": {"rf": hit(score_rf[m], aff[m], k)[0], "two_indicator": hit(score_lr[m], aff[m], k)[0], "random": float(aff[m].mean())} for k in KS}})
A["per_fold"] = fold_hits
blocks = np.unique(groups); by_block = {b: np.flatnonzero(groups == b) for b in blocks}


def boot(score_a, score_b, mask, reps=1000):
    diffs = {k: [] for k in KS}
    for _ in range(reps):
        pick = rng.choice(blocks, len(blocks), replace=True)
        ix = np.concatenate([by_block[b] for b in pick]); ix = ix[mask[ix]]
        for k in KS:
            diffs[k].append(hit(score_a[ix], aff[ix], k)[0] - (hit(score_b[ix], aff[ix], k)[0] if score_b is not None else aff[ix].mean()))
    return {f"top{int(k*100)}%": [float(np.quantile(v, .025)), float(np.quantile(v, .5)), float(np.quantile(v, .975))] for k, v in diffs.items()}


A["block_bootstrap_difference_2.5_50_97.5_percentile"] = {
    "forest_minus_random_all_cells": boot(score_rf, None, np.ones(len(y), bool)),
    "forest_minus_two_indicator_all_cells": boot(score_rf, score_lr, np.ones(len(y), bool)),
    "forest_minus_persistence_cells_with_proxy": boot(score_rf, persist, has_p),
    "two_indicator_minus_persistence_cells_with_proxy": boot(score_lr, persist, has_p)}
A["margin_rule_from_2.2"] = "forest must beat each reference by at least 10 percentage points at 5%, 10% and 20%"
R["A_hit_rate_at_K"] = A
say("\nA. HIT RATE AT K (development cells, out of fold)")
for label in ("all_development_cells", "cells_with_persistence_proxy"):
    say(f"  {label}: {A[label]['cells']} cells")
    for name, v in A[label]["results"].items():
        say("   %-30s" % name + "  ".join(f"{k}: hit {x['hit_rate']:.3f} capture {x['capture']:.3f}" for k, x in v.items()))
say("  forest minus two-indicator, bootstrap 95% interval:", {k: [round(x, 3) for x in (v[0], v[2])] for k, v in A["block_bootstrap_difference_2.5_50_97.5_percentile"]["forest_minus_two_indicator_all_cells"].items()})

# ------------------------------------------------------------------ Part B: map confidence states
say("\nB. MAP CONFIDENCE STATES")
n_pts = dev.n_pts.to_numpy()
B = {"states": {
    "observed_layer": {"classified": "label rests on 3 or more usable InSAR points", "low_confidence_sparse_data": "label rests on 1 or 2 points (hatched)", "no_data_excluded": "no usable point (grey; no observed class)"},
    "model_layer": {"shown_with_class": "top-class probability at or above the threshold", "flagged_low_confidence": "probability below the threshold; the class is still stored but labelled low confidence"}}}


def block_table(accept):
    rows = []
    for b in blocks:
        ix = by_block[b]; a = accept[ix]
        rows.append({"block": int(b), "cells": len(ix), "accepted": int(a.sum()), "accepted_wrong": int((wrong[ix] & a).sum())})
    return pd.DataFrame(rows)


sweep = {}
for tau in (0.40, 0.50, 0.65, 0.80):
    acc = conf >= tau
    bt = block_table(acc)
    wb = bt[bt.accepted_wrong > 0].sort_values("accepted_wrong", ascending=False)
    sup = {}
    for nm, lo, hi in [("1-2 points", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 999)]:
        m = acc & (n_pts >= lo) & (n_pts <= hi)
        sup[nm] = {"accepted": int(m.sum()), "wrong": int((wrong & m).sum()), "error_rate": float((wrong & m).sum() / m.sum()) if m.sum() else None}
    big = bt[bt.accepted >= 8]
    sweep[f"confidence>={tau}"] = {
        "coverage": float(acc.mean()), "accepted": int(acc.sum()), "accepted_wrong": int((wrong & acc).sum()),
        "accuracy_of_accepted": float((~wrong[acc]).mean()), "within_one_class_of_accepted": float((np.abs(y - pred)[acc] <= 1).mean()),
        "blocks_with_any_accepted_cell": int((bt.accepted > 0).sum()), "blocks_with_an_accepted_error": int((bt.accepted_wrong > 0).sum()), "blocks_total": int(len(bt)),
        "share_of_accepted_errors_in_the_worst_2_blocks": float(wb.accepted_wrong.head(2).sum() / max(bt.accepted_wrong.sum(), 1)),
        "blocks_with_8plus_accepted_cells": int(len(big)),
        "accepted_accuracy_across_those_blocks_min_median_max": [float((1 - big.accepted_wrong / big.accepted).min()), float((1 - big.accepted_wrong / big.accepted).median()), float((1 - big.accepted_wrong / big.accepted).max())] if len(big) else None,
        "by_label_support": sup}
B["threshold_sweep"] = sweep
for k, v in sweep.items():
    say(f"  {k}: covers {v['coverage']:.1%} ({v['accepted']} cells), {v['accepted_wrong']} wrong (accuracy {v['accuracy_of_accepted']:.3f}, within one {v['within_one_class_of_accepted']:.3f}); "
        f"errors fall in {v['blocks_with_an_accepted_error']} of {v['blocks_with_any_accepted_cell']} blocks that have accepted cells; worst 2 blocks hold {v['share_of_accepted_errors_in_the_worst_2_blocks']:.0%}")
# confidence versus accuracy per block (is the average confidence honest in each neighbourhood?)
pb = pd.DataFrame({"b": groups, "conf": conf, "ok": ~wrong}).groupby("b").agg(cells=("ok", "size"), mean_conf=("conf", "mean"), acc=("ok", "mean"))
pb = pb[pb.cells >= 20]
pb["gap"] = pb.mean_conf - pb.acc
B["per_block_confidence_vs_accuracy_blocks_with_20plus_cells"] = {"blocks": int(len(pb)), "mean_gap_conf_minus_accuracy_min_median_max": [float(pb.gap.min()), float(pb.gap.median()), float(pb.gap.max())],
                                                                   "blocks_overconfident_by_0.10_or_more": int((pb.gap >= 0.10).sum()), "blocks_underconfident_by_0.10_or_more": int((pb.gap <= -0.10).sum())}
say("  per block (20+ cells):", json.dumps(B["per_block_confidence_vs_accuracy_blocks_with_20plus_cells"]))

# the support rule, tested as a rule: does the label get more reliable at 3 points?  (a) the model's error, (b) the label's own internal agreement
rule = {}
for k in range(1, 9):
    m = n_pts >= k
    rule[f"support>={k}"] = {"cells": int(m.sum()), "macro_F1": float(f1_score(y[m], pred[m], average="macro")), "accuracy": float((~wrong[m]).mean())}
B["support_rule_threshold_sweep_model_scores"] = rule
loo = dev.loo.to_numpy()
B["label_internal_agreement_by_support_median_leave_one_out_class_agreement"] = {nm: (float(np.nanmedian(loo[(n_pts >= lo) & (n_pts <= hi)])) if np.isfinite(loo[(n_pts >= lo) & (n_pts <= hi)]).any() else None)
                                                                                for nm, lo, hi in [("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 999)]}
# repeat over other block partitions (10 different partitions): does accuracy at support>=3 beat support<3 every time, and does confidence>=0.65 stay reliable?
reps = []
for s in range(10):
    pr2 = np.zeros((len(y), 5))
    order = np.random.default_rng(s).permutation(blocks)
    fold_of_block = {b: i % 5 for i, b in enumerate(order)}
    fid = np.array([fold_of_block[g] for g in groups])
    for f in range(5):
        tr, va = np.flatnonzero(fid != f), np.flatnonzero(fid == f)
        m = forest().fit(X.iloc[tr], y[tr]); pr2[va] = m.predict_proba(X.iloc[va])
    p2 = pr2.argmax(1); c2 = pr2.max(1); w2 = p2 != y
    hi_ = c2 >= 0.65
    reps.append({"partition": s, "macro_F1": float(f1_score(y, p2, average="macro")),
                 "accuracy_support_ge3": float((~w2[n_pts >= 3]).mean()), "accuracy_support_lt3": float((~w2[n_pts < 3]).mean()),
                 "cells_conf_ge_0.65": int(hi_.sum()), "accuracy_conf_ge_0.65": float((~w2[hi_]).mean()) if hi_.sum() else None,
                 "accuracy_conf_lt_0.65": float((~w2[~hi_]).mean())})
B["ten_block_partitions"] = reps
rp = pd.DataFrame(reps)
B["ten_block_partitions_summary"] = {"support_ge3_better_than_lt3_in_partitions": int((rp.accuracy_support_ge3 > rp.accuracy_support_lt3).sum()),
                                     "accuracy_gap_ge3_minus_lt3_min_mean_max": [float((rp.accuracy_support_ge3 - rp.accuracy_support_lt3).min()), float((rp.accuracy_support_ge3 - rp.accuracy_support_lt3).mean()), float((rp.accuracy_support_ge3 - rp.accuracy_support_lt3).max())],
                                     "accuracy_when_conf_ge_0.65_min_mean_max": [float(rp["accuracy_conf_ge_0.65"].min()), float(rp["accuracy_conf_ge_0.65"].mean()), float(rp["accuracy_conf_ge_0.65"].max())],
                                     "cells_when_conf_ge_0.65_min_mean_max": [int(rp["cells_conf_ge_0.65"].min()), float(rp["cells_conf_ge_0.65"].mean()), int(rp["cells_conf_ge_0.65"].max())],
                                     "macro_F1_min_mean_max": [float(rp.macro_F1.min()), float(rp.macro_F1.mean()), float(rp.macro_F1.max())]}
say("  support rule, label internal agreement:", json.dumps(B["label_internal_agreement_by_support_median_leave_one_out_class_agreement"]))
say("  ten partitions:", json.dumps(B["ten_block_partitions_summary"]))
R["B_confidence_states"] = B
(OUT / f"ranking_and_confidence_{VERSION}.json").write_text(json.dumps(R, indent=2))
pd.DataFrame({"grid_id": dev.grid_id, "block_id": groups, "fold": fold_id, "label_class": y, "pred_class": pred, "confidence": conf, "n_points": n_pts,
              "score_forest": score_rf, "score_two_indicator": score_lr, "persistence_proxy": persist}).to_csv(OUT / f"development_oof_predictions_{VERSION}.csv", index=False)
say(f"\nSaved evidence/ranking_and_confidence_{VERSION}.json and evidence/development_oof_predictions_{VERSION}.csv")
