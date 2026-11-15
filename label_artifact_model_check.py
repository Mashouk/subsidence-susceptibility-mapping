"""Does the selected model learn subsidence, or the label's restart artifact?  (Milestone 4 feedback, fix 1 and the closing question)

Run from the repository root AFTER label_source_check.py (needs evidence/label_v2_cells.csv); no internet.
    python label_artifact_model_check.py

E10 (new, a sensitivity run, nothing existing is replaced):
  * the SELECTED forest (same code path as selected_rf_error_analysis.py: 100 trees, min_samples_leaf=5, seed 42,
    5-fold StratifiedGroupKFold over 5 km blocks, inputs fitted inside folds) trained on the label as built (v1);
  * the same forest on the reset-aware label (v2), same blocks, same split, class cut points from the v2 training cells.
For each: macro-F1, ordinal agreement, and how closely the out-of-fold ranking follows (a) the v1 label, (b) the v2 label,
(c) the share of a cell's points that restart after a gap (the artifact), (d) DWR's own VEL_V.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("evidence")
NUM = ["n_years_cropped", "n_years_fallow", "is_perennial_crop", "land_use_changed", "clay_pct", "sand_pct", "silt_pct",
       "organic_matter_pct", "ksat_um_per_s", "available_water_capacity", "elevation_m", "slope_deg",
       "gw_decline_m_per_yr_2016_2019", "gw_mean_depth_m", "gw_latest_depth_m", "gw_depth_std_m", "gw_n_measurements",
       "dist_to_nearest_well_m", "soil_missing"]
CAT = ["drainagecl", "hydgrp", "taxclname", "land_use_name_last_year"]
R = {}


def say(*a):
    print(*a, flush=True)


def prep():
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                              ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])


def forest():
    return Pipeline([("prep", prep()), ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=1))])


def oof(dev, y, groups):
    X = dev[NUM + CAT]
    pred = np.zeros(len(y), int); proba = np.zeros((len(y), 5))
    for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups):
        m = forest().fit(X.iloc[tr], y[tr]); pred[va] = m.predict(X.iloc[va]); proba[va] = m.predict_proba(X.iloc[va])
    return pred, proba


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic)


def partial_sp(x, y, z):
    """Spearman of x and y after removing what the rank of z explains in each."""
    r = lambda v: pd.Series(v).rank().to_numpy()
    x, y, z = r(x), r(y), r(z)
    res = lambda a: a - np.polyval(np.polyfit(z, a, 1), z)
    return float(np.corrcoef(res(x), res(y))[0, 1])


df = pd.read_csv("data/subsidence_dataset_processed.csv")
v2 = pd.read_csv(OUT / "label_v2_cells.csv")
df = df.merge(v2[["grid_id", "label_v2_pos_sinking_mm", "n_points_v2", "reset_share_of_usable_points", "median_VEL_V_v2_points"]], on="grid_id", how="left")
dev = df[df.susceptibility_class.notna() & (df.split == "train")].reset_index(drop=True)
CLASSES = ["very_low", "low", "moderate", "high", "very_high"]
y1 = dev.susceptibility_class.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
groups = dev.block_id.to_numpy()
exp = lambda p: p @ np.arange(5)                       # expected class from the probabilities: the ranking score

# ---------- (1) the selected forest on the label as built
pred1, proba1 = oof(dev, y1, groups)
say("macro-F1 on v1 label: %.4f (must equal 0.3692)" % f1_score(y1, pred1, average="macro"))
s1 = exp(proba1)
has2 = dev.label_v2_pos_sinking_mm.notna().to_numpy()
rs = dev.reset_share_of_usable_points.to_numpy()
lab1 = dev.subsidence.to_numpy()                       # positive = sinking, v1
lab2 = dev.label_v2_pos_sinking_mm.to_numpy()
vel = -dev.median_VEL_V_v2_points.to_numpy()           # positive = sinking, DWR's own velocity, cell median
A = {"dev_cells_v1": int(len(dev)), "macro_F1_v1_label": float(f1_score(y1, pred1, average="macro")),
     "ranking_score_vs_v1_label_spearman": sp(s1, lab1),
     "ranking_score_vs_reset_share_spearman_negative_means_more_resets_rank_lower": sp(s1, rs),
     "dev_cells_with_both_labels": int(has2.sum()),
     "ranking_score_vs_v2_label_spearman_cells_with_both": sp(s1[has2], lab2[has2]),
     "ranking_score_vs_DWR_VEL_V_spearman_cells_with_both": sp(s1[has2], vel[has2]),
     "ranking_score_vs_v2_label_after_removing_reset_share": partial_sp(s1[has2], lab2[has2], rs[has2]),
     "v1_label_vs_reset_share_spearman": sp(lab1, rs),
     "v2_label_vs_reset_share_spearman": sp(lab2[has2], rs[has2])}
R["selected_forest_trained_on_v1"] = A
say(json.dumps(A, indent=2))

# can the same inputs predict the artifact itself?  (forest regression of the reset share, same block folds)
Xd = dev[NUM + CAT]
ok_ = np.isfinite(rs)
pr = np.zeros(len(dev))
for tr, va in GroupKFold(n_splits=5).split(Xd[ok_], groups=groups[ok_]):
    idx = np.flatnonzero(ok_)
    m = Pipeline([("prep", prep()), ("rf", RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=1))])
    m.fit(Xd.iloc[idx[tr]], rs[idx[tr]]); pr[idx[va]] = m.predict(Xd.iloc[idx[va]])
R["inputs_predict_the_artifact"] = {"out_of_fold_spearman_predicted_vs_actual_reset_share": sp(pr[ok_], rs[ok_]),
                                    "out_of_fold_R2": float(1 - ((rs[ok_] - pr[ok_]) ** 2).sum() / ((rs[ok_] - rs[ok_].mean()) ** 2).sum())}
say(json.dumps(R["inputs_predict_the_artifact"], indent=2))

# ---------- (2) the same forest on the reset-aware label v2 (same blocks and split)
d2 = dev[has2].reset_index(drop=True)
cuts2 = np.quantile(d2.label_v2_pos_sinking_mm.to_numpy(), [.2, .4, .6, .8])
y2 = np.digitize(d2.label_v2_pos_sinking_mm.to_numpy(), cuts2)
pred2, proba2 = oof(d2, y2, d2.block_id.to_numpy())
s2 = exp(proba2)
rs2 = d2.reset_share_of_usable_points.to_numpy(); vel2 = -d2.median_VEL_V_v2_points.to_numpy()
# reference: the two-indicator rule of E0 (groundwater decline + clay), same folds
ref = np.zeros(len(d2), int)
Xi = d2[["gw_decline_m_per_yr_2016_2019", "clay_pct"]]
for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xi, y2, d2.block_id.to_numpy()):
    m = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]).fit(Xi.iloc[tr], y2[tr])
    ref[va] = m.predict(Xi.iloc[va])
B = {"dev_cells_v2": int(len(d2)), "v2_cut_points_from_dev_cells_mm": [float(c) for c in cuts2],
     "macro_F1_v2_label": float(f1_score(y2, pred2, average="macro")), "accuracy": float(accuracy_score(y2, pred2)),
     "within_one_class": float((np.abs(y2 - pred2) <= 1).mean()),
     "two_indicator_rule_macro_F1": float(f1_score(y2, ref, average="macro")),
     "chance_macro_F1_approx": 0.20,
     "ranking_score_vs_v2_label_spearman": sp(s2, d2.label_v2_pos_sinking_mm),
     "ranking_score_vs_DWR_VEL_V_spearman": sp(s2, vel2),
     "ranking_score_vs_reset_share_spearman": sp(s2, rs2),
     "ranking_score_vs_v2_label_after_removing_reset_share": partial_sp(s2, d2.label_v2_pos_sinking_mm.to_numpy(), rs2)}
R["same_forest_trained_on_v2"] = B
say(json.dumps(B, indent=2))
(OUT / "label_artifact_model_check.json").write_text(json.dumps(R, indent=2))
say("Saved evidence/label_artifact_model_check.json")
