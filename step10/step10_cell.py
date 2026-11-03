# ---- Step 10 (Milestone 4): three checks prompted by the Milestone 3 feedback ----------------------------------------
# A. Label support. Each label is the median of the InSAR points inside a cell (insar_n_points of them). How do the
#    scores change when a cell's label rests on one point, a few points, or many?
# B. Block edges. The 5 km blocks are not separated by a buffer. How many validation cells touch a training cell
#    from another block, and what happens to the score when a buffer is removed from the training side?
# C. Class cut points. The four cut points were computed from all 2,572 development cells, so inside the folds the validation
#    cells' labels helped set them. Here they are recomputed from the training part of each fold only.
# Model = the selected forest (100 trees, min_samples_leaf 5), same block-grouped folds, same preprocessing as Run 2.
# The 967 test cells are used only as positions (to count neighbours). Their labels are never read. Runtime: about 1 minute.
import os, numpy as np, pandas as pd, time
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score

S10_SEED = 42
S10_CLASSES = ["very_low", "low", "moderate", "high", "very_high"]
S10_NUM = ["n_years_cropped", "n_years_fallow", "is_perennial_crop", "land_use_changed", "clay_pct", "sand_pct", "silt_pct",
           "organic_matter_pct", "ksat_um_per_s", "available_water_capacity", "elevation_m", "slope_deg",
           "gw_decline_m_per_yr_2016_2019", "gw_mean_depth_m", "gw_latest_depth_m", "gw_depth_std_m", "gw_n_measurements",
           "dist_to_nearest_well_m", "soil_missing"]
S10_CAT = ["drainagecl", "hydgrp", "taxclname", "land_use_name_last_year"]

_path = next(p for p in ["data/subsidence_dataset_processed.csv", "../data/subsidence_dataset_processed.csv"] if os.path.exists(p))
full = pd.read_csv(_path)
lab = full[full["susceptibility_class"].notna()]
dev10 = lab[lab["split"] == "train"].reset_index(drop=True)             # the 2,572 development cells
y10 = dev10["susceptibility_class"].map({c: i for i, c in enumerate(S10_CLASSES)}).to_numpy()
g10 = dev10["block_id"].to_numpy()
X10 = dev10[S10_NUM + S10_CAT]
npts = dev10["insar_n_points"].to_numpy()
assert len(dev10) == 2572 and set(dev10["split"]) == {"train"}

def s10_model(seed):
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), S10_NUM),
                              ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), S10_CAT)])
    return Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1))])

folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=S10_SEED).split(X10, y10, g10))
def mf1(idx, pred): return f1_score(y10[idx], pred[idx], average="macro", labels=range(5), zero_division=0)
def acc(idx, pred): return accuracy_score(y10[idx], pred[idx])
def w1(idx, pred): return float((np.abs(y10[idx] - pred[idx]) <= 1).mean())

def oof(seed=S10_SEED, drop_train=None):
    """Out-of-fold predictions. drop_train(tr_idx, va_idx) -> boolean mask of training cells to KEEP."""
    pred = np.zeros(len(y10), int)
    for tr, va in folds:
        if drop_train is not None:
            tr = tr[drop_train(tr, va)]
        m = s10_model(seed).fit(X10.iloc[tr], y10[tr]); pred[va] = m.predict(X10.iloc[va])
    return pred

base = oof()
allidx = np.arange(len(y10))
print(f"selected forest, seed {S10_SEED}: macro-F1 {mf1(allidx, base):.4f} on all {len(y10)} validation cells (must equal 0.3692)\n")

# ---------------------------------------------------------------- A. label support
print("A. LABEL SUPPORT: how many InSAR points does each development label rest on?")
bins = [("1 point", npts == 1), ("2 points", npts == 2), ("3-5 points", (npts >= 3) & (npts <= 5)),
        ("6-10 points", (npts >= 6) & (npts <= 10)), ("11 or more", npts >= 11)]      # same groups as Step 9
print("   cells by support:", {k: int(m.sum()) for k, m in bins}, "| median", int(np.median(npts)))
rowsA = []
for name, m in bins + [("3 or more (cumulative)", npts >= 3), ("5 or more (cumulative)", npts >= 5)]:
    idx = np.where(m)[0]
    rowsA.append({"cells with": name, "n": len(idx), "macro-F1": mf1(idx, base), "accuracy": acc(idx, base), "within one class": w1(idx, base)})
tabA = pd.DataFrame(rowsA).set_index("cells with")
print(tabA.round(3).to_string())
print("   (macro-F1 on a small group is noisy: a class with few cells can swing it. Accuracy is steadier.)")

# a quality rule applied to TRAINING only: keep a training cell only if its label rests on >= k points; score ALL validation cells
print("\n   Rule applied to the training side only (validation cells unchanged, so scores are comparable):")
rowsA2 = []
for k in (1, 2, 3, 5):
    keep = (lambda k: (lambda tr, va: npts[tr] >= k))(k)
    sc = [mf1(allidx, oof(seed=s, drop_train=keep)) for s in range(5)]
    ntrain = np.mean([int((npts[tr] >= k).sum()) for tr, va in folds])
    rowsA2.append({"training labels need at least": f"{k} point(s)", "mean training cells per fold": ntrain,
                   "macro-F1 (seeds 0-4 mean)": np.mean(sc), "sd": np.std(sc, ddof=1)})
tabA2 = pd.DataFrame(rowsA2).set_index("training labels need at least")
print(tabA2.round(4).to_string())

# ---------------------------------------------------------------- B. block edges
print("\nB. BLOCK EDGES")
R = 6371008.8; lat0 = np.deg2rad(full["latitude"].mean())
def xy(d): return np.c_[np.deg2rad(d["longitude"]) * np.cos(lat0) * R, np.deg2rad(d["latitude"]) * R]
P_all = xy(full); spacing = float(np.median(cKDTree(P_all).query(P_all, k=2)[0][:, 1]))
adj = 1.5 * spacing                                   # reaches the eight neighbours of a cell (diagonal is 1.41 x spacing)
print(f"   median spacing between neighbouring grid cells: {spacing:.0f} m; 'adjacent' = within {adj:.0f} m")

test_pos = xy(full[(full["split"] == "test") & full["susceptibility_class"].notna()]); dev_pos = xy(dev10)
nb_test = (cKDTree(dev_pos).query(test_pos, k=1)[0] <= adj)
print(f"   real split: {int(nb_test.sum())} of {len(test_pos)} labeled test cells ({100 * nb_test.mean():.1f}%) have a development cell next to them "
      f"(positions only; no test label is read)")

edge = np.zeros(len(y10), bool); rowsB = []
for k_, (tr, va) in enumerate(folds, 1):
    d_to_train = cKDTree(dev_pos[tr]).query(dev_pos[va], k=1)[0]
    edge[va] = d_to_train <= adj
    print(f"   fold {k_}: {int(edge[va].sum())} of {len(va)} validation cells ({100 * edge[va].mean():.1f}%) touch a training cell from another block")
print(f"   all folds: {int(edge.sum())} of {len(y10)} ({100 * edge.mean():.1f}%)")
interior = ~edge
print(f"   unbuffered score on edge cells {mf1(np.where(edge)[0], base):.3f} (acc {acc(np.where(edge)[0], base):.3f}), "
      f"on interior cells {mf1(np.where(interior)[0], base):.3f} (acc {acc(np.where(interior)[0], base):.3f})")

print("\n   Buffered training: remove every training cell within r of a validation cell, then score the same validation cells:")
for r_m in (0, adj, 2 * adj, 4 * adj):
    def keepfun(tr, va, r_m=r_m):
        if r_m == 0: return np.ones(len(tr), bool)
        return cKDTree(dev_pos[va]).query(dev_pos[tr], k=1)[0] > r_m
    sc_all, sc_int, ntr = [], [], []
    for s in range(5):
        p = oof(seed=s, drop_train=keepfun)
        sc_all.append(mf1(allidx, p)); sc_int.append(mf1(np.where(interior)[0], p))
    ntr = np.mean([int(keepfun(tr, va).sum()) for tr, va in folds])
    rowsB.append({"buffer": "none" if r_m == 0 else f"{r_m / 1000:.2f} km", "training cells per fold": ntr,
                  "macro-F1, all validation cells": np.mean(sc_all), "sd": np.std(sc_all, ddof=1), "macro-F1, interior cells only": np.mean(sc_int)})
tabB = pd.DataFrame(rowsB).set_index("buffer")
print(tabB.round(4).to_string())
print("   (seeds 0-4 change only the forest's randomness; the folds stay as for seed 42. Compare the drop with the sd.)")

# ---------------------------------------------------------------- C. cut points recomputed inside each fold
print("\nC. CLASS CUT POINTS FROM THE TRAINING PART OF EACH FOLD ONLY")
sub10 = dev10["subsidence"].to_numpy()                      # positive = sinking; the classes are its quintiles
cuts_all = np.quantile(sub10, [0.2, 0.4, 0.6, 0.8])
print("   cut points from all 2,572 development cells:", np.round(cuts_all, 1), "(the values stored in the classes: check they reproduce them)")
assert (np.digitize(sub10, cuts_all) == y10).mean() > 0.99
rowsC = []
for s in range(5):
    pred_f = np.zeros(len(y10), int); y_f = np.zeros(len(y10), int)
    for tr, va in folds:
        cuts = np.quantile(sub10[tr], [0.2, 0.4, 0.6, 0.8])
        yy = np.digitize(sub10, cuts)
        m = s10_model(s).fit(X10.iloc[tr], yy[tr]); pred_f[va] = m.predict(X10.iloc[va]); y_f[va] = yy[va]
    rowsC.append(f1_score(y_f, pred_f, average="macro", labels=range(5), zero_division=0))
    rowsC[-1] = (mf1(allidx, oof(seed=s)), rowsC[-1])
a = np.array(rowsC)
print(f"   cut points from all development cells : macro-F1 {a[:,0].mean():.4f} +/- {a[:,0].std(ddof=1):.4f} (seeds 0-4, same folds)")
print(f"   cut points from each fold's training  : macro-F1 {a[:,1].mean():.4f} +/- {a[:,1].std(ddof=1):.4f}   difference {a[:,1].mean()-a[:,0].mean():+.4f}")
tabC = pd.DataFrame({"cut points from": ["all development cells", "training part of each fold"], "macro-F1 (seeds 0-4 mean)": a.mean(0), "sd": a.std(0, ddof=1)})

out = pd.concat({"A_support_by_group": tabA.reset_index(), "A_training_rule": tabA2.reset_index(), "B_buffer": tabB.reset_index(), "C_cutpoints": tabC})
out.to_csv("experiment_results_step10.csv"); print("\nsaved experiment_results_step10.csv")
