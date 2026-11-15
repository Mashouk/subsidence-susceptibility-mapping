"""prepare.py - trains the selected model and saves per-cell predictions for the Gradio app (app.py).

Model = the one selected in Milestone 4: Random Forest, 100 trees, min_samples_leaf=5, seed 42, same
preprocessing as selected_rf_error_analysis.py.
Label: SUBSIDENCE_DATA chooses it. data/v2/... (default) is the reset-aware label, data/subsidence_dataset_processed.csv is
the label as built for Milestone 4 (v1). The model, folds and inputs are identical; only the outcome label differs.

How the prediction for each cell is made (so the app never shows a score the model has already seen):
  * labeled training cells -> out-of-fold prediction (5-fold, whole 5 km blocks held out), the same folds
    that give the macro-F1 reported in the README
  * labeled test cells and cells with no InSAR label -> the model fitted on all training cells

Run:  python prepare.py          (app.py also runs it automatically when artifacts/ is missing)
"""
import hashlib
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA = os.environ.get("SUBSIDENCE_DATA", "data/v2/subsidence_dataset_processed.csv")   # v2 = reset-aware label (default); v1 = data/subsidence_dataset_processed.csv
OUT = os.environ.get("SUBSIDENCE_ARTIFACTS", "artifacts")
SEED = 42
CLASSES = ["very_low", "low", "moderate", "high", "very_high"]
NUM = ["n_years_cropped", "n_years_fallow", "is_perennial_crop", "land_use_changed", "clay_pct", "sand_pct",
       "silt_pct", "organic_matter_pct", "ksat_um_per_s", "available_water_capacity", "elevation_m", "slope_deg",
       "gw_decline_m_per_yr_2016_2019", "gw_mean_depth_m", "gw_latest_depth_m", "gw_depth_std_m",
       "gw_n_measurements", "dist_to_nearest_well_m", "soil_missing"]
CAT = ["drainagecl", "hydgrp", "taxclname", "land_use_name_last_year"]
FEATURES = NUM + CAT


def make_model():
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("prep", prep),
                     ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                   random_state=SEED, n_jobs=1))])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(force=False):
    """Train (or reuse) the model and the per-cell predictions. Returns a small metadata dict."""
    os.makedirs(OUT, exist_ok=True)
    meta_path = os.path.join(OUT, "meta.json")
    pred_path = os.path.join(OUT, "predictions.csv")
    model_path = os.path.join(OUT, "model.joblib")
    digest = _sha256(DATA)

    if not force and all(os.path.exists(p) for p in (meta_path, pred_path, model_path)):
        meta = json.load(open(meta_path))
        if meta.get("data_sha256") == digest and meta.get("sklearn") == sklearn.__version__ and "label_version" in meta:
            return meta

    df = pd.read_csv(DATA)
    labeled = df.susceptibility_class.notna()
    dev = (labeled & (df.split == "train")).to_numpy()
    y_all = df.susceptibility_class.map({c: i for i, c in enumerate(CLASSES)})
    X = df[FEATURES]

    # out-of-fold predictions for the training cells (whole 5 km blocks held out)
    idx = np.flatnonzero(dev)
    y = y_all.iloc[idx].to_numpy().astype(int)
    groups = df.block_id.iloc[idx].to_numpy()
    proba = np.zeros((len(df), 5))
    source = np.array(["fitted on all training cells (cell not used for training)"] * len(df), dtype=object)
    oof = np.zeros((len(idx), 5))
    for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X.iloc[idx], y, groups):
        m = make_model().fit(X.iloc[idx[tr]], y[tr])
        oof[va] = m.predict_proba(X.iloc[idx[va]])
    proba[idx] = oof
    source[idx] = "cross-validated (this cell's 5 km block was left out when the model made this prediction)"
    macro_f1 = float(f1_score(y, oof.argmax(1), average="macro"))

    # final model on all training cells -> every other cell
    final = make_model().fit(X.iloc[idx], y)
    rest = np.flatnonzero(~dev)
    proba[rest] = final.predict_proba(X.iloc[rest])

    out = pd.DataFrame({"grid_id": df.grid_id,
                        "pred_class": [CLASSES[i] for i in proba.argmax(1)],
                        "confidence": proba.max(1),
                        "pred_source": source})
    for i, c in enumerate(CLASSES):
        out["p_" + c] = proba[:, i]
    out.to_csv(pred_path, index=False)
    joblib.dump(final, model_path)

    log = {}
    log_path = os.path.join(os.path.dirname(DATA), "label_build_log.json")
    if os.path.exists(log_path):
        log = json.load(open(log_path))
    meta = {"data_sha256": digest, "sklearn": sklearn.__version__, "oof_macro_f1": round(macro_f1, 4),
            "n_cells": int(len(df)), "n_train_cells": int(len(idx)),
            "n_test_cells": int((labeled & (df.split == "test")).sum()),
            "n_unlabeled": int((~labeled).sum()),
            "cutpoints": log.get("cutpoints_from_train_only"),
            "label_version": log.get("label_version", "v1"), "label_method": log.get("label_method"),
            "window": [log.get("settings", {}).get("START"), log.get("settings", {}).get("END")]}
    json.dump(meta, open(meta_path, "w"), indent=2)
    return meta


if __name__ == "__main__":
    m = build(force=True)
    print("label %s, out-of-fold macro-F1: %.4f (README: 0.3692 for v1, 0.4292 for v2, seed 42)" % (m["label_version"], m["oof_macro_f1"]))
    print(m)
