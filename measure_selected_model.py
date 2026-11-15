"""Measures the SELECTED model (Run 2 forest: 100 trees, min_samples_leaf=5, seed 42) and, for comparison on the same machine,
Run 1 (min_samples_leaf=1): time per fold, full fit, saved size, whole-grid prediction, single-cell prediction, load time, memory.
Run from the repository root:   python measure_selected_model.py            (about 1 minute)
Writes evidence/model_measurements_<machine>.json. Run it on the machine that serves the app and commit the file."""
import json, os, platform, resource, statistics, sys, tempfile, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
DATA = os.environ.get("SUBSIDENCE_DATA", "data/subsidence_dataset_processed.csv")
SEED = 42
NUM = ["n_years_cropped", "n_years_fallow", "is_perennial_crop", "land_use_changed", "clay_pct", "sand_pct", "silt_pct", "organic_matter_pct",
       "ksat_um_per_s", "available_water_capacity", "elevation_m", "slope_deg", "gw_decline_m_per_yr_2016_2019", "gw_mean_depth_m",
       "gw_latest_depth_m", "gw_depth_std_m", "gw_n_measurements", "dist_to_nearest_well_m", "soil_missing"]
CAT = ["drainagecl", "hydgrp", "taxclname", "land_use_name_last_year"]


def model(leaf):
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                              ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf, random_state=SEED, n_jobs=1))])


def clock(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); out.append(time.perf_counter() - t)
    return out


df = pd.read_csv(DATA)
dev = df[df.susceptibility_class.notna() & (df.split == "train")].reset_index(drop=True)
y = dev.susceptibility_class.map({c: i for i, c in enumerate(["very_low", "low", "moderate", "high", "very_high"])}).to_numpy()
X, Xall = dev[NUM + CAT], df[NUM + CAT]
res = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
       "python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
       "data": {"path": DATA, "development_cells": int(len(dev)), "grid_cells": int(len(df))}, "note": "single process, n_jobs=1, as in prepare.py"}
for name, leaf in [("run2_selected_min_samples_leaf_5", 5), ("run1_min_samples_leaf_1", 1)]:
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, dev.block_id.to_numpy()))
    fold_t = [clock(lambda: model(leaf).fit(X.iloc[tr], y[tr]), 1)[0] for tr, _ in folds]
    full_t = clock(lambda: model(leaf).fit(X, y), 5)
    m = model(leaf).fit(X, y)
    path = Path(tempfile.mkdtemp()) / "m.joblib"; joblib.dump(m, path); size = path.stat().st_size
    load_t = clock(lambda: joblib.load(path), 5)
    grid_t = clock(lambda: m.predict_proba(Xall), 20)
    one = Xall.iloc[[0]]; one_t = clock(lambda: m.predict_proba(one), 200)
    res[name] = {"fit_seconds_per_fold_median": statistics.median(fold_t), "fit_seconds_per_fold_all": fold_t,
                 "full_fit_seconds_median_of_5": statistics.median(full_t), "full_fit_seconds_all": full_t,
                 "saved_model_MB": size / 1e6, "load_seconds_median": statistics.median(load_t),
                 "whole_grid_predict_seconds_median_of_20": statistics.median(grid_t), "per_cell_ms_in_grid_call": 1000 * statistics.median(grid_t) / len(df),
                 "single_cell_predict_ms_median_of_200": 1000 * statistics.median(one_t), "single_cell_predict_ms_p95": 1000 * float(np.quantile(one_t, .95))}
res["peak_resident_memory_MB_of_this_process"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
tag = platform.node().replace(" ", "_") or "machine"
Path("evidence").mkdir(exist_ok=True)
Path(f"evidence/model_measurements_{tag}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
