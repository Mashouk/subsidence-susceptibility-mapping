"""Writes data/v2/subsidence_dataset_processed.csv and data/v2/label_build_log.json: the SAME grid, predictors, blocks and
train/test split as data/subsidence_dataset_processed.csv, with the outcome label rebuilt so that a point's straight-line fit
never runs across a restart of its series (see label_source_check.py, sections G and H).
Run from the repository root after label_source_check.py:    python build_label_v2.py
The v1 files are not touched.  Needs evidence/label_v2_cells.csv."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

LABELS = ["very_low", "low", "moderate", "high", "very_high"]
v1 = pd.read_csv("data/subsidence_dataset_processed.csv")
c = pd.read_csv("evidence/label_v2_cells.csv")
assert (v1.grid_id.values == c.grid_id.values).all()
log1 = json.load(open("data/label_build_log.json"))
out = v1.copy()
out["subsidence"] = c.label_v2_pos_sinking_mm.values                       # positive = sinking, mm over the window
out["insar_disp_window"] = -c.label_v2_pos_sinking_mm.values               # file sign (negative = sinking), as in v1
out["insar_n_points"] = c.n_points_v2.fillna(0).astype(int).values
out["susceptibility_class"] = pd.Series(c.class_v2.values).map(lambda k: LABELS[int(k)] if pd.notna(k) else np.nan)
tr = (out.split == "train") & out.subsidence.notna()
cuts = out.loc[tr, "subsidence"].quantile([.2, .4, .6, .8]).to_numpy()
chk = pd.cut(out.subsidence, [-np.inf, *cuts, np.inf], labels=LABELS).astype(object)
assert (chk[out.subsidence.notna()].values == out.susceptibility_class[out.subsidence.notna()].values).all(), "classes do not match cut points"
Path("data/v2").mkdir(parents=True, exist_ok=True)
out.to_csv("data/v2/subsidence_dataset_processed.csv", index=False)
sha = hashlib.sha256(open("data/v2/subsidence_dataset_processed.csv", "rb").read()).hexdigest()
log = {"label_version": "v2",
       "label_method": ("per-point least-squares slope over the readings in the window, fitted with a separate intercept for every uninterrupted stretch "
                        "of readings (a new stretch starts after a gap of 30+ days, where DWR's cumulative series restarts at 0), times the window length; "
                        "a point is usable with at least 20 readings and stretches that together cover at least 60% of the window; "
                        "cell label = median over the usable points inside the cell; positive = sinking"),
       "derived_from": {"v1_dataset_sha256": hashlib.sha256(open("data/subsidence_dataset_processed.csv", "rb").read()).hexdigest(),
                        "study_area_csv_sha256": log1["study_area_csv"]["sha256"], "code": "label_source_check.py (sections G, H) and build_label_v2.py"},
       "settings": dict(log1["settings"], MIN_SPAN_FRAC_applies_to="sum of the stretch spans", GAP_DAYS=30),
       "cells_labeled": int(out.subsidence.notna().sum()), "train_cells": int(tr.sum()),
       "test_cells": int(((out.split == "test") & out.subsidence.notna()).sum()),
       "cutpoints_from_train_only": [float(x) for x in cuts],
       "output_csv": {"path": "data/v2/subsidence_dataset_processed.csv", "sha256": sha, "rows": len(out), "cols": out.shape[1]}}
json.dump(log, open("data/v2/label_build_log.json", "w"), indent=2)
print("labeled cells", log["cells_labeled"], "train", log["train_cells"], "test", log["test_cells"]); print("cut points", np.round(cuts, 1)); print("sha256", sha)
print(out.groupby("split").susceptibility_class.value_counts().unstack().reindex(columns=LABELS))
