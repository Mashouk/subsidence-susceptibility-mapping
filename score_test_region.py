"""Score the 14 held-out test blocks, ONCE (Milestone 4 target (b): macro-F1 > 0.50 on the held-out test region).

Why this is a separate script and why it has a guard.
  Every other number in this project comes from the development blocks (2,193 labeled cells with the v2 label). The 822 labeled
  test cells were never read by any script, so one score on them is still a clean final check. Scoring is a one-way act: after
  you have seen the number you cannot tune anything and call the result clean. The script therefore refuses to run twice.

Run from the repository root:
    python score_test_region.py --dry-run       # checks the wiring, prints the cell counts, does NOT compute any score
    python score_test_region.py --score-once    # scores the test cells with the model fitted on the development cells and writes
                                                # evidence/test_region_score_once.json; refuses to run again if that file exists
What it scores: the predictions in artifacts/predictions.csv for labeled test cells, which prepare.py makes with the model fitted
on all development cells (the test cells are not used for fitting, imputing or class cut points). Run prepare.py first (or just run this).
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import prepare

OUT = os.path.join("evidence", "test_region_score_once.json")
mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode not in ("--dry-run", "--score-once"):
    sys.exit(__doc__)
if mode == "--score-once" and os.path.exists(OUT):
    sys.exit(f"{OUT} exists: the test region was already scored once. Do not run this again and then tune on the result.")

meta = prepare.build()
df = pd.read_csv(prepare.DATA)
pred = pd.read_csv(os.path.join(prepare.OUT, "predictions.csv"))
assert (pred.grid_id.values == df.grid_id.values).all(), "predictions.csv does not match the data; delete artifacts/ and rerun"
test = ((df.split == "test") & df.susceptibility_class.notna()).to_numpy()
src = pred.pred_source[test].str.startswith("fitted on all training cells")
assert src.all(), "a test cell carries a cross-validated prediction; something is wrong with prepare.py"
print(f"label {meta.get('label_version')}: {int(test.sum())} labeled test cells in {df.block_id[test].nunique()} blocks; "
      f"{int(((df.split == 'train') & df.susceptibility_class.notna()).sum())} labeled development cells; "
      f"predictions for the test cells come from the model fitted on development cells only.")
if mode == "--dry-run":
    print("dry run: no score computed.")
    sys.exit(0)

CL = prepare.CLASSES
y = df.susceptibility_class[test].map({c: i for i, c in enumerate(CL)}).to_numpy()
p = pred.pred_class[test].map({c: i for i, c in enumerate(CL)}).to_numpy()
res = {"scored_at": dt.datetime.now().isoformat(timespec="seconds"), "label_version": meta.get("label_version"),
       "data_sha256": meta.get("data_sha256"), "test_cells": int(test.sum()), "test_blocks": int(df.block_id[test].nunique()),
       "macro_F1": float(f1_score(y, p, average="macro")), "accuracy": float(accuracy_score(y, p)),
       "within_one_class": float((np.abs(y - p) <= 1).mean()),
       "per_class_F1": dict(zip(CL, [float(v) for v in f1_score(y, p, average=None, labels=range(5))])),
       "confusion_matrix_rows_true_cols_predicted": confusion_matrix(y, p, labels=range(5)).tolist(),
       "target_macro_F1": 0.50,
       "note": "Scored once. Nothing was tuned after this number was seen; if you change the model or the label, this score no longer counts as a held-out result."}
os.makedirs("evidence", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
