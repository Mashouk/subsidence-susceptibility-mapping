"""Export the susceptibility map as a GeoTIFF for QGIS / ArcGIS (requirement 2.2, row "export the susceptibility map as a GeoTIFF").

Run from the repository root:
    pip install rasterio          (the only package beyond requirements-app.txt)
    python export_geotiff.py      -> exports/subsidence_susceptibility_<label>_<date>.tif   (about 2 seconds after prepare.py)
Optional: SUBSIDENCE_DATA (which label; default data/v2/...), EXPORT_DIR, GIT_COMMIT (written into the file's tags).

What the file holds. One 88 x 59 grid of 500 m cells in EPSG:5070 (the grid the model was built on), 5 bands, float32:
    1  predicted class, 1 = very low ... 5 = very high           (a RELATIVE ranking from 2020-2024 InSAR history, not a forecast)
    2  confidence = the largest class probability of the forest   (0 to 1; see 7.5: bands, not raw probability, are what to read)
    3  map state: 1 classified (confidence >= 0.4), 2 low confidence (< 0.4), 3 excluded (soil data missing, inputs were imputed)
    4  observed class from the InSAR label (0 = this cell has no label)
    5  number of InSAR points behind the observed label (0 = none; fewer than 3 = thin label)
The tags (QGIS: Layer Properties > Information) carry the version, generation date, predictor cutoff, label version, data checksum,
model, cross-validated macro-F1 and the appropriate-use caveat. A line is appended to exports/CHANGELOG.md on every run.
The probabilities of labeled development cells are out-of-fold; test cells and unlabeled cells use the model fitted on all
development cells (prepare.py); band 4 is the label, not a prediction.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

import prepare

CLASSES = prepare.CLASSES
CELL = 500.0
LOW_CONF = 0.4          # the same threshold as the app's "low confidence" statement (app.py) and the map states of 3.3 / 7.5
VERSION = "1.0"

t0 = time.perf_counter()
meta = prepare.build()                                     # reuses artifacts/ when the data and library versions match
t_prepare = time.perf_counter() - t0
t1 = time.perf_counter()
df = pd.read_csv(prepare.DATA)
pred = pd.read_csv(os.path.join(prepare.OUT, "predictions.csv"))
assert (pred.grid_id.values == df.grid_id.values).all(), "predictions.csv does not match the data table; delete artifacts/ and rerun"

x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(df.longitude.values, df.latitude.values)
col = np.round((x - x.min()) / CELL).astype(int)
row = np.round((y.max() - y) / CELL).astype(int)
H, W = row.max() + 1, col.max() + 1
assert H * W == len(df) and len(set(zip(row, col))) == len(df), "the cell centres do not form a complete 500 m grid"
left, top = x.min() - CELL / 2, y.max() + CELL / 2
transform = from_origin(left, top, CELL, CELL)

cls = pred.pred_class.map({c: i + 1 for i, c in enumerate(CLASSES)}).to_numpy(float)
conf = pred.confidence.to_numpy(float)
state = np.where(df.soil_missing.to_numpy() == 1, 3, np.where(conf < LOW_CONF, 2, 1)).astype(float)
obs = df.susceptibility_class.map({c: i + 1 for i, c in enumerate(CLASSES)}).fillna(0).to_numpy(float)
npts = df.insar_n_points.fillna(0).to_numpy(float)

bands = [("predicted class (1 very low ... 5 very high)", cls), ("confidence (largest class probability)", conf),
         ("map state (1 classified, 2 low confidence, 3 excluded: soil missing)", state),
         ("observed class from InSAR label (0 = no label)", obs), ("InSAR points behind the label", npts)]
arr = np.full((len(bands), H, W), -9999, dtype="float32")
for b, (_, v) in enumerate(bands):
    arr[b, row, col] = v

try:
    commit = os.environ.get("GIT_COMMIT") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
except Exception:
    commit = ""
today = dt.date.today().isoformat()
label = meta.get("label_version", "v1")
outdir = os.environ.get("EXPORT_DIR", "exports")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, f"subsidence_susceptibility_{label}_{today}.tif")
tags = {"title": "Relative land-subsidence susceptibility, Tulare-Corcoran study area, San Joaquin Valley",
        "map_version": VERSION, "generated": today, "git_commit": commit or "not recorded",
        "predictor_window": "2016-01-01 to 2019-12-31 (archived-availability cutoff 2019-12-31, see 2.1B)",
        "outcome_window": " to ".join(str(v) for v in meta.get("window", ["2020-01-01", "2024-12-31"])),
        "label_version": label, "label_method": str(meta.get("label_method")),
        "class_cut_points_mm_training_quintiles": json.dumps([round(c, 1) for c in (meta.get("cutpoints") or [])]),
        "model": "RandomForestClassifier(n_estimators=100, min_samples_leaf=5, seed 42), 19 numeric + 4 categorical inputs",
        "cross_validated_macro_F1_development_blocks": str(meta.get("oof_macro_f1")),
        "test_region_scored": "no",
        "data_sha256": meta.get("data_sha256", ""),
        "crs_note": "EPSG:5070 (NAD83 / Conus Albers), 500 m cells, 59 rows x 88 columns",
        "appropriate_use": "Relative ranking of 2020-2024 InSAR-derived subsidence history, from 2016-2019 conditions, to help prioritize "
                           "monitoring. NOT a forecast, NOT a guarantee that damage will occur, valid only for this study area and period. "
                           "Confident predictions can still be wrong across whole neighborhoods (see the final document, 7.5)."}
with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=len(bands), dtype="float32", crs="EPSG:5070",
                   transform=transform, nodata=-9999, compress="deflate") as dst:
    dst.write(arr)
    for b, (name, _) in enumerate(bands, start=1):
        dst.set_band_description(b, name)
    dst.update_tags(**tags)
t_write = time.perf_counter() - t1

cl = os.path.join(outdir, "CHANGELOG.md")
line = (f"- {today} v{VERSION} label {label}, commit {commit or 'not recorded'}: {os.path.basename(path)}; "
        f"cross-validated macro-F1 {meta.get('oof_macro_f1')}; data sha256 {meta.get('data_sha256', '')[:12]}\n")
old = open(cl, encoding="utf-8").read() if os.path.exists(cl) else ""
if line not in old:                                   # one line per distinct export
    with open(cl, "a", encoding="utf-8") as f:
        if not old:
            f.write("# Map export changelog\n\nOne line per export. Version = the map file format and rules; label = v1 (as built for Milestone 4) or v2 (reset-aware).\n\n")
        f.write(line)

# read the file back the way a GIS program would and check it
with rasterio.open(path) as src:
    back = src.read()
    ok = {"crs_is_EPSG_5070": src.crs.to_epsg() == 5070, "shape": [src.count, src.height, src.width],
          "cell_size_m": [src.res[0], src.res[1]], "bounds": [round(v, 1) for v in src.bounds],
          "tags_present": all(k in src.tags() for k in ("map_version", "generated", "predictor_window", "label_version", "appropriate_use")),
          "classes_in_1_to_5": bool(np.isin(back[0], [1, 2, 3, 4, 5]).all()),
          "confidence_in_0_1": bool(((back[1] >= 0) & (back[1] <= 1)).all()),
          "no_nodata_cells_inside_the_grid": bool((back != -9999).all())}
    # sample the raster at every cell centre and compare with the table (a round trip through the CRS and the transform)
    vals = np.array([v[0] for v in src.sample(list(zip(x, y)), indexes=1)])
    ok["every_cell_centre_returns_its_own_class"] = bool((vals == cls).all())
    counts = {int(k): int(v) for k, v in zip(*np.unique(back[2], return_counts=True))}
ok["map_state_counts_1_classified_2_low_conf_3_excluded"] = counts
ok["seconds_prepare_or_reuse_model"] = round(t_prepare, 2)
ok["seconds_to_write_and_check_geotiff"] = round(t_write, 2)
ok["file"] = path
ok["file_bytes"] = os.path.getsize(path)
os.makedirs("evidence", exist_ok=True)
json.dump(ok, open(os.path.join("evidence", "geotiff_export_check.json"), "w"), indent=2)
print(json.dumps(ok, indent=2))
if not all(v for k, v in ok.items() if isinstance(v, bool)):
    sys.exit("GeoTIFF check failed; see the lines above")
