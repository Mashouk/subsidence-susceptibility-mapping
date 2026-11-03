# Land Subsidence Susceptibility Mapping: Data Workspace

Data pipeline for **EGN 6216 AI Systems, Milestone 3 (Data Collection and Management)**.

The project builds a relative land-subsidence susceptibility ranking for 500 m grid cells in the
Tulare-Corcoran area of California's San Joaquin Valley. The intended user is a groundwater manager at a
Groundwater Sustainability Agency (GSA) who uses the map to decide where to prioritize monitoring.
The output is a **relative ranking from 2020-2024 InSAR history. It is not a forecast and not a guarantee
that damage will occur at any location.**

This repository holds the data work only: the grid, the predictor layers, the InSAR-based outcome label,
the train/test split and the files that record where everything came from. Milestone 4 adds the model experiments described below.

---

## Repository contents

| Path | What it is |
|---|---|
| `build_dataset.py` | Downloads the public sources and joins them onto the 500 m grid. Writes `data/subsidence_dataset.csv`. |
| `playground.ipynb` | Exploration notebook for the grid. |
| `insar_label.ipynb` | Reads the DWR InSAR point file, builds the outcome label, the spatial split and the class boundaries. Writes `data/subsidence_dataset_processed.csv`, `data/label_build_log.json` and the label figures. |
| `fig44_sketch.ipynb` | Draws the Figure 4.4 mockup (exported susceptibility map and legend) from the labeled grid. |
| `requirements.txt` / `environment.yml` | Python packages with the versions the experiments ran with. |
| `training_experiments.ipynb` | Milestone 4: all experiments in one notebook. Part A: Random Forest runs, leakage checks, LIME, gradient boosting (E5), ablation (E6), fold check (C2), error analysis of the selected forest, and Step 10 (label support, block-edge buffer, fold-wise cut points). Part B: CNN runs, LIME checks. |
| `selected_rf_error_analysis.py` | Milestone 4: confidence and error figures for the selected forest. |
| `experiment_results.csv`, `experiment_results_cnn.csv`, `experiment_results_e5_e6.csv`, `experiment_results_step10.csv` | Validation scores saved by Part A, Part B, Step 8 and Step 10 of the notebook. |
| `verify_data.py`, `data/checksums.json` | Recomputes the SHA-256 of every data file and compares it with the recorded value (see File checksums). |
| `check_units_against_dwr_raster.py` | Compares the label with DWR's own feet rasters at the same places, to check the unit and sign of the point file (needs internet). |
| `data/subsidence_dataset.csv` | Predictor table, 5,192 cells x 30 columns (about 1.6 MB, version-controlled). |
| `data/subsidence_dataset_processed.csv` | Same table plus the label, 5,192 cells x 37 columns (about 1.9 MB, version-controlled). |
| `data/label_build_log.json` | Settings, counts, class cut points and file checksums for the label build. |
| `data/insar_extract_log.json` | Which DWR files reached the study area, and the checksum of the downloaded zip. |
| `figures/` | `label_and_split_maps.png`, `subsidence_distribution.png`, `fig_4_4_sketch.png`. |

Not in the repository (listed in `.gitignore`, because of size; the small CDL and DEM rasters in `data/` are committed): `data/insar_points.zip` (5.3 GB),
`data/insar_study_area.csv` (159 MB) and the raw per-source downloads (CDL rasters, DEM).

---

## The dataset

### Predictors (`data/subsidence_dataset.csv`)

5,192 cells on a 500 m grid, 30 columns. Retrieved 2026-10-05.

- **Groundwater:** California DWR Periodic Groundwater Level Measurements (Tulare and Kings counties, up to 300
  wells, 2016-2019 window). Each cell takes its nearest well, so cells far from a well have weakly supported values.
- **Land use:** USDA NASS Cropland Data Layer, 2016-2019.
- **Soil:** USDA NRCS SSURGO.
- **Elevation and slope:** USGS 3DEP 30 m DEM, through `py3dep`.

### Outcome label (`data/subsidence_dataset_processed.csv`)

Built by `insar_label.ipynb`, built 2026-10-12. Six columns are added to the predictor table, plus a
`soil_missing` flag (1 where soil values are missing; nothing is imputed or dropped):

| Column | Meaning |
|---|---|
| `insar_disp_window` | Median displacement of the usable InSAR points in the cell over 2020-01-01 to 2024-12-31, in the file's own sign (negative = sinking). |
| `subsidence` | Same value with the sign flipped, so **positive = sinking**. This is the label. |
| `insar_n_points` | Number of usable InSAR points in the cell. 0 means no label. |
| `block_id` | 5 km block the cell belongs to. |
| `split` | `train` or `test`, assigned by whole block. |
| `susceptibility_class` | `very_low`, `low`, `moderate`, `high`, `very_high`; empty where there is no label. |

**InSAR source.** DWR's TRE ALTAMIRA "Vertical Displacement Point Data" (package `tre-altamira-insar-subsidence`
on data.cnra.ca.gov, resource `verticaldisplacementpointdata.zip`). It holds 35 regional CSV files of about
217,000 points each (100 m x 100 m averages, 676 dates from 2015-01-01 to 2026-04-01). Only two files reach the
study area.

**How the label is built**

1. Keep the points inside the study-area box: 101,843 points (files 18 and 19).
2. For each point, fit a straight line through its valid readings from 2020-01-01 to 2024-12-31 and multiply the
   slope by the window length (4.9993 years). A point is usable if it has at least 20 readings covering at least
   60% of the window: 52,628 points (51.7%). A trend is used instead of subtracting two dates because on any one date only
   29-83% of points have a reading.
3. Flip the sign so positive = sinking.
4. Assign each usable point to the 500 m cell it falls in; the cell label is the **median** of its points.
5. Split by 5 km blocks (not single cells): 25% of labeled blocks go to test, seed 42.
6. Cut the five classes at the 20th, 40th, 60th and 80th percentile of the **training** cells only
   (-91.2, -21.0, 268.0 and 654.5). The test cells never influence the boundaries.

**Result.** 3,539 of 5,192 cells (68.2%) are labeled: 2,572 train and 967 test. 1,653 cells (31.8%) have no usable
InSAR point and are left out of training and evaluation.

---

## Model experiments (Milestone 4)

All scores are 5-fold cross-validation over whole 5 km blocks of the 40 training blocks (2,572 cells); the 967 test cells are not scored. Selected model: Random Forest, 100 trees, `min_samples_leaf=5` (macro-F1 0.369 at seed 42, 0.372 +/- 0.003 over five seeds, against the 0.50 target). The CNN was not better (0.356 +/- 0.021 over three seeds). Run `python selected_rf_error_analysis.py` from the repository root to reproduce the selected forest's calibration figures (it first prints macro-F1 0.3692). Step 10 of the notebook checks label support, a buffer along block edges and fold-wise class cut points (results in `experiment_results_step10.csv`).

---

## Known issues

- **Units are inferred, not confirmed.** DWR's catalog text says its rasters are in feet and states no unit for the point
  file (checked 2026-11-02). The technical report that probably defines the point-file columns is a PDF I could not open
  from my work environment. The values cannot be feet (the median velocity VEL_V of -107.8 per year would be about 108
  feet per year; the largest window reading, 2,536.6, would be 773 m), and they are consistent with millimetres, which is
  how the label is treated. `check_units_against_dwr_raster.py` compares the label with DWR's own feet rasters at the same
  places. Result of that check: not yet recorded here.
- **The label is a fitted-trend displacement** (slope of a line through the 2020-2024 readings times 4.9993 years), not the
  directly observed change between the first and last reading. The two have not been compared.
- **31.8% of cells have no label**, and they are most common in the south-west of the study area.
- **Thin support:** 865 labeled cells (24.4%) rest on fewer than 3 points (476 on a single point). `insar_n_points`
  is kept so this can be checked. Step 10 of the notebook shows the selected forest scores macro-F1 0.25 on development
  cells that rest on 1 or 2 points and 0.43 on cells that rest on 11 or more.
- **The label is bimodal.** One hump sits near zero and another at about 500-1,000. The `very_low` and `low` classes
  both lie in the near-zero hump, about 70 units apart.
- **Cells that appear to rise:** 650 labeled cells (18.4%) have a window displacement below -100 (480 of the 2,572
  development cells). They concentrate in thin labels: 59.8% of the development cells below -500 rest on 1 or 2 points,
  against 24.1% of all development cells, and only 0.9% of the cells that rest on 11 or more points are below -500. That
  points to measurement noise more than rebound. But the neighbors of these cells also rise more often than chance (32%
  against 19%), so some real rebound or a shared artifact is possible. No cell was removed; all of them fall in
  `very_low`. The individual point time series have not been examined.
- **Test classes are not balanced:** class shares in the test cells run from 16.3% to 26.3%, because whole blocks
  were held out.
- **Soil:** 418 cells (8.1%) have no soil values (`soil_missing = 1`).
- Geologic layers are not included yet.

---

## Run it

Python 3.11 or newer. `insar_label.ipynb` was run with Python 3.12.13 and pandas 3.0.5.

```bash
pip install -r requirements.txt        # needs pandas, numpy, scipy, pyproj, matplotlib, jupyter
python build_dataset.py                # optional: rebuilds data/subsidence_dataset.csv from the public sources
```

The InSAR step needs a manual download (no account or API key is required):

1. Download `verticaldisplacementpointdata.zip` (5.3 GB) from the DWR open-data portal, package
   `tre-altamira-insar-subsidence`, and save it as `data/insar_points.zip`.
2. Open `insar_label.ipynb`. Step 1 reads the zip and writes `data/insar_study_area.csv` (slow; it is skipped on
   later runs while that file exists).
3. Check Step 2 and Step 2b, then set `CONFIRMED_UNITS_AND_SIGN = True` in the Settings cell and run the rest. Step 3
   writes the labeled table and the log; Step 4 writes the figures.
4. Optional: run `fig44_sketch.ipynb` to redraw the Figure 4.4 mockup.

### File checksums (SHA-256)

| File | SHA-256 |
|---|---|
| `data/insar_points.zip` (5,262,324,358 bytes) | `b048035c38b6f9317d6d307c4e692f296405c99ff8340e88a00fae1fc1c1ff18` |
| `data/subsidence_dataset.csv` (5,192 x 30) | `62dd71198ccd9953d0cba2a0873fe76387b1184d779d8e08b7c867c2fb066a7a` |
| `data/insar_study_area.csv` | `59d6153b78a86536dc6563fc29a7d8c97a0bfaa479d87cf4603dacd2a71ce163` |
| `data/subsidence_dataset_processed.csv` (5,192 x 37) | `ad033f34e110042e193ef9135dcae13e5905e9a487d2b881265aa53dc35cbd50` |
| `data/dwr_groundwater_sample.csv` | `34a2a7bcb70786f3b61e3f6ed8b03d1568a35c3512c46511c6a01f0f2135fc7d` |
| `data/cdl_2016.tif` | `fc7ebe58a518ad61b0b1767132fb34e22c29115ee9d6413f34c5bb0fc512ec75` |
| `data/cdl_2017.tif` | `9925c68f88650a32eec2becb734f3484c3b0955ff57fbb8d49a3142d5b3e70bb` |
| `data/cdl_2018.tif` | `e064da096db0ffb04cb4660cad9be181d895ee247ef55497009b5b7f418afb95` |
| `data/cdl_2019.tif` | `a9b842721a409c2eeac418d4e5b7130016fd52e60910dad27974bc007296ede0` |
| `data/dem_30m.tif` | `68670d080999df40f95f4a45bdb1b1aa3f39d351683e47429d0a32633c5b5e68` |

Run `python verify_data.py` to check all of these at once. The CSV checksums are for the bytes as written on Windows (CRLF
line endings); `.gitattributes` (`*.csv -text`) makes git keep those bytes on every machine. If the script reports LINE
ENDINGS CHANGED, the content is right and only the line endings differ. If it reports MISMATCH, the file is not the one
recorded in `data/label_build_log.json`; opening and re-saving a CSV in a spreadsheet program changes its bytes.
The CDL and DEM rasters and the groundwater CSV are the downloads as used. The SSURGO soil responses were not saved;
the soil values are identified only through the checksum of `data/subsidence_dataset.csv`.

---

## Licenses and credit

- **DWR groundwater measurements:** the portal lists the license as "Other (Public Domain)" with no license URL.
- **DWR TRE ALTAMIRA InSAR point data: license unresolved.** Three copies of DWR's record disagree (checked 2026-11-02).
  The CKAN record on data.cnra.ca.gov says "License not specified". The catalog.data.gov copy says "No license information
  was provided", "This dataset is intended for public access and use" and "This dataset is covered by different Terms of
  Use than Data.gov". A copy on lab.data.ca.gov says "Public domain". DWR's text also disclaims warranties and gives no
  credit line or terms for derived data. None of this is a permission statement for sharing derivatives, so the question
  is open and has to be put to DWR. Until it is answered, use the derived columns for the course project only. This
  repository holds one derived value per 500 m cell (`insar_disp_window`, `subsidence`, `insar_n_points`,
  `susceptibility_class`), not the InSAR points. Credit: California Department of Water Resources and TRE ALTAMIRA.
- **USDA NASS CDL, USDA NRCS SSURGO, USGS 3DEP:** U.S. Government data, public domain.
- **Code:** MIT License (see `LICENSE`).

## Privacy and deployment

The repository contains no credentials or personal data, only public land, soil, elevation and well-location
data. The intended deployment target is Docker.
