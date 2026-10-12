# Land Subsidence Susceptibility Mapping: Data Workspace

Data pipeline for **EGN 6216 AI Systems, Milestone 3 (Data Collection and Management)**.

The project builds a relative land-subsidence susceptibility ranking for 500 m grid cells in the
Tulare-Corcoran area of California's San Joaquin Valley. The intended user is a groundwater manager at a
Groundwater Sustainability Agency (GSA) who uses the map to decide where to prioritize monitoring.
The output is a **relative ranking from 2020-2024 InSAR history. It is not a forecast and not a guarantee
that damage will occur at any location.**

This repository holds the data work only: the grid, the predictor layers, the InSAR-based outcome label,
the train/test split and the files that record where everything came from. No model has been trained yet.

---

## Repository contents

| Path | What it is |
|---|---|
| `build_dataset.py` | Downloads the public sources and joins them onto the 500 m grid. Writes `data/subsidence_dataset.csv`. |
| `playground.ipynb` | Exploration notebook for the grid. |
| `insar_label.ipynb` | Reads the DWR InSAR point file, builds the outcome label, the spatial split and the class boundaries. Writes `data/subsidence_dataset_processed.csv`, `data/label_build_log.json` and the label figures. |
| `fig44_sketch.ipynb` | Draws the Figure 4.4 mockup (exported susceptibility map and legend) from the labeled grid. |
| `requirements.txt` | Python packages. |
| `data/subsidence_dataset.csv` | Predictor table, 5,192 cells x 30 columns (about 1.6 MB, version-controlled). |
| `data/subsidence_dataset_processed.csv` | Same table plus the label, 5,192 cells x 37 columns (about 1.9 MB, version-controlled). |
| `data/label_build_log.json` | Settings, counts, class cut points and file checksums for the label build. |
| `data/insar_extract_log.json` | Which DWR files reached the study area, and the checksum of the downloaded zip. |
| `figures/` | `label_and_split_maps.png`, `subsidence_distribution.png`, `fig_4_4_sketch.png`. |

Not in the repository (listed in `.gitignore`, because of size): `data/insar_points.zip` (5.3 GB),
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

## Known issues

- **Units are inferred, not confirmed.** The point file does not state its unit. The largest window reading (2,536.6)
  is only plausible as millimetres, so the label is treated as millimetres. This has not been confirmed in DWR's data
  report.
- **31.8% of cells have no label**, and they are most common in the south-west of the study area.
- **Thin support:** 865 labeled cells (24.4%) rest on fewer than 3 points (476 on a single point). `insar_n_points`
  is kept so this can be checked.
- **The label is bimodal.** One hump sits near zero and another at about 500-1,000. The `very_low` and `low` classes
  both lie in the near-zero hump, about 70 units apart.
- **Cells that appear to rise:** 650 labeled cells (18.4%) have a window displacement below -100. Whether this is
  real rebound or measurement error is not yet known. All of these cells fall in `very_low`.
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

If the processed CSV on your machine has a different checksum, it is not the file recorded in
`data/label_build_log.json`. Opening and re-saving a CSV in a spreadsheet program changes its bytes.

---

## Licenses and credit

- **DWR groundwater measurements:** the portal lists the license as "Other (Public Domain)" with no license URL.
- **DWR TRE ALTAMIRA InSAR point data:** the portal lists the license as "License not specified" and gives no
  license URL. DWR makes it publicly downloadable without an account, but no terms are stated. Credit goes to DWR
  and TRE ALTAMIRA, and the license question has to be settled with DWR before the derived susceptibility map is
  shared with the end user.
- **USDA NASS CDL, USDA NRCS SSURGO, USGS 3DEP:** U.S. Government data, public domain.
- **Code:** MIT License (see `LICENSE`).

## Privacy and deployment

The repository contains no credentials or personal data, only public land, soil, elevation and well-location
data. The intended deployment target is Docker.
