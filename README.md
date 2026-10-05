# Land Subsidence Susceptibility Mapping — Data Workspace

Module 7 (data collection warm-up) for EGN 6216 AI Systems. This is the data-access
groundwork for **Milestone 3: Data Pipeline & Management**.

## Project

Predicting land subsidence susceptibility for ~500 m grid cells in the Tulare-Corcoran
area of California's San Joaquin Valley, for a GSA groundwater manager prioritizing
monitoring. Full system design is in the Milestone 2 plan (`Milestone2_Subsidence_Project.docx`,
tracked separately from this repo).

## Repository structure

```
playground.ipynb     # experimentation notebook — run top to bottom
README.md            # this file
requirements.txt      # Python dependencies
.gitignore
build_dataset.py     # downloads all sources and builds the dataset
data/                 # dataset (see Dataset below)
  subsidence_dataset.csv
```

## Dataset

`data/subsidence_dataset.csv` is built by `build_dataset.py` from real public sources. Each row is one ~500 m grid cell in the Tulare-Corcoran area (5,192 cells, 30 columns).

| Layer | Source | Notes |
|---|---|---|
| Groundwater | CA DWR Periodic Groundwater Level Measurements (data.cnra.ca.gov) | Tulare and Kings counties, sample of wells; trend, mean/latest depth, spread, count; each cell takes its nearest well |
| Land use | USDA NASS Cropland Data Layer, 2016-2019 | Crop codes per year plus derived fields |
| Soil | USDA NRCS Soil Data Access (SSURGO) | Dominant component per map unit |
| Elevation / slope | USGS 3DEP 30 m DEM | Via the `py3dep` package |

| | |
|---|---|
| **Retrieved** | 2026-10-05 |
| **Licenses** | USDA and USGS data are public domain. DWR license text: (paste the line printed by the script) |
| **Storage** | The CSV (about 1.6 MB) is in `data/`; raw rasters are downloaded by the script and gitignored |
| **To rebuild** | `python build_dataset.py` |

**Not included yet:** the InSAR displacement label. The DWR InSAR raster files are hosted on a Box page rather than a direct download, so I am working from the point-data download instead. Geologic layers are also not included yet; soil is the only substrate layer so far.

**Known data-quality issues** (see Section 4 of the notebook): about 6-8% of cells have no soil data, and groundwater values are filled from the nearest well, so far-from-well cells are weakly supported.

## Running the notebook

1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
2. (Optional) Rebuild the dataset from the public sources: `python build_dataset.py`
3. Open `playground.ipynb` and run all cells top to bottom. The notebook reads `data/subsidence_dataset.csv` by a relative path and needs no API keys or accounts.

## Deployment note

Target deployment is Docker. `requirements.txt` is kept current as dependencies are added.
Developed and run on Python 3.11.

## No credentials / no sensitive data

This repository contains no API keys, tokens or credentials. All data comes from public
government sources and describes land, soil and wells (well locations only, no owner or
personal information), so nothing here is personal or sensitive.
