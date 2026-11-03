"""Check the unit and sign of the InSAR label against DWR's own rasters, which DWR states are in FEET.

Run from the repository root (needs internet; about 2 minutes):   python check_units_against_dwr_raster.py

Why: the DWR point file does not state its unit (see README, Known issues). DWR's catalog text does say that its vertical
displacement rasters are "in feet" and are totals "relative to June 13, 2015". The rasters are built from the same
InSAR points, so the change between two of them, at the same place, is the same quantity in a unit DWR states.
For a sample of well-supported labeled cells this script
  1. asks DWR's image services for the raster value (feet) at the cell centre on 2020-02-01 and on 2024-12-01,
  2. takes the difference (the change from 2020-02 to 2024-12, 4.83 years; my label window is 5.00 years),
  3. compares it with insar_disp_window (the label in the point file's own unit and sign, negative = sinking).
If the point file is in millimetres, label / (change in feet x 304.8) comes out near 1 (about 1.03, because of the longer window).
If it were in feet it would come out near 0.003. The sign of the ratio shows whether the raster and the point file agree on
what negative means. Result: unit_check_results.csv and a printed verdict.
Not run by the author against the live service: if a request fails, the script says so and stops without a verdict.
"""
import json, sys, time
import numpy as np, pandas as pd, requests

BASE = "https://gis.water.ca.gov/arcgisimg/rest/services/SAR/Vertical_Displacement_TRE_ALTAMIRA_Total_Since_20150613_{}/ImageServer"
EARLY, LATE = "20200201", "20241201"
N_CELLS, MIN_POINTS, SEED = 200, 5, 0
FT_TO_MM = 304.8

def raster_value(end_date, lon, lat, tries=2):
    url = BASE.format(end_date) + "/identify"
    params = {"geometry": json.dumps({"x": float(lon), "y": float(lat), "spatialReference": {"wkid": 4326}}),
              "geometryType": "esriGeometryPoint", "returnGeometry": "false", "returnCatalogItems": "false", "f": "json"}
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60); r.raise_for_status(); j = r.json()
            if "error" in j: raise RuntimeError(j["error"])
            v = j.get("value")
            return float(v) if v not in (None, "NoData", "") else np.nan
        except Exception as e:
            err = e; time.sleep(1.5)
    raise RuntimeError(f"request failed for {url}: {err}")

def verdict(ratio_median, rho, share_same_sign):
    if rho < 0.7: return "INCONCLUSIVE (the label and the raster change do not track each other closely enough)"
    if 0.5 < abs(ratio_median) < 2.0: unit = "consistent with MILLIMETRES"
    elif 0.0016 < abs(ratio_median) < 0.0066: unit = "consistent with FEET"
    else: unit = "neither millimetres nor feet (ratio %.4g)" % ratio_median
    sign = "same sign convention as the raster" if ratio_median > 0 and share_same_sign > 0.9 else "OPPOSITE or unclear sign convention relative to the raster"
    return f"{unit}; {sign}"

if __name__ == "__main__":
    d = pd.read_csv("data/subsidence_dataset_processed.csv")
    d = d[d["insar_disp_window"].notna() & (d["insar_n_points"] >= MIN_POINTS)]
    s = d.sample(N_CELLS, random_state=SEED).reset_index(drop=True)
    rows = []
    for i, r in s.iterrows():
        a, b = raster_value(EARLY, r.longitude, r.latitude), raster_value(LATE, r.longitude, r.latitude)
        rows.append((r.grid_id, r.longitude, r.latitude, r.insar_n_points, r.insar_disp_window, a, b))
        time.sleep(0.05)
        if (i + 1) % 25 == 0: print(f"  {i + 1} of {N_CELLS} cells queried")
    t = pd.DataFrame(rows, columns=["grid_id", "lon", "lat", "n_points", "label_file_unit", "raster_ft_2020_02", "raster_ft_2024_12"])
    t["raster_change_ft"] = t["raster_ft_2024_12"] - t["raster_ft_2020_02"]
    t.to_csv("unit_check_results.csv", index=False)
    ok = t.dropna(subset=["raster_change_ft"])
    print(f"\n{len(ok)} of {len(t)} cells have raster values at both dates")
    if len(ok) < 30: sys.exit("too few cells with raster values to judge; check unit_check_results.csv")
    big = ok[ok["raster_change_ft"].abs() > 0.2]                       # skip cells that barely moved: the ratio is noise there
    ratio = big["label_file_unit"] / (big["raster_change_ft"] * FT_TO_MM)
    rho = ok[["label_file_unit", "raster_change_ft"]].corr(method="spearman").iloc[0, 1]
    same = float((np.sign(big["label_file_unit"]) == np.sign(big["raster_change_ft"])).mean())
    print(f"Spearman correlation, label vs raster change: {rho:.3f}")
    print(f"cells with |raster change| > 0.2 ft: {len(big)}; median of label / (change in feet x 304.8): {ratio.median():.3f} (IQR {ratio.quantile(.25):.3f} to {ratio.quantile(.75):.3f})")
    print(f"share of those cells where label and raster change have the same sign: {same:.2f}")
    print("\nVERDICT:", verdict(float(ratio.median()), float(rho), same))
