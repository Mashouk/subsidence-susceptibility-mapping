"""
Builds one grid-cell CSV (500 m cells) for the Tulare-Corcoran subsidence project.
Run from the repo root:  python build_dataset.py
Needs: pandas numpy geopandas shapely rasterio pyproj scipy requests py3dep
One run downloads everything (groundwater, InSAR, land use, soil, terrain) and builds the CSV.
Check the printed InSAR choice and the missing-value report at the end.
"""
import os, re, numpy as np, pandas as pd, geopandas as gpd, rasterio, requests
from shapely import wkt
from pyproj import Transformer
from scipy.spatial import cKDTree

# ============================ SETTINGS ============================
XMIN, YMIN, XMAX, YMAX = -119.70, 36.00, -119.30, 36.35   # lon/lat box
CELL = 500                                                 # meters
CDL_YEARS = [2016, 2017, 2018, 2019]                       # predictor window
PRED_START, PRED_END = "2016-01-01", "2019-12-31"
COUNTIES = ["Tulare", "Kings"]                             # Corcoran is in Kings County
MAX_WELLS = 300                                            # keep the download small
GW_CSV = "data/dwr_groundwater_sample.csv"
INSAR_ENABLED = False   # set True once the InSAR source is sorted out
INSAR_URL = ""      # optional: paste a direct file URL to skip the automatic search
INSAR_FILE = "data/insar_displacement.tif"
OUT = "data/subsidence_dataset.csv"
os.makedirs("data", exist_ok=True)
# ==================================================================


# ---------- 0a. groundwater download (CA DWR open data, CKAN API) ----------
import json, zipfile, io
CKAN = "https://data.cnra.ca.gov/api/3/action"

def ckan_fetch(resource_id, filters=None, page=30000):
    rows, offset = [], 0
    while True:
        p = {"resource_id": resource_id, "limit": page, "offset": offset}
        if filters: p["filters"] = json.dumps(filters)
        res = requests.get(f"{CKAN}/datastore_search", params=p, timeout=180).json()["result"]
        rows += res["records"]
        if len(res["records"]) < page: break
        offset += page
    return pd.DataFrame(rows)

if not os.path.exists(GW_CSV):
    pkg = requests.get(f"{CKAN}/package_show", params={"id": "periodic-groundwater-level-measurements"}, timeout=60).json()["result"]
    print("Groundwater license:", pkg.get("license_title"))
    find = lambda w: next(r["id"] for r in pkg["resources"] if w in r["name"].lower())
    stations = pd.concat([ckan_fetch(find("stations"), {"county_name": c}) for c in COUNTIES])
    sites = stations["site_code"].dropna().unique()[:MAX_WELLS].tolist()
    meas = pd.concat([ckan_fetch(find("measurements"), {"site_code": sites[i:i+50]}) for i in range(0, len(sites), 50)])
    (meas.merge(stations[["site_code", "latitude", "longitude", "well_use", "county_name"]], on="site_code", how="left")
         .to_csv(GW_CSV, index=False))
    print("Saved", GW_CSV)

# ---------- 0b. InSAR download (DWR subsidence data; picks a likely raster automatically) ----------
if INSAR_ENABLED:
    if os.path.exists(INSAR_FILE):                       # discard a bad file from an earlier run
        with open(INSAR_FILE, "rb") as f:
            head = f.read(4)
        if head not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            print("Existing InSAR file is not a valid GeoTIFF - deleting and re-downloading.")
            os.remove(INSAR_FILE)
    if not os.path.exists(INSAR_FILE):
        url = INSAR_URL
        if not url:
            res = requests.get(f"{CKAN}/package_search", params={"q": "InSAR subsidence", "rows": 10}, timeout=60).json()["result"]["results"]
            cands = []
            for p in res:
                for r in p["resources"]:
                    fmt = (r.get("format") or "").lower(); name = (r.get("name") or "").lower()
                    if fmt in ("tif", "tiff", "geotiff", "zip"):
                        score = sum(k in name for k in ["vertical", "displacement", "total", "cumulative"])
                        cands.append((score, p["title"], r["name"], r["url"], p.get("license_title")))
            cands.sort(key=lambda c: -c[0])
            print("InSAR candidates found:")
            for c in cands[:8]: print("  ", c[1], "|", c[2], "|", c[3])
            if not cands:
                raise SystemExit("No InSAR raster found automatically. Paste a direct URL into INSAR_URL and rerun.")
            url = cands[0][3]
            print("USING:", cands[0][1], "|", cands[0][2], "| license:", cands[0][4])
        resp = requests.get(url, timeout=900)
        raw = resp.content
        print("Downloaded", len(raw) // 1024, "KB, content-type:", resp.headers.get("content-type"))
        if raw[:2] == b"PK":                                   # a zip, whatever the URL says
            z = zipfile.ZipFile(io.BytesIO(raw))
            print("Zip contains:", z.namelist()[:15])
            tifs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tifs:
                raise SystemExit("That zip has no .tif inside (see list above). It may be points/shapefile data. "
                                 "Pick a different file from the candidates and paste its URL into INSAR_URL.")
            open(INSAR_FILE, "wb").write(z.read(tifs[0]))
        elif raw[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):   # real GeoTIFF
            open(INSAR_FILE, "wb").write(raw)
        else:
            print("First bytes of the download:", raw[:200])
            raise SystemExit("The download is NOT a GeoTIFF (probably a web page, CSV or shapefile). "
                             "Pick a different file from the candidate list and paste its URL into INSAR_URL.")
        print("Saved", INSAR_FILE)

to5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
to4326 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

# ---------- 1. grid ----------
x1, y1 = to5070.transform(XMIN, YMIN)
x2, y2 = to5070.transform(XMAX, YMAX)
gx, gy = np.meshgrid(np.arange(x1 + CELL/2, x2, CELL), np.arange(y1 + CELL/2, y2, CELL))
grid = pd.DataFrame({"x5070": gx.ravel(), "y5070": gy.ravel()})
grid["longitude"], grid["latitude"] = to4326.transform(grid.x5070.values, grid.y5070.values)
grid["grid_id"] = [f"TC-{i:05d}" for i in range(len(grid))]
print("Grid cells:", len(grid))

def sample_raster(path, x, y, src_crs="EPSG:5070"):
    with rasterio.open(path) as src:
        px, py = Transformer.from_crs(src_crs, src.crs, always_xy=True).transform(x, y)
        vals = np.array([v[0] for v in src.sample(zip(px, py))], dtype="float64")
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    return vals

# ---------- 2. land use: CDL for each year + readable names ----------
CDL_NAMES = {1:"corn",2:"cotton",3:"rice",24:"winter_wheat",36:"alfalfa",37:"other_hay",
 54:"tomatoes",61:"fallow_idle",66:"cherries",67:"peaches",69:"grapes",72:"citrus",74:"pecans",
 75:"almonds",76:"walnuts",111:"open_water",121:"developed_open",122:"developed_low",
 123:"developed_med",124:"developed_high",141:"deciduous_forest",142:"evergreen_forest",
 143:"mixed_forest",152:"shrubland",176:"grass_pasture",190:"woody_wetlands",
 195:"herbaceous_wetlands",204:"pistachios",212:"oranges"}
PERENNIAL = {66,67,69,72,74,75,76,204,212}   # tree/vine crops: fixed water demand
CROPS = {1,2,3,24,36,37,54,66,67,69,72,74,75,76,204,212}

def get_cdl(year):
    path = f"data/cdl_{year}.tif"
    if not os.path.exists(path):
        r = requests.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                         params={"year": year, "bbox": f"{x1},{y1},{x2},{y2}"}, timeout=120)
        url = re.search(r"<returnURL>(.*?)</returnURL>", r.text).group(1)
        open(path, "wb").write(requests.get(url, timeout=300).content)
    return path

codes = {}
for yr in CDL_YEARS:
    codes[yr] = sample_raster(get_cdl(yr), grid.x5070.values, grid.y5070.values)
    grid[f"cdl_code_{yr}"] = codes[yr]
last = codes[CDL_YEARS[-1]]
grid["land_use_name_last_year"] = pd.Series(last).map(CDL_NAMES).fillna("other").values
code_arr = np.vstack([codes[y] for y in CDL_YEARS])
grid["n_years_cropped"] = np.isin(code_arr, list(CROPS)).sum(axis=0)
grid["n_years_fallow"] = (code_arr == 61).sum(axis=0)
grid["is_perennial_crop"] = np.isin(last, list(PERENNIAL)).astype(int)
grid["land_use_changed"] = (np.nanmax(code_arr, axis=0) != np.nanmin(code_arr, axis=0)).astype(int)

# ---------- 3. soil: polygons + more attributes ----------
wkt_box = f"POLYGON(({XMIN} {YMIN},{XMAX} {YMIN},{XMAX} {YMAX},{XMIN} {YMAX},{XMIN} {YMIN}))"
def sda(q):
    r = requests.post("https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
                      json={"query": q, "format": "JSON+COLUMNNAME"}, timeout=300)
    t = r.json()["Table"]
    return pd.DataFrame(t[1:], columns=t[0])

polys = sda(f"""SELECT mukey, mupolygongeo.STAsText() AS geom FROM mupolygon
WHERE mupolygonkey IN (SELECT * FROM SDA_Get_Mupolygonkey_from_intersection_with_WktWgs84('{wkt_box}'))""")
polys_gdf = gpd.GeoDataFrame(polys[["mukey"]], geometry=polys.geom.apply(wkt.loads), crs="EPSG:4326")
polys_gdf["mukey"] = polys_gdf["mukey"].astype(str)

attrs = sda(f"""
SELECT mu.mukey, c.comppct_r, c.drainagecl, c.hydgrp, c.taxclname,
       ch.claytotal_r, ch.sandtotal_r, ch.silttotal_r, ch.om_r, ch.ksat_r, ch.awc_r
FROM mapunit mu JOIN component c ON c.mukey = mu.mukey
LEFT JOIN chorizon ch ON ch.cokey = c.cokey AND ch.hzdept_r = 0
WHERE mu.mukey IN (SELECT DISTINCT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt_box}'))""")
attrs["comppct_r"] = pd.to_numeric(attrs["comppct_r"], errors="coerce")
attrs = attrs.sort_values("comppct_r", ascending=False).drop_duplicates("mukey")
attrs["mukey"] = attrs["mukey"].astype(str)
soil_cols = ["drainagecl","hydgrp","taxclname","claytotal_r","sandtotal_r","silttotal_r","om_r","ksat_r","awc_r"]
for c in soil_cols[3:]:
    attrs[c] = pd.to_numeric(attrs[c], errors="coerce")

pts = gpd.GeoDataFrame(grid[["grid_id"]], geometry=gpd.points_from_xy(grid.longitude, grid.latitude), crs="EPSG:4326")
j = gpd.sjoin(pts, polys_gdf, how="left", predicate="within").drop_duplicates("grid_id")
j = j.merge(attrs[["mukey"] + soil_cols], on="mukey", how="left")
grid = grid.merge(j[["grid_id", "mukey"] + soil_cols], on="grid_id", how="left")
grid = grid.rename(columns={"claytotal_r": "clay_pct", "sandtotal_r": "sand_pct", "silttotal_r": "silt_pct",
                            "om_r": "organic_matter_pct", "ksat_r": "ksat_um_per_s",
                            "awc_r": "available_water_capacity"})

# ---------- 4. terrain: elevation + slope from a 3DEP DEM ----------
try:
    import py3dep
    dem = py3dep.get_dem((XMIN, YMIN, XMAX, YMAX), resolution=30)   # meters, EPSG:4326
    dem = dem.rio.reproject("EPSG:5070")
    dem.rio.to_raster("data/dem_30m.tif")
    grid["elevation_m"] = sample_raster("data/dem_30m.tif", grid.x5070.values, grid.y5070.values)
    with rasterio.open("data/dem_30m.tif") as src:
        a = src.read(1).astype(float); a[a == src.nodata] = np.nan
        gyy, gxx = np.gradient(a, src.res[1], src.res[0])
        np.save("data/_slope.npy", np.degrees(np.arctan(np.hypot(gxx, gyy))))
    # sample slope by row/col of the DEM
    with rasterio.open("data/dem_30m.tif") as src:
        slope = np.load("data/_slope.npy")
        rc = [src.index(px, py) for px, py in zip(grid.x5070, grid.y5070)]
        grid["slope_deg"] = [slope[r, c] if 0 <= r < slope.shape[0] and 0 <= c < slope.shape[1] else np.nan for r, c in rc]
    os.remove("data/_slope.npy")
except Exception as e:
    print("Terrain step skipped:", e)

# ---------- 5. groundwater: richer per-well stats, nearest well to each cell ----------
gw = pd.read_csv(GW_CSV, parse_dates=["msmt_date"])
gw = gw[(gw.msmt_date >= PRED_START) & (gw.msmt_date <= PRED_END)]
gw = gw.dropna(subset=["gse_gwe", "latitude", "longitude"])
gw["depth_m"] = gw.gse_gwe * 0.3048          # DWR depth-to-water is in feet

def slope_m_per_yr(g):
    if len(g) < 3: return np.nan
    t = (g.msmt_date - g.msmt_date.min()).dt.days / 365.25
    return np.polyfit(t, g.depth_m, 1)[0]    # positive = water level falling

by = gw.sort_values("msmt_date").groupby("site_code")
wells = pd.DataFrame({
    "gw_decline_m_per_yr_2016_2019": by.apply(slope_m_per_yr),
    "gw_mean_depth_m": by.depth_m.mean(),
    "gw_latest_depth_m": by.depth_m.last(),
    "gw_depth_std_m": by.depth_m.std(),
    "gw_n_measurements": by.depth_m.count(),
}).reset_index().merge(gw.drop_duplicates("site_code")[["site_code", "latitude", "longitude"]], on="site_code")
wells = wells.dropna(subset=["gw_decline_m_per_yr_2016_2019"])
wx, wy = to5070.transform(wells.longitude.values, wells.latitude.values)
dist, idx = cKDTree(np.c_[wx, wy]).query(np.c_[grid.x5070, grid.y5070])
for c in ["gw_decline_m_per_yr_2016_2019", "gw_mean_depth_m", "gw_latest_depth_m",
          "gw_depth_std_m", "gw_n_measurements"]:
    grid[c] = wells[c].values[idx]
grid["dist_to_nearest_well_m"] = dist.round(0)

# ---------- 6. InSAR outcome + quantile label ----------
if INSAR_ENABLED:
    grid["insar_displacement"] = sample_raster(INSAR_FILE, grid.x5070.values, grid.y5070.values)
    d = grid["insar_displacement"]
    # Subsidence is negative, so the MOST negative cells are the highest susceptibility.
    # If your file uses positive = sinking, change -d to d.
    s = -d
    q = s.quantile([.2, .4, .6, .8]).to_numpy()
    grid["susceptibility_class"] = pd.cut(s, [-np.inf, *q, np.inf],
                                          labels=["very_low", "low", "moderate", "high", "very_high"])
else:
    print("InSAR skipped (INSAR_ENABLED = False): no displacement or susceptibility label in this CSV.")

# ---------- 7. save + report ----------
df = grid.drop(columns=["x5070", "y5070"])
df.to_csv(OUT, index=False)
print(df.shape, "->", OUT, f"({os.path.getsize(OUT)/1e6:.1f} MB)")
print("\nMissing fraction per column:\n", df.isna().mean().round(3))
