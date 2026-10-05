# Downloads DWR's InSAR vertical-displacement POINT data and shows what is inside.
import requests, zipfile, io, os
URL = "https://data.cnra.ca.gov/dataset/5e2d49e1-9ed0-425e-9f3e-2cda4a213c26/resource/6e855ae7-f365-4d40-984f-3c016b54f69d/download/verticaldisplacementpointdata.zip"
os.makedirs("data/insar_points", exist_ok=True)
r = requests.get(URL, timeout=900)
print("Downloaded", len(r.content) // 1024, "KB, content-type:", r.headers.get("content-type"))
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("data/insar_points")
print("Files in zip:")
for n in z.namelist()[:40]:
    print("  ", n)
import pandas as pd, geopandas as gpd
for n in z.namelist():
    p = os.path.join("data/insar_points", n)
    low = n.lower()
    try:
        if low.endswith(".csv"):
            d = pd.read_csv(p, nrows=5); print("\nCSV", n, "\ncolumns:", list(d.columns)[:60]); print(d.head(3))
        elif low.endswith(".shp"):
            d = gpd.read_file(p, rows=5); print("\nSHP", n, "\ncolumns:", list(d.columns)[:60], "\ncrs:", d.crs); print(d.head(3))
    except Exception as e:
        print("Could not read", n, e)
