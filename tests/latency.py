"""End-to-end latency of the deployed app (Sections 8.5 and 9.1).

Run from the repository root while the stack is up:   python tests/latency.py            (about 3 minutes)
Each interaction is timed from the client call to the finished response (network + Gradio queue + handler + map redraw), and the
handler's own time is read from the app's Prometheus histogram (subsidence_request_seconds) for the same calls, so the two can be
compared. 40 calls per interaction on different random cells; one warm-up call is not counted. Writes tests/latency_result.json and
prints a table to paste into the document. Set APP_URL / METRICS_URL if the stack is not on the default local ports.
"""
import json, os, re, statistics, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from gradio_client import Client

APP = os.environ.get("APP_URL", "http://127.0.0.1:7860")
METRICS = os.environ.get("METRICS_URL", "http://127.0.0.1:8000/metrics")
DATA = os.environ.get("SUBSIDENCE_DATA", "data/v2/subsidence_dataset_processed.csv")
N = int(os.environ.get("N", "40"))
rng = np.random.default_rng(0)
ids = pd.read_csv(DATA).grid_id.to_numpy()
LAYERS = ["Observed class (InSAR 2020-2024)", "Model prediction", "Model confidence"]
c = Client(APP, verbose=False)


def hist(handler):
    t = requests.get(METRICS, timeout=10).text
    g = lambda suffix: float(re.search(rf'^subsidence_request_seconds_{suffix}{{handler="{handler}"}} ([0-9.e+-]+)$', t, re.M).group(1))
    return g("sum"), g("count")


def run(name, handler, fn):
    fn(); s0, n0 = hist(handler)
    ts = []
    for i in range(N):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    s1, n1 = hist(handler)
    ts = sorted(ts)
    q = lambda p: ts[min(len(ts) - 1, int(round(p * (len(ts) - 1))))]
    return {"interaction": name, "calls": N, "median_s": statistics.median(ts), "p95_s": q(0.95), "max_s": ts[-1],
            "handler_mean_s_from_prometheus_histogram": (s1 - s0) / max(n1 - n0, 1)}


pick = lambda: str(rng.choice(ids))
rows = [
    run("search a cell by ID (details, probabilities, map redraw)", "id_search", lambda: c.predict(LAYERS[1], pick(), api_name="/on_id_search")),
    run("look up a cell by latitude/longitude", "latlon_lookup", lambda: c.predict(LAYERS[0], 36.0 + 0.35 * rng.random(), -119.8 + 0.5 * rng.random(), api_name="/on_lookup")),
    run("switch the map layer", "layer_switch", lambda: c.predict(LAYERS[int(rng.integers(3))], pick(), api_name="/on_layer")),
    run("'Try the model' (what-if)", "whatif", lambda: c.predict(pick(), -1.0, 60.0, 63.0, 30.0, 0.5, 70.0, api_name="/whatif")),
]
t = time.perf_counter(); r = requests.get(APP, timeout=30); page = time.perf_counter() - t
out = {"app": APP, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "calls_per_interaction": N, "page_load_s": page, "page_bytes": len(r.content), "rows": rows}
Path("tests").mkdir(exist_ok=True); Path("tests/latency_result.json").write_text(json.dumps(out, indent=2))
print(f"page load {page:.2f} s ({len(r.content) / 1e3:.0f} kB of HTML)\n")
print("| Interaction | Calls | Median (s) | 95th percentile (s) | Max (s) | Handler mean (s) |\n|---|---|---|---|---|---|")
for x in rows:
    print(f"| {x['interaction']} | {x['calls']} | {x['median_s']:.2f} | {x['p95_s']:.2f} | {x['max_s']:.2f} | {x['handler_mean_s_from_prometheus_histogram']:.2f} |")
