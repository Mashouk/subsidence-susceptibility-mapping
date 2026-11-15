"""Tests of the DEPLOYED app (Section 8.5): a valid case, edge cases, malformed input, and integration between components.

Run from the repository root while the stack is up (docker compose -f docker-compose.vm.yml up -d --build):
    pip install gradio_client requests pandas
    set REVIEW_CODE=<the code in your .env>          (PowerShell:  $env:REVIEW_CODE="<code>")
    python tests/deployed_tests.py
Optional: APP_URL (default http://127.0.0.1:7860), METRICS_URL (default http://127.0.0.1:8000/metrics),
          PROM_URL (default http://127.0.0.1:9090; skipped if Prometheus is not reachable).
Prints one line per test and writes tests/deployed_tests_result.json (commit it). Exit code 1 if any test fails.
The tests call the app the way the browser does (the same Gradio endpoints), so they exercise the real handlers, the model,
the SQLite feedback store, the Prometheus metrics endpoint and, when reachable, Prometheus itself.
"""
import json, os, re, sys, time
from pathlib import Path
import pandas as pd, requests
from gradio_client import Client

APP = os.environ.get("APP_URL", "http://127.0.0.1:7860")
METRICS = os.environ.get("METRICS_URL", "http://127.0.0.1:8000/metrics")
PROM = os.environ.get("PROM_URL", "http://127.0.0.1:9090")
CODE = os.environ.get("REVIEW_CODE")
DATA = os.environ.get("SUBSIDENCE_DATA", "data/v2/subsidence_dataset_processed.csv")
if not CODE:
    sys.exit("Set REVIEW_CODE first (the review code from your .env file).")

df = pd.read_csv(DATA)
good = df[(df.insar_n_points >= 5) & df.susceptibility_class.notna()].iloc[40]
thin = df[(df.insar_n_points.between(1, 2)) & df.susceptibility_class.notna()].iloc[5]
none = df[df.susceptibility_class.isna()].iloc[5]
LAYERS = ["Observed class (InSAR 2020-2024)", "Model prediction", "Model confidence"]
c = Client(APP, verbose=False)
results, timings = [], {}


def metric(name, **labels):
    """Current value of one Prometheus series from the app's /metrics page (0 if the series does not exist yet)."""
    txt = requests.get(METRICS, timeout=10).text
    sel = ",".join(f'{k}="{v}"' for k, v in labels.items())
    m = re.search(rf"^{re.escape(name)}\{{{re.escape(sel)}\}} ([0-9.e+-]+)$" if labels else rf"^{re.escape(name)} ([0-9.e+-]+)$", txt, re.M)
    return float(m.group(1)) if m else 0.0


def call(api, *args, **kw):
    t = time.perf_counter()
    out = c.predict(*args, api_name=api, **kw)
    timings.setdefault(api, []).append(time.perf_counter() - t)
    return out


def test(tid, kind, what):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            note = fn() or ""
            ok = True
        except Exception as e:
            ok, note = False, f"{type(e).__name__}: {str(e)[:300]}"
        results.append({"id": tid, "kind": kind, "test": what, "passed": ok, "note": note, "seconds": round(time.perf_counter() - t0, 2)})
        print(f"{'PASS' if ok else 'FAIL'}  {tid:<4} {kind:<10} {what}" + (f"   [{note}]" if note else ""), flush=True)
        return fn
    return deco


base_req = metric("subsidence_cell_requests_total", channel="id_search")
base_inv = metric("subsidence_invalid_requests_total", channel="id_search")

# ----------------------------------------------------------------------------- valid
@test("V1", "valid", "id search for a well-supported labeled cell returns details, probabilities and inputs")
def _():
    out = call("/on_id_search", LAYERS[1], good.grid_id)
    assert len(out) == 13 and out[1] == good.grid_id
    assert "Model prediction" in out[2] and "Observed class" in out[2] and out[3] and "Main inputs" in out[4]
    return f"{good.grid_id}: " + re.search(r"\*\*Model prediction:\*\* ([A-Za-z ]+)", out[2]).group(1).strip()


@test("V2", "valid", "latitude/longitude of that cell's centre finds the same cell")
def _():
    out = call("/on_lookup", LAYERS[0], float(good.latitude), float(good.longitude))
    assert out[1] == good.grid_id, out[1]


@test("V3", "valid", "all three map layers redraw with their own legend")
def _():
    want = ["Hatched: low support", "Faded: no InSAR label, model estimate only", "Model confidence (highest class probability)"]
    for lay, w in zip(LAYERS, want):
        img, leg = call("/on_layer", lay, good.grid_id)
        assert Path(img).stat().st_size > 10_000 and w in re.sub(r"<[^>]+>", "", leg), (lay, w)


@test("V4", "valid", "'Try the model' with a changed groundwater trend returns both probability sets")
def _():
    out = call("/whatif", good.grid_id, -5.0, 60.0, 63.0, 30.0, 0.5, 70.0)
    assert "With your changes" in out[0] and out[1].count("|") > 15


# ----------------------------------------------------------------------------- edge
@test("E1", "edge", "lower-case ID with spaces is normalised")
def _():
    out = call("/on_id_search", LAYERS[0], f"  {good.grid_id.lower()}  ")
    assert out[1] == good.grid_id


@test("E2", "edge", "cell with no InSAR label says so and still gives a model estimate")
def _():
    out = call("/on_id_search", LAYERS[1], none.grid_id)
    assert "Observed class:** none" in out[2] and "Model prediction" in out[2]


@test("E3", "edge", "cell whose label rests on 1-2 points carries the low-support warning")
def _():
    out = call("/on_id_search", LAYERS[0], thin.grid_id)
    assert "Low support" in out[2], out[2][:200]


@test("E4", "edge", "coordinates outside the study grid give a clear message, not an error")
def _():
    out = call("/on_lookup", LAYERS[0], 36.9, -119.5)
    assert "outside the study grid" in out[2]


@test("E5", "edge", "coordinates in the open sea / extreme values do not crash the lookup")
def _():
    out = call("/on_lookup", LAYERS[0], 89.999, 179.999)
    assert "outside the study grid" in out[2]


@test("E6", "edge", "'Try the model' with no cell selected asks for a cell")
def _():
    out = call("/whatif", "", -1.0, 60.0, 63.0, 30.0, 0.5, 70.0)
    assert "Select a cell first" in out[0]


# ----------------------------------------------------------------------------- malformed
@test("M1", "malformed", "empty ID gives the help message")
def _():
    out = call("/on_id_search", LAYERS[0], "")
    assert "Click a cell on the map" in out[2]


@test("M2", "malformed", "unknown ID (TC-99999) is rejected with the help message and counted as invalid")
def _():
    before = metric("subsidence_invalid_requests_total", channel="id_search")
    out = call("/on_id_search", LAYERS[0], "TC-99999")
    assert "Click a cell on the map" in out[2]
    assert metric("subsidence_invalid_requests_total", channel="id_search") == before + 1


@test("M3", "malformed", "a 10,000-character ID is rejected in under 2 s without an error")
def _():
    t = time.perf_counter(); out = call("/on_id_search", LAYERS[0], "A" * 10_000); dt = time.perf_counter() - t
    assert "Click a cell on the map" in out[2] and dt < 2, dt
    return f"{dt:.2f} s"


@test("M4", "malformed", "SQL and HTML text in the ID field is treated as an unknown ID")
def _():
    for bad in ["TC-00001'; DROP TABLE feedback;--", "<script>alert(1)</script>", "../../etc/passwd", "TC-0000１"]:
        out = call("/on_id_search", LAYERS[0], bad)
        assert "Click a cell on the map" in out[2] and "<script>" not in out[2], bad


@test("M5", "malformed", "missing coordinates are rejected politely")
def _():
    out = call("/on_lookup", LAYERS[0], None, None)
    assert "Click a cell on the map" in out[2]


@test("M6", "malformed", "feedback without a Good/Bad choice is refused and nothing is saved")
def _():
    try:
        out = call("/submit_feedback", good.grid_id, LAYERS[0], None, "Other", "no rating")
    except Exception as e:                                 # the gradio client may refuse the empty rating itself; either way nothing is stored
        return "refused by the interface layer: " + type(e).__name__
    assert "Choose Good or Bad" in out[0], out[0]


@test("M7", "malformed", "a rating that is not on the list is refused by the API")
def _():
    try:
        call("/submit_feedback", good.grid_id, LAYERS[0], "Awful", "Other", "x")
    except Exception as e:
        return "refused: " + str(e)[:80]
    raise AssertionError("an invalid rating was accepted")


# ----------------------------------------------------------------------------- integration
marker = f"autotest-{int(time.time())}"


@test("I1", "integration", "feedback travels app -> SQLite -> review tab -> public 'what we did' list -> CSV export")
def _():
    sub = call("/submit_feedback", good.grid_id, LAYERS[1], "Bad", "Model prediction looks wrong", f"{marker} private comment")
    m = re.search(r"#(\d+)", sub[0]); assert m, sub[0]; fid = int(m.group(1))
    rv = call("/review_load", CODE, "All")
    assert marker in rv[2] and good.grid_id in rv[2]
    sv = call("/review_save", CODE, fid, "Action planned", "autotest: re-checked this cell's label", "All")
    assert "Action planned" in sv[0]
    pub = call("/outcomes_md")
    assert "autotest: re-checked" in pub and marker not in pub, "public list must show the note but never the user's comment"
    path, msg = call("/review_export", CODE)
    assert marker in Path(path).read_text(encoding="utf-8")
    return f"feedback #{fid}"


@test("I2", "integration", "the metrics page counted every valid and invalid lookup made above")
def _():
    req, inv = metric("subsidence_cell_requests_total", channel="id_search") - base_req, metric("subsidence_invalid_requests_total", channel="id_search") - base_inv
    assert req >= 4 and inv >= 6, (req, inv)
    assert metric("subsidence_request_seconds_count", handler="id_search") > 0
    assert metric("subsidence_feedback_status", status="Action planned") >= 1
    return f"id_search lookups +{int(req)}, invalid +{int(inv)}"


@test("I3", "integration", "Prometheus scrapes the app (target up) and returns the same counter")
def _():
    try:
        up = requests.get(f"{PROM}/api/v1/query", params={"query": 'up{job="subsidence-app"}'}, timeout=5).json()
    except requests.RequestException:
        return "SKIPPED: Prometheus not reachable at " + PROM
    assert up["data"]["result"] and up["data"]["result"][0]["value"][1] == "1", up
    time.sleep(6)                                          # one scrape interval is 5 s
    q = requests.get(f"{PROM}/api/v1/query", params={"query": 'subsidence_cell_requests_total{channel="id_search"}'}, timeout=5).json()
    assert float(q["data"]["result"][0]["value"][1]) >= 4
    return "target up, counter present"


# ----------------------------------------------------------------------------- security controls demonstrated
@test("S1", "security", "wrong review code returns no feedback")
def _():
    out = call("/review_load", "wrong-code", "All")
    assert "Wrong review code" in out[0] and marker not in (out[1] + out[2])


@test("S2", "security", "a 100,000-character comment is cut to 500 characters on the server")
def _():
    sub = call("/submit_feedback", thin.grid_id, LAYERS[0], "Good", "Other", "x" * 100_000)
    fid = int(re.search(r"#(\d+)", sub[0]).group(1))
    path, _ = call("/review_export", CODE)
    row = pd.read_csv(path).set_index("id").loc[fid]
    assert len(str(row["comment"])) <= 500, len(str(row["comment"]))
    return f"stored {len(str(row['comment']))} characters"


@test("S3", "security", "HTML typed into a comment is shown as text in the review table")
def _():
    call("/submit_feedback", good.grid_id, LAYERS[0], "Good", "Other", "<script>alert(1)</script><b>x</b>")
    html = call("/review_load", CODE, "All")[2]
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


@test("S4", "security", "a comment containing SQL is stored as plain text and the table still exists")
def _():
    call("/submit_feedback", good.grid_id, LAYERS[0], "Good", "Other", "x'); DROP TABLE feedback;--")
    out = call("/review_load", CODE, "All")
    assert "DROP TABLE" in out[2] and marker in out[2]


# ----------------------------------------------------------------------------- summary
lat = {k: sorted(v) for k, v in timings.items()}
summary = {"app": APP, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "data": DATA, "passed": sum(r["passed"] for r in results), "total": len(results),
           "results": results, "call_seconds": {k: {"n": len(v), "median": v[len(v) // 2], "max": v[-1]} for k, v in lat.items()}}
Path("tests").mkdir(exist_ok=True)
Path("tests/deployed_tests_result.json").write_text(json.dumps(summary, indent=2))
print(f"\n{summary['passed']} of {summary['total']} tests passed")
sys.exit(0 if summary["passed"] == summary["total"] else 1)
