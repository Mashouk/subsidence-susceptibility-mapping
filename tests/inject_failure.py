"""Deliberate failures for Section 8.5 and the monitoring demonstration in 9.4.

Run from the repository root while the stack is up. Three failures, each reports what the USER saw and whether the monitoring noticed:
    python tests/inject_failure.py db_lock     # a second process holds the feedback database locked while someone submits feedback
    python tests/inject_failure.py drift       # 60 consecutive lookups of cells where the model is unsure (a shift in what the model outputs)
    python tests/inject_failure.py down        # prints the two commands that stop the app container and bring it back (alert SubsidenceAppDown)
Optional environment: APP_URL, METRICS_URL, PROM_URL, REVIEW_CODE (not needed), CONTAINER (default subsidence-app; set CONTAINER= to lock a local
feedback/feedback.db directly instead of through docker exec), FEEDBACK_DB (the local file in that case).
Prometheus evaluates alerts every 15 s, so allow about a minute. Nothing here changes the model or the data.
"""
import json, os, re, subprocess, sys, threading, time
import pandas as pd, requests
from gradio_client import Client

APP = os.environ.get("APP_URL", "http://127.0.0.1:7860")
METRICS = os.environ.get("METRICS_URL", "http://127.0.0.1:8000/metrics")
PROM = os.environ.get("PROM_URL", "http://127.0.0.1:9090")
CONTAINER = os.environ.get("CONTAINER", "subsidence-app")
DB = os.environ.get("FEEDBACK_DB", "feedback/feedback.db")
DATA = os.environ.get("SUBSIDENCE_DATA", "data/v2/subsidence_dataset_processed.csv")
LOCK_SECONDS = 25
LAY = "Observed class (InSAR 2020-2024)"


def metric(name, **labels):
    t = requests.get(METRICS, timeout=10).text
    sel = ",".join(f'{k}="{v}"' for k, v in labels.items())
    m = re.search(rf"^{re.escape(name)}\{{{re.escape(sel)}\}} ([0-9.e+-]+)$", t, re.M)
    return float(m.group(1)) if m else 0.0


def alerts():
    try:
        return {a["labels"]["alertname"]: a["state"] for a in requests.get(f"{PROM}/api/v1/alerts", timeout=5).json()["data"]["alerts"]}
    except Exception:
        return None


def watch(name, seconds=120):
    print(f"\nwatching Prometheus for alert {name} (up to {seconds} s) ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < seconds:
        a = alerts()
        if a is None:
            print("  Prometheus is not reachable at", PROM, "- look at the Alerts page yourself later"); return None
        if a.get(name) == "firing":
            print(f"  {name} is FIRING after {time.time() - t0:.0f} s"); return time.time() - t0
        time.sleep(5)
    print(f"  {name} did not fire within {seconds} s (state: {alerts()})"); return None


mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "db_lock":
    c = Client(APP, verbose=False)
    cell = pd.read_csv(DATA).query("insar_n_points >= 5").grid_id.iloc[10]
    err0 = metric("subsidence_errors_total", handler="feedback")
    code = f"import sqlite3,time; c=sqlite3.connect('{DB if not CONTAINER else 'feedback/feedback.db'}', timeout=1); c.execute('BEGIN EXCLUSIVE'); time.sleep({LOCK_SECONDS})"
    if CONTAINER:
        subprocess.Popen(["docker", "exec", CONTAINER, "python", "-c", code])
    else:
        threading.Thread(target=lambda: subprocess.run([sys.executable, "-c", code]), daemon=True).start()
    time.sleep(2)
    print(f"feedback database locked for {LOCK_SECONDS} s by a second process; submitting feedback for {cell} ...", flush=True)
    t = time.perf_counter()
    out = c.predict(cell, LAY, "Good", "Other", "failure-injection test", api_name="/submit_feedback")
    waited = time.perf_counter() - t
    print(f"\nWHAT THE USER SAW after waiting {waited:.1f} s:\n  {out[0]}")
    print(f"errors recorded for handler=feedback: +{metric('subsidence_errors_total', handler='feedback') - err0:.0f}")
    time.sleep(LOCK_SECONDS)
    ok = c.predict(cell, LAY, "Good", "Other", "failure-injection test, after the lock was released", api_name="/submit_feedback")
    print("after the lock was released the same action gives:\n  ", ok[0][:90])
    watch("SubsidenceHandlerErrors", 60)
elif mode == "drift":
    c = Client(APP, verbose=False)
    ids = pd.read_csv(DATA).grid_id.sample(frac=1, random_state=1).tolist()
    low = []
    print("step 1: screening cells with the what-if handler (it does not count as a lookup) to find 60 where the model is unsure ...", flush=True)
    for gid in ids:
        msg = c.predict(gid, -1.0, 60.0, 63.0, 30.0, 0.5, 70.0, api_name="/whatif")[0]
        m = re.search(r"Cell as it is: \*\*[^*]+\*\* \(([0-9.]+)\)", msg)
        if m and float(m.group(1)) < 0.4:
            low.append(gid)
            if len(low) >= 60:
                break
    print(f"step 2: looking up those {len(low)} cells, one after the other, as a user who only ever opens uncertain places would", flush=True)
    for gid in low:
        c.predict(LAY, gid, api_name="/on_id_search")
    print("share below 0.4 over the last 5 minutes as Prometheus sees it:", end=" ")
    try:
        time.sleep(6)
        q = requests.get(f"{PROM}/api/v1/query", params={"query": 'sum(increase(subsidence_prediction_confidence_bucket{le="0.4"}[5m])) / sum(increase(subsidence_prediction_confidence_count[5m]))'}, timeout=5).json()
        print(q["data"]["result"][0]["value"][1] if q["data"]["result"] else "no data yet")
    except Exception:
        print("Prometheus not reachable")
    watch("SubsidenceConfidenceShift", 120)
elif mode == "down":
    print(f"1) docker stop {CONTAINER}      (then open {APP} in a browser: the page does not load; open {PROM}/alerts after 1-2 minutes: SubsidenceAppDown fires)")
    print(f"2) docker start {CONTAINER}     (the alert resolves; the feedback database is in a volume, so nothing is lost)")
else:
    sys.exit(__doc__)
