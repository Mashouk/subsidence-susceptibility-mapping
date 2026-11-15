"""metrics.py - Prometheus metrics for the Gradio app (same pattern as the Module 8 warm-up).

start() opens a small HTTP server on port 8000 (override with METRICS_PORT). Prometheus scrapes
http://<app>:8000/metrics and Grafana draws the numbers. Labels only hold short fixed values (class names,
channels, handler names). Never put a cell ID or free text in a label: every new value makes a new time series.
"""
import functools
import os
import time

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

CHANNELS = ("map_click", "id_search", "latlon_lookup")
HANDLERS = ("map_click", "id_search", "latlon_lookup", "layer_switch", "whatif", "feedback", "review")

REQUESTS = Counter("subsidence_cell_requests_total", "Cell lookups that found a grid cell", ["channel"])
INVALID = Counter("subsidence_invalid_requests_total",
                  "Lookups that found no grid cell (click or coordinates outside the grid, unknown ID)", ["channel"])
PREDICTIONS = Counter("subsidence_predictions_total", "Model predictions shown to users",
                      ["predicted_class", "prediction_source"])
CONFIDENCE = Histogram("subsidence_prediction_confidence", "Highest class probability of the predictions shown",
                       buckets=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
LAST_CONFIDENCE = Gauge("subsidence_last_prediction_confidence", "Confidence of the most recent prediction shown")
LOW_SUPPORT = Counter("subsidence_low_support_views_total",
                      "Views of cells whose observed label rests on fewer than 3 InSAR points")
UNLABELED = Counter("subsidence_unlabeled_views_total", "Views of cells with no InSAR label (model estimate only)")
LAYER = Counter("subsidence_map_layer_total", "Map layer selections", ["layer"])
WHATIF = Counter("subsidence_whatif_runs_total", "Runs of the 'Try the model' panel")
WHATIF_CHANGED = Counter("subsidence_whatif_class_changed_total",
                         "What-if runs where the changed inputs gave a different predicted class")
FEEDBACK = Counter("subsidence_feedback_total", "Feedback entries submitted since the app started",
                   ["feedback_type", "predicted_class"])
FEEDBACK_STATUS = Gauge("subsidence_feedback_status", "Feedback entries in the database, by review status "
                        "(all time, survives restarts)", ["status"])
FEEDBACK_RATING = Gauge("subsidence_feedback_rating", "Feedback entries in the database, by rating (all time)",
                        ["rating"])
LATENCY = Histogram("subsidence_request_seconds", "Time spent in each request handler", ["handler"],
                    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5])
ERRORS = Counter("subsidence_errors_total", "Handler calls that raised an error", ["handler"])
MODEL_F1 = Gauge("subsidence_model_cv_macro_f1", "Cross-validated macro-F1 of the model being served")
MODEL_INFO = Info("subsidence_model", "Which model and data the app is serving")


def timed(handler):
    """Decorator: record how long the handler takes and count the errors it raises."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            except Exception:
                ERRORS.labels(handler).inc()
                raise
            finally:
                LATENCY.labels(handler).observe(time.perf_counter() - t0)
        return wrapper
    return deco


def set_feedback_gauges(by_status, by_rating):
    """Copy the feedback database counts into the gauges (called at start and after every submit or review change)."""
    for k, v in by_status.items():
        FEEDBACK_STATUS.labels(k).set(v)
    for k, v in by_rating.items():
        FEEDBACK_RATING.labels(k).set(v)


def init(classes, layers, meta):
    """Create every label combination at 0 so the Grafana panels show series before the first click."""
    for c in CHANNELS:
        REQUESTS.labels(c)
        INVALID.labels(c)
    for k in classes:
        for src in ("cross_validated", "fitted_on_all_training_cells"):
            PREDICTIONS.labels(k, src)
        for t in ("Good", "Bad"):
            FEEDBACK.labels(t, k)
    for lay in layers:
        LAYER.labels(lay)
    for h in HANDLERS:
        LATENCY.labels(h)
        ERRORS.labels(h)
    MODEL_F1.set(meta["oof_macro_f1"])
    MODEL_INFO.info({"model": "RandomForest(n_estimators=100, min_samples_leaf=5)", "sklearn": meta["sklearn"],
                     "data_sha256": meta["data_sha256"][:12]})


def start(port=None):
    port = int(port or os.environ.get("METRICS_PORT", "8000"))
    try:
        start_http_server(port)
    except OSError as e:     # port busy (another Jupyter/app already uses it, or the cell was run twice)
        print(f"Metrics server not started on port {port}: {e}. The app still works. "
              f"Set METRICS_PORT to another number (for example 8001) to get metrics.", flush=True)
