# Gradio app for the subsidence susceptibility project

Put these files in the repository root (next to `data/`):

    app.py  prepare.py  metrics.py  feedback.py  requirements-app.txt  Dockerfile  .dockerignore
    docker-compose.yml  docker-compose.vm.yml  .env.example  prometheus.yml  alerts.yml  grafana/  tests/

The start-here instructions (clean clone, commit, `.env`, ports) are at the top of `README.md`. This file explains the pieces.
`docker-compose.vm.yml` is the hardened version used for the final submission (needs `.env`; only the app is reachable from other
computers). `docker-compose.yml` is the plain version for trying things on one computer.

## Run without Docker

    pip install -r requirements-app.txt
    python app.py
    # app      http://127.0.0.1:7860
    # metrics  http://127.0.0.1:8000/metrics

The first start trains the selected Random Forest (about 5 seconds) and saves it in `artifacts/`.
`prepare.py` prints the cross-validated macro-F1 so you can see it matches the README: **0.4292 on the default v2 label**, 0.3692 on the
v1 label of Milestone 4 (`SUBSIDENCE_DATA=data/subsidence_dataset_processed.csv python app.py`). The model is cached together with the
label version, so changing the data path retrains automatically.
Add `artifacts/` and `feedback/` to `.gitignore`: `artifacts/` is generated and `feedback/` holds users' comments.

## Run the app with Prometheus and Grafana (Docker)

    docker compose up --build

| What | Address |
|---|---|
| Gradio app | http://localhost:7860 |
| Raw metrics from the app | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 (Status > Targets should show `subsidence-app` as UP) |
| Grafana | http://localhost:3000 (plain compose file: admin / admin or anonymous view; hardened file: admin and your `GRAFANA_PASSWORD`) |

Grafana opens on the dashboard "Subsidence susceptibility app" because the data source and the dashboard are
provisioned from `grafana/`. Click around in the app for a minute, then watch the panels move (they refresh every 5 s).
Stop with `docker compose down`.

The hardened file (`docker compose -f docker-compose.vm.yml up -d --build`) publishes only port 7860 to other computers; the metrics (8000),
Prometheus (9090) and Grafana (3000) ports are bound to 127.0.0.1, and it refuses to start without `REVIEW_CODE` and `GRAFANA_PASSWORD` in `.env`.

Only the app on its own (no monitoring):

    docker build -t subsidence-ui .
    docker run --rm -p 7860:7860 -p 8000:8000 -v feedback-data:/app/feedback -e REVIEW_CODE=mycode subsidence-ui

### What is measured

| Metric | Meaning | Labels |
|---|---|---|
| `subsidence_cell_requests_total` | Lookups that found a cell | `channel`: map_click, id_search, latlon_lookup |
| `subsidence_invalid_requests_total` | Click, ID or coordinates that matched no cell | `channel` |
| `subsidence_predictions_total` | Predictions shown | `predicted_class`, `prediction_source` (cross_validated or fitted_on_all_training_cells) |
| `subsidence_prediction_confidence` | Histogram of the highest class probability shown | none |
| `subsidence_low_support_views_total` / `subsidence_unlabeled_views_total` | Views of cells with a weak or missing InSAR label | none |
| `subsidence_map_layer_total` | Map layer selections | `layer` |
| `subsidence_whatif_runs_total` / `subsidence_whatif_class_changed_total` | Use of "Try the model" | none |
| `subsidence_feedback_total` | Feedback submitted since the app started | `feedback_type`, `predicted_class` |
| `subsidence_feedback_status` | Feedback in the database by review status (all time) | `status` |
| `subsidence_feedback_rating` | Feedback in the database by rating (all time) | `rating` |
| `subsidence_request_seconds` | Histogram of handler time | `handler` |
| `subsidence_errors_total` | Exceptions raised in a handler | `handler` |
| `subsidence_model_cv_macro_f1`, `subsidence_model_info` | Which model and which label version (`label_version`: v1 or v2) is being served | `label_version` |

Labels only hold short fixed values. Cell IDs and feedback comments are never used as labels (every new label value
creates a new time series in Prometheus). Comments are stored in the feedback database (next section).

## Feedback: submit, review, act

1. **Submit.** In the Map tab, select a cell, open "Send feedback about this cell", choose Good or Bad, say what it is about
   and add a comment (up to 500 characters). It is saved with the cell ID, the model's class and confidence, the observed
   class and the map layer, so you can see exactly what the user was looking at. No name, e-mail or IP address is stored.
2. **Review.** Open the "Review feedback" tab and enter the review code (`REVIEW_CODE`, default `review`; set your own
   before sharing the app: `REVIEW_CODE=mycode docker compose up --build`). The table shows every entry, newest first; filter
   by status. This is a simple access code, not full user accounts.
3. **Act.** Type the feedback number, choose a status (New, Reviewed, Action planned, Done, No action) and write what was
   decided or done, then Save. Typical actions: fix a wrong label or location in the data, reword the explanation in the
   About text, add a warning for a kind of cell, or put a case on the list for the next model update.
4. **Close the loop.** Entries marked "Action planned" or "Done" with a note appear for everyone at the bottom of the About
   tab (only the issue type and your note, never the user's comment or the cell). Grafana shows how many entries wait in
   each status, so a growing "New" bar means the backlog needs attention.
5. **Export.** "Export all feedback as CSV" downloads everything for your report or for a model-update review. The CSV
   keeps the cell ID, the predicted and observed class and the status, so flagged cells are easy to list.

Feedback is **not** used to retrain the model automatically. Users can be wrong, and a few people could push the labels in
a bad direction, so a person decides what to change. The database is `feedback/feedback.db` (SQLite). With Docker it lives in
the `feedback-data` volume, so it survives `docker compose down` and rebuilds; `docker compose down -v` deletes it. To copy the
file out: `docker compose cp subsidence-app:/app/feedback/feedback.db .`

## Alerts (monitoring plan, Section 9.2 to 9.4)

`alerts.yml` holds five rules that Prometheus evaluates every 15 seconds (open http://127.0.0.1:9090/alerts). Each has an expected range and a
`response` annotation that says what the operator does. There is no pager or e-mail; someone has to look at the page.

| Alert | Signal | Fires when |
|---|---|---|
| `SubsidenceAppDown` | availability | the app's metrics endpoint cannot be scraped for 1 minute |
| `SubsidenceHandlerErrors` | errors | any handler raised an exception in the last 10 minutes |
| `SubsidenceSlowResponses` | latency | 95th percentile of handler time above 1 s for 5 minutes |
| `SubsidenceConfidenceShift` | what the model outputs | outside 38% to 88% of the predictions shown in 5 minutes have confidence below 0.4 (at least 40 shown) |
| `SubsidenceNegativeFeedback` | user feedback | more than half of at least 10 feedback entries in 7 days are "Bad" |

To see them work, run a deliberate failure (the stack must be up): `python tests/inject_failure.py db_lock`, `drift` or `down`.

## Tests

`python tests/deployed_tests.py` (24 checks: valid, edge, malformed, integration, security; set `REVIEW_CODE` to the code in `.env`),
`python tests/latency.py` (end-to-end timings from the client) and `python tests/inject_failure.py` (above). Results are written next to the
scripts as JSON. The `*_sandbox_python_app.json` files in `tests/` come from a run of `python app.py` in a development container, not from
the Docker stack; they only show that the scripts work. Run them against your own stack and use those results.

## What the app does

- Map tab: three layers (observed InSAR class, model prediction, model confidence). Click a cell for its class, the
  number of InSAR points behind it, the model's class probabilities and its main inputs. Type a cell ID or a
  latitude/longitude to jump to a cell. "Try the model" changes six inputs and shows how the probabilities move.
  A feedback box saves a Good / Bad rating, the issue and a comment about the selected cell.
- Review feedback tab: the project team reads the feedback, records a decision and exports it (see below).
- About tab: how the classes and predictions are made, the cross-validated score, and the limits from the README.

## Files it reads

- `data/v2/subsidence_dataset_processed.csv` (required, the default) and `data/v2/label_build_log.json` (optional, for the class cut points and the label description); the v1 files in `data/` are used when `SUBSIDENCE_DATA` points to them
- Set `SUBSIDENCE_DATA` or `SUBSIDENCE_ARTIFACTS` to use other paths; set `PORT` for the Gradio port,
  `METRICS_PORT` for the metrics port, `FEEDBACK_DB` for the feedback database file and `REVIEW_CODE` for the Review tab code.
