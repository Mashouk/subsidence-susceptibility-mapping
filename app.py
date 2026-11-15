"""app.py - Gradio user interface for the land-subsidence susceptibility project (Tulare-Corcoran, CA).

Run:   python app.py        then open http://127.0.0.1:7860
Needs: data/subsidence_dataset_processed.csv  (already in the repository)
       prepare.py             (trains the selected Random Forest the first time, ~5 s, and caches artifacts/)
"""
import os
import tempfile

import joblib
import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw
from pyproj import Transformer

import feedback
import metrics
import prepare

feedback.init()
META = prepare.build()                                    # trains once, then reuses artifacts/
CLASSES = prepare.CLASSES
LABEL = {c: c.replace("_", " ").capitalize() for c in CLASSES}
PALETTE = ["#8bb8f0", "#4f94e6", "#2878d8", "#1d5aa6", "#103f7e"]      # light -> dark blue (low -> high)
NO_LABEL = "#d9d8d4"
BG = "#fbfbf9"

# ---------------------------------------------------------------- data
df = pd.read_csv(prepare.DATA).merge(pd.read_csv(os.path.join(prepare.OUT, "predictions.csv")), on="grid_id")
MODEL = joblib.load(os.path.join(prepare.OUT, "model.joblib"))
to5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)   # the grid is 500 m squares in EPSG:5070
to4326 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
df["x"], df["y"] = to5070.transform(df.longitude.to_numpy(), df.latitude.to_numpy())
CELL = 500.0
KX = float(np.cos(np.deg2rad(df.latitude.mean())))      # plot x = longitude * KX, plot y = latitude (north is up)
CORNERS = []
for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
    lo, la = to4326.transform((df.x + dx * CELL / 2).to_numpy(), (df.y + dy * CELL / 2).to_numpy())
    CORNERS.append(np.c_[lo * KX, la])
VERTS = np.stack(CORNERS, axis=1)                          # (cells, 4 corners, xy) in plot units
PAD = 0.004
XMIN, XMAX = VERTS[..., 0].min() - PAD, VERTS[..., 0].max() + PAD
YMIN, YMAX = VERTS[..., 1].min() - PAD, VERTS[..., 1].max() + PAD
W, DPI = 1100, 100
LEFT, RIGHT, TOP, BOT = 64, 14, 12, 40                     # pixel margins around the map axes
AW = W - LEFT - RIGHT
AH = int(round(AW * (YMAX - YMIN) / (XMAX - XMIN)))
H = AH + TOP + BOT
BY_ID = df.set_index("grid_id", drop=False)
XY = df[["x", "y"]].to_numpy()
HAS_LABEL = df.susceptibility_class.notna().to_numpy()
LOW_SUPPORT = (HAS_LABEL & (df.insar_n_points < 3)).to_numpy()


def support_level(n):
    n = int(n)
    return "none" if n == 0 else "low" if n < 3 else "medium" if n < 5 else "high"


# ---------------------------------------------------------------- map drawing
LAYERS = ["Observed class (InSAR 2020-2024)", "Model prediction", "Model confidence"]
_base = {}


def _colors(layer):
    n = len(df)
    rgba = np.zeros((n, 4))
    pal = np.array([matplotlib.colors.to_rgba(c) for c in PALETTE])
    grey = matplotlib.colors.to_rgba(NO_LABEL)
    if layer == LAYERS[0]:
        idx = df.susceptibility_class.map({c: i for i, c in enumerate(CLASSES)})
        for i in range(n):
            rgba[i] = grey if np.isnan(idx.iloc[i]) else pal[int(idx.iloc[i])]
    else:
        if layer == LAYERS[1]:
            idx = df.pred_class.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
            rgba = pal[idx].copy()
        else:
            rgba = colormaps["cividis"](np.clip((df.confidence.to_numpy() - 0.2) / 0.6, 0, 1))
        rgba[~HAS_LABEL, 3] = 0.45            # faded = no InSAR label, model estimate only
    return rgba


def base_map(layer):
    if layer not in _base:
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes([LEFT / W, BOT / H, AW / W, AH / H])
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_facecolor(BG)
        lons = [v for v in np.arange(-119.8, -119.19, 0.1) if XMIN < v * KX < XMAX]
        lats = [v for v in np.arange(36.0, 36.36, 0.05) if YMIN < v < YMAX]
        ax.set_xticks([v * KX for v in lons], [f"{abs(v):.1f}\u00b0W" for v in lons], fontsize=9, color="#444")
        ax.set_yticks(lats, [f"{v:.2f}\u00b0N" for v in lats], fontsize=9, color="#444")
        ax.tick_params(length=3, color="#999")
        for sp in ax.spines.values():
            sp.set_color("#bbb")
        ax.add_collection(PolyCollection(VERTS, facecolors=_colors(layer), edgecolors="white", linewidths=0.3))
        if layer == LAYERS[0]:                # hatch = observed label rests on fewer than 3 InSAR points
            plt.rcParams["hatch.linewidth"] = 0.8
            ax.add_collection(PolyCollection(VERTS[LOW_SUPPORT], facecolors="none", edgecolors="white",
                                             linewidths=0, hatch="////"))
        k = 5000 / 111320.0                                    # 5 km in plot units
        x0, y0 = XMAX - PAD - k - 0.012, YMIN + PAD + 0.006
        ax.add_patch(plt.Rectangle((x0 - 0.008, y0 - 0.006), k + 0.016, 0.03, color=BG, zorder=5))
        ax.plot([x0, x0 + k], [y0 + 0.004, y0 + 0.004], color="#222", lw=2.5, zorder=6, solid_capstyle="butt")
        ax.text(x0 + k / 2, y0 + 0.012, "5 km", ha="center", va="bottom", fontsize=9, color="#222", zorder=6)
        fig.canvas.draw()
        _base[layer] = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    return _base[layer]


def to_pixel(px_plot, py_plot):
    return LEFT + (px_plot - XMIN) / (XMAX - XMIN) * AW, TOP + (YMAX - py_plot) / (YMAX - YMIN) * AH


def draw_map(layer, grid_id=None):
    img = base_map(layer).copy()
    if grid_id in BY_ID.index:
        v = VERTS[int(BY_ID.index.get_loc(grid_id))]
        pts = np.array([to_pixel(a, b) for a, b in v])
        c = pts.mean(0)
        pts = [tuple(c + (p - c) * 1.5) for p in pts]             # slightly larger than the cell
        d = ImageDraw.Draw(img)
        d.line(pts + [pts[0]], fill="white", width=5, joint="curve")
        d.line(pts + [pts[0]], fill="#ff6a00", width=3, joint="curve")
    return img


def legend_html(layer):
    def sw(c, t, extra=""):
        return (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                f'<span style="width:18px;height:18px;border-radius:3px;background:{c};{extra}"></span>'
                f'<span style="font-size:14px">{t}</span></div>')
    hatch = "background-image:repeating-linear-gradient(135deg,transparent 0 3px,#fff 3px 4px);"
    if layer == LAYERS[2]:
        stops = ",".join(matplotlib.colors.to_hex(colormaps["cividis"](v)) for v in np.linspace(0, 1, 6))
        body = (f'<div style="font-weight:600;margin-bottom:4px">Model confidence (highest class probability)</div>'
                f'<div style="height:14px;border-radius:3px;background:linear-gradient(90deg,{stops})"></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:13px"><span>0.20 or less</span>'
                f'<span>0.80 or more</span></div>'
                f'<div style="font-size:13px;margin-top:4px">Faded cells have no InSAR label (model estimate only). '
                f'Five classes means a random guess is about 0.20.</div>')
    else:
        items = "".join(sw(c, LABEL[k]) for c, k in zip(PALETTE, CLASSES))
        if layer == LAYERS[0]:
            items += (sw(NO_LABEL, f"No InSAR label ({int((~HAS_LABEL).sum()):,} cells)") +
                      sw("#3f7fd4", f"Hatched: low support, fewer than 3 InSAR points ({int(LOW_SUPPORT.sum()):,} cells)", hatch))
        else:
            items += sw("#8bb8f0;opacity:.45", "Faded: no InSAR label, model estimate only")
        body = '<div style="font-weight:600;margin-bottom:4px">Susceptibility (relative ranking)</div>' + items
    return f'<div style="padding:10px 12px;border:1px solid #d6d6d0;border-radius:8px;background:{BG};color:#222">{body}</div>'


# ---------------------------------------------------------------- cell details
INPUT_ROWS = [("Land use, last year", "land_use_name_last_year", "{}"), ("Soil drainage", "drainagecl", "{}"),
              ("Clay (%)", "clay_pct", "{:.0f}"), ("Elevation (m)", "elevation_m", "{:.1f}"),
              ("Slope (degrees)", "slope_deg", "{:.2f}"),
              ("Groundwater level trend 2016-2019 (m/yr)", "gw_decline_m_per_yr_2016_2019", "{:.2f}"),
              ("Mean groundwater depth (m)", "gw_mean_depth_m", "{:.1f}"),
              ("Distance to nearest well (m)", "dist_to_nearest_well_m", "{:,.0f}")]


def prob_figure(row):
    p = [row["p_" + c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(4.6, 2.3), dpi=110)
    bars = ax.barh([LABEL[c] for c in CLASSES], p, color=PALETTE)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.6, max(p) + 0.1))
    ax.set_xlabel("Model probability")
    for b, v in zip(bars, p):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    if isinstance(row["susceptibility_class"], str):
        i = CLASSES.index(row["susceptibility_class"])
        bars[i].set_edgecolor("#ff6a00")
        bars[i].set_linewidth(2.5)
        ax.set_title("Orange outline = class observed from InSAR", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def describe(grid_id):
    """-> (markdown, probability figure, inputs table, slider values) for one grid cell."""
    r = BY_ID.loc[grid_id]
    n = int(r.insar_n_points)
    lines = [f"### Cell {grid_id}", f"{r.latitude:.4f} N, {abs(r.longitude):.4f} W"]
    if isinstance(r.susceptibility_class, str):
        lines.append(f"**Observed class (InSAR, 2020-2024):** {LABEL[r.susceptibility_class]}  \n"
                     f"Window displacement: {r.subsidence:,.0f} (positive = sinking; units inferred as mm, "
                     f"see About), from **{n} InSAR point{'s' if n != 1 else ''}** "
                     f"(support: {support_level(n)})")
        if n < 3:
            lines.append("> Low support: this label rests on fewer than 3 InSAR points and is the least reliable "
                         "kind in this project.")
    else:
        lines.append("**Observed class:** none. No usable InSAR point falls in this cell, so the model estimate "
                     "below is the only information.")
    lines.append(f"**Model prediction:** {LABEL[r.pred_class]} (probability {r.confidence:.2f})  \n"
                 f"<sub>{r.pred_source}</sub>")
    if r.confidence < 0.4:
        lines.append("> Low confidence: the model is close to indifferent between classes here.")
    if isinstance(r.susceptibility_class, str):
        gap = abs(CLASSES.index(r.pred_class) - CLASSES.index(r.susceptibility_class))
        lines.append("Prediction matches the observed class." if gap == 0 else
                     f"Prediction is {gap} class{'es' if gap > 1 else ''} away from the observed class.")
    rows = "\n".join(f"| {name} | {fmt.format(r[col]) if pd.notna(r[col]) else 'missing'} |"
                     for name, col, fmt in INPUT_ROWS)
    table = "**Main inputs for this cell**\n\n| Input | Value |\n|---|---|\n" + rows
    return "\n\n".join(lines), prob_figure(r), table


SLIDERS = [("gw_decline_m_per_yr_2016_2019", "Groundwater level trend (m/yr)", 0.1),
           ("gw_mean_depth_m", "Mean groundwater depth (m)", 0.5),
           ("gw_latest_depth_m", "Latest groundwater depth (m)", 0.5),
           ("clay_pct", "Clay (%)", 1), ("slope_deg", "Slope (degrees)", 0.05),
           ("elevation_m", "Elevation (m)", 0.5)]


def slider_values(grid_id):
    r = BY_ID.loc[grid_id]
    out = []
    for col, _, _ in SLIDERS:
        v = r[col] if pd.notna(r[col]) else df[col].median()
        out.append(float(np.clip(v, df[col].min(), df[col].max())))
    return out


def record_view(grid_id, channel):
    """Update the Prometheus metrics for one successful cell lookup (no cell IDs or text in labels)."""
    r = BY_ID.loc[grid_id]
    metrics.REQUESTS.labels(channel).inc()
    kind = "cross_validated" if r.pred_source.startswith("cross-validated") else "fitted_on_all_training_cells"
    metrics.PREDICTIONS.labels(r.pred_class, kind).inc()
    metrics.CONFIDENCE.observe(r.confidence)
    metrics.LAST_CONFIDENCE.set(r.confidence)
    if not isinstance(r.susceptibility_class, str):
        metrics.UNLABELED.inc()
    elif r.insar_n_points < 3:
        metrics.LOW_SUPPORT.inc()


def show_cell(layer, grid_id, channel="id_search"):
    if grid_id not in BY_ID.index:
        metrics.INVALID.labels(channel).inc()
        msg = "Click a cell on the map, or type a cell ID such as TC-01234."
        return [draw_map(layer), grid_id or "", msg, None, None, "", None] + [gr.update()] * len(SLIDERS)
    record_view(grid_id, channel)
    md, fig, table = describe(grid_id)
    return [draw_map(layer, grid_id), grid_id, md, fig, table, "", None] + slider_values(grid_id)


def nearest_cell(x, y, max_dist=CELL * 0.71):
    d = np.hypot(XY[:, 0] - x, XY[:, 1] - y)
    i = int(d.argmin())
    return df.grid_id.iloc[i] if d[i] <= max_dist else None


@metrics.timed("map_click")
def on_click(layer, evt: gr.SelectData):
    px, py = evt.index
    lon = (XMIN + (px - LEFT) / AW * (XMAX - XMIN)) / KX
    lat = YMAX - (py - TOP) / AH * (YMAX - YMIN)
    x, y = to5070.transform(lon, lat)
    return show_cell(layer, nearest_cell(x, y), "map_click")


@metrics.timed("id_search")
def on_id_search(layer, grid_id):
    return show_cell(layer, (grid_id or "").strip().upper(), "id_search")


@metrics.timed("latlon_lookup")
def on_lookup(layer, lat, lon):
    if lat is None or lon is None:
        return show_cell(layer, None, "latlon_lookup")
    x, y = to5070.transform(lon, lat)
    gid = nearest_cell(x, y)
    if gid is None:
        out = show_cell(layer, None, "latlon_lookup")
        out[2] = "That location is outside the study grid (Tulare-Corcoran area, about 36.0-36.35 N, 119.8-119.2 W)."
        return out
    return show_cell(layer, gid, "latlon_lookup")


@metrics.timed("layer_switch")
def on_layer(layer, grid_id):
    metrics.LAYER.labels(layer).inc()
    return draw_map(layer, grid_id), legend_html(layer)


@metrics.timed("whatif")
def whatif(grid_id, *vals):
    if grid_id not in BY_ID.index:
        return "Select a cell first.", None
    row = BY_ID.loc[[grid_id], prepare.FEATURES].copy()
    base = MODEL.predict_proba(row)[0]
    for (col, _, _), v in zip(SLIDERS, vals):
        row[col] = v
    new = MODEL.predict_proba(row)[0]
    metrics.WHATIF.inc()
    if new.argmax() != base.argmax():
        metrics.WHATIF_CHANGED.inc()
    tab = "| Class | Cell as it is | With your changes |\n|---|---|---|\n" + "\n".join(
        f"| {LABEL[c]} | {a:.2f} | {b:.2f} |" for c, a, b in zip(CLASSES, base, new))
    msg = (f"Cell as it is: **{LABEL[CLASSES[base.argmax()]]}** ({base.max():.2f}). "
           f"With your changes: **{LABEL[CLASSES[new.argmax()]]}** ({new.max():.2f}).  \n"
           "This shows how the model reacts to its inputs. It does not say what would happen to the land.")
    return msg, tab


def refresh_feedback_gauges():
    by_status, by_rating, _ = feedback.counts()
    metrics.set_feedback_gauges(by_status, by_rating)


@metrics.timed("feedback")
def submit_feedback(grid_id, layer, rating, issue, comment):
    keep = gr.update()
    if grid_id not in BY_ID.index:
        return "Select a cell first (click the map or type its ID).", keep, keep, keep
    if rating not in feedback.RATINGS:
        return "Choose Good or Bad first.", keep, keep, keep
    r = BY_ID.loc[grid_id]
    observed = r.susceptibility_class if isinstance(r.susceptibility_class, str) else ""
    try:
        fid, is_new = feedback.add(grid_id, rating, issue, comment, map_layer=layer, predicted_class=r.pred_class,
                                   confidence=r.confidence, observed_class=observed, prediction_source=r.pred_source)
    except Exception as e:                       # database file not writable, disk full, ...
        metrics.ERRORS.labels("feedback").inc()
        print(f"Could not save feedback: {e!r}", flush=True)
        return "Sorry, your feedback could not be saved. Please try again later.", keep, keep, keep
    if is_new:
        metrics.FEEDBACK.labels(rating, r.pred_class).inc()
        refresh_feedback_gauges()
    return (f"Thank you. Your feedback was saved as **#{fid}**. The project team reads it in the Review tab and "
            "records what was done about it in the About tab."), None, feedback.ISSUES[0], ""


# ---- review tab (the project team): read feedback, decide, record what was done, export
def _review_view(status_filter):
    st = None if status_filter in (None, "All") else status_filter
    return feedback.summary_md(), feedback.table_html(feedback.rows(st))


def _code_problem(code):
    if not code:
        return "Enter the review code first."
    return None if feedback.review_code_ok(code) else "Wrong review code."


@metrics.timed("review")
def review_load(code, status_filter):
    bad = _code_problem(code)
    if bad:
        return bad, "", ""
    return ("",) + _review_view(status_filter)


@metrics.timed("review")
def review_save(code, fid, status, note, status_filter):
    bad = _code_problem(code)
    if bad:
        return bad, "", "", gr.update()
    if fid is None:
        return "Enter the feedback number (the # column).", gr.update(), gr.update(), gr.update()
    if not feedback.update(int(fid), status, note):
        return f"There is no feedback #{int(fid)}.", gr.update(), gr.update(), gr.update()
    refresh_feedback_gauges()
    return (f"Saved: feedback #{int(fid)} is now **{status}**.",) + _review_view(status_filter) + (feedback.outcomes_md(),)


@metrics.timed("review")
def review_export(code):
    bad = _code_problem(code)
    if bad:
        return None, bad
    path = os.path.join(tempfile.mkdtemp(), "subsidence_app_feedback.csv")
    n = feedback.export_csv(path)
    return path, f"Exported {n} entries."


# ---------------------------------------------------------------- page
BANNER = ("**Relative susceptibility ranking from 2020-2024 InSAR history, used to prioritize monitoring. "
          "Not a forecast, and not a guarantee that damage will occur at any location.**")

ABOUT = f"""
### What this tool is
A relative ranking of 500 m grid cells in the Tulare-Corcoran area (San Joaquin Valley, California) into five
susceptibility classes, for a groundwater manager deciding where to prioritize monitoring.

### Where the classes come from
- **Observed class:** the median 2020-01-01 to 2024-12-31 displacement of the InSAR points in the cell (DWR / TRE ALTAMIRA),
  cut into five classes at the 20th, 40th, 60th and 80th percentile of the training cells
  ({", ".join(f"{c:,.1f}" for c in META["cutpoints"]) if META.get("cutpoints") else "see data/label_build_log.json"}).
  {META["n_unlabeled"]:,} of {META["n_cells"]:,} cells have no usable InSAR point and no observed class.
- **Model prediction:** Random Forest (100 trees, minimum 5 samples per leaf) using groundwater, land use, soil, elevation and slope.
  Training cells: {META["n_train_cells"]:,} (whole 5 km blocks). Held-out test cells: {META["n_test_cells"]:,}.
- **How each prediction is made:** training cells show a cross-validated prediction (the cell's block was left out),
  every other cell shows the model fitted on all training cells. So no cell shows a prediction made by a model that saw its label.

### How good is it?
Cross-validated macro-F1 on the training blocks: **{META["oof_macro_f1"]:.3f}** (the project target was 0.50). Five classes means a random
guess scores about 0.20. Treat the prediction as a weak second opinion that is most useful next to the observed class, and read the
confidence. This app does not compute a score on the test cells.

### Limits to keep in mind
- **Units are inferred, not confirmed.** DWR does not state the unit of the point file. Values are treated as millimetres.
- **Thin labels:** about a quarter of labeled cells rest on fewer than 3 InSAR points (hatched on the map).
- **The label is a fitted trend**, not the observed change between two dates.
- **Cells with no label** are mostly in the south-west. Model estimates there are extrapolations from other places.
- Geologic layers are not included yet.

### Data and credit
California DWR (groundwater levels; TRE ALTAMIRA InSAR point data), USDA NASS Cropland Data Layer, USDA NRCS SSURGO,
USGS 3DEP. The InSAR data license is unresolved (see the repository README), so use this for the course project only.
"""

with gr.Blocks(title="Land Subsidence Susceptibility, Tulare-Corcoran") as demo:
    gr.Markdown("# Land Subsidence Susceptibility Map\nTulare-Corcoran area, San Joaquin Valley, California")
    gr.Markdown(BANNER)
    with gr.Tabs():
        with gr.Tab("Map"):
            with gr.Row():
                with gr.Column(scale=3):
                    layer = gr.Radio(LAYERS, value=LAYERS[0], label="Map layer")
                    map_img = gr.Image(value=draw_map(LAYERS[0]), type="pil", interactive=False, show_label=False,
                                       label="Susceptibility map")
                    gr.Markdown("Click a cell to see its details.")
                    legend = gr.HTML(legend_html(LAYERS[0]))
                with gr.Column(scale=2):
                    cell_id = gr.Textbox(label="Selected cell (click the map or type an ID like TC-01234)",
                                         placeholder="TC-01234")
                    details = gr.Markdown("Click a cell on the map.")
                    probs = gr.Plot(label="Model probabilities", show_label=False)
                    inputs_tbl = gr.Markdown()
                    with gr.Accordion("Find a location by latitude and longitude", open=False):
                        with gr.Row():
                            lat = gr.Number(label="Latitude", value=36.15, precision=5)
                            lon = gr.Number(label="Longitude", value=-119.5, precision=5)
                        go = gr.Button("Show this cell")
                    with gr.Accordion("Try the model: change some inputs for the selected cell", open=False):
                        sliders = [gr.Slider(float(df[c].min()), float(df[c].max()), step=s, label=lab,
                                             value=float(df[c].median())) for c, lab, s in SLIDERS]
                        wi_btn = gr.Button("Run the model with these inputs")
                        wi_msg = gr.Markdown()
                        wi_tbl = gr.Markdown()
                    with gr.Accordion("Send feedback about this cell", open=False):
                        fb_type = gr.Radio(feedback.RATINGS, label="Was the information for this cell useful?")
                        fb_issue = gr.Dropdown(feedback.ISSUES, value=feedback.ISSUES[0], label="What is it about?")
                        fb_text = gr.Textbox(label="Comment (optional)", lines=2, max_length=feedback.MAX_COMMENT,
                                             info="Up to 500 characters. Please do not include personal information.")
                        fb_btn = gr.Button("Send feedback")
                        fb_msg = gr.Markdown()
        with gr.Tab("Review feedback"):
            gr.Markdown("For the project team. Enter the review code to read the feedback, decide what to do about it, "
                        "and record what was done. Items marked **Action planned** or **Done** with a note are shown "
                        "to everyone in the About tab.")
            with gr.Row():
                rv_code = gr.Textbox(label="Review code", type="password", scale=2)
                rv_filter = gr.Dropdown(["All"] + feedback.STATUSES, value="All", label="Show", scale=1)
                rv_load = gr.Button("Load feedback", scale=1)
            rv_msg = gr.Markdown()
            rv_summary = gr.Markdown()
            rv_table = gr.HTML()
            with gr.Accordion("Record a decision or action", open=True):
                with gr.Row():
                    rv_id = gr.Number(label="Feedback # (from the table)", precision=0, scale=1)
                    rv_status = gr.Dropdown(feedback.STATUSES, value="Reviewed", label="New status", scale=2)
                rv_note = gr.Textbox(label="What was decided or done", lines=2, max_length=feedback.MAX_NOTE,
                                     info="Shown publicly (without the user's comment) when the status is "
                                          "Action planned or Done.")
                rv_save = gr.Button("Save")
            with gr.Row():
                rv_export = gr.Button("Export all feedback as CSV")
                rv_file = gr.File(label="CSV file", interactive=False)
        with gr.Tab("About the model and data"):
            gr.Markdown(ABOUT)
            outcomes_md = gr.Markdown(feedback.outcomes_md())

    outs = [map_img, cell_id, details, probs, inputs_tbl, wi_msg, wi_tbl] + sliders
    map_img.select(on_click, [layer], outs)
    cell_id.submit(on_id_search, [layer, cell_id], outs)
    go.click(on_lookup, [layer, lat, lon], outs)
    layer.change(on_layer, [layer, cell_id], [map_img, legend])
    wi_btn.click(whatif, [cell_id] + sliders, [wi_msg, wi_tbl])
    fb_btn.click(submit_feedback, [cell_id, layer, fb_type, fb_issue, fb_text], [fb_msg, fb_type, fb_issue, fb_text])
    rv_load.click(review_load, [rv_code, rv_filter], [rv_msg, rv_summary, rv_table])
    rv_filter.change(review_load, [rv_code, rv_filter], [rv_msg, rv_summary, rv_table])
    rv_save.click(review_save, [rv_code, rv_id, rv_status, rv_note, rv_filter], [rv_msg, rv_summary, rv_table, outcomes_md])
    rv_export.click(review_export, [rv_code], [rv_file, rv_msg])
    demo.load(feedback.outcomes_md, None, outcomes_md)

if __name__ == "__main__":
    metrics.init(CLASSES, LAYERS, META)
    refresh_feedback_gauges()
    if "REVIEW_CODE" not in os.environ:
        print("REVIEW_CODE is not set: the Review tab uses the default code 'review'. "
              "Set REVIEW_CODE before sharing the app with anyone.", flush=True)
    metrics.start()                                           # Prometheus metrics on port 8000 (METRICS_PORT)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
