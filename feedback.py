"""feedback.py - saves user feedback so it can be reviewed and acted on (SQLite, Python standard library only).

Flow:  user submits feedback (Map tab)  ->  row saved with status "New"  ->  the project team opens the Review tab,
reads it, sets a status and writes what was decided or done  ->  items marked "Action planned" or "Done" are shown to
everyone in the About tab, so the people who gave feedback can see what happened to it.

The database is one file (default feedback/feedback.db, set FEEDBACK_DB to change). It is not wiped when the app restarts.
No names, e-mail addresses or IP addresses are stored.
"""
import csv
import hmac
import html
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("FEEDBACK_DB", os.path.join("feedback", "feedback.db"))
STATUSES = ["New", "Reviewed", "Action planned", "Done", "No action"]
RATINGS = ["Good", "Bad"]
ISSUES = ["Just a rating", "Observed class looks wrong", "Model prediction looks wrong",
          "Confidence seems wrong", "Location or map problem", "Something is unclear", "Other"]
MAX_COMMENT = 500
MAX_NOTE = 500
_lock = threading.Lock()
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

COLUMNS = ["id", "created_utc", "grid_id", "map_layer", "rating", "issue", "comment", "predicted_class", "confidence",
           "observed_class", "prediction_source", "status", "action_note", "reviewed_utc"]


def _connect():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init():
    with _lock, _connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL, grid_id TEXT NOT NULL, map_layer TEXT,
            rating TEXT NOT NULL, issue TEXT, comment TEXT, predicted_class TEXT, confidence REAL, observed_class TEXT,
            prediction_source TEXT, status TEXT NOT NULL DEFAULT 'New', action_note TEXT, reviewed_utc TEXT)""")


def clean(text, limit):
    """Plain text only: control characters removed, trimmed, cut to `limit` characters."""
    return _CONTROL.sub("", (text or "")).strip()[:limit]


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def add(grid_id, rating, issue, comment, map_layer="", predicted_class="", confidence=None, observed_class="",
        prediction_source=""):
    """Save one feedback entry and return (id, is_new). An identical entry from the last 30 s is treated as a double click."""
    if rating not in RATINGS:
        raise ValueError("rating must be Good or Bad")
    issue = issue if issue in ISSUES else "Other"
    comment = clean(comment, MAX_COMMENT)
    with _lock, _connect() as con:
        since = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        dup = con.execute("SELECT id FROM feedback WHERE grid_id=? AND rating=? AND issue=? AND comment=? "
                          "AND created_utc>=? ORDER BY id DESC LIMIT 1", (grid_id, rating, issue, comment, since)).fetchone()
        if dup:
            return dup["id"], False
        cur = con.execute(
            "INSERT INTO feedback (created_utc, grid_id, map_layer, rating, issue, comment, predicted_class, confidence, "
            "observed_class, prediction_source) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_now(), grid_id, map_layer, rating, issue, comment, predicted_class,
             None if confidence is None else round(float(confidence), 4), observed_class, prediction_source))
        return cur.lastrowid, True


def rows(status=None, limit=500):
    q, args = "SELECT * FROM feedback", []
    if status in STATUSES:
        q += " WHERE status=?"
        args.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _lock, _connect() as con:
        return [dict(r) for r in con.execute(q, args).fetchall()]


def update(fid, status, note):
    """Set status and action note of one entry. Returns False if the id does not exist."""
    if status not in STATUSES:
        raise ValueError("unknown status")
    with _lock, _connect() as con:
        cur = con.execute("UPDATE feedback SET status=?, action_note=?, reviewed_utc=? WHERE id=?",
                          (status, clean(note, MAX_NOTE), _now(), int(fid)))
        return cur.rowcount == 1


def counts():
    """-> ({status: n}, {rating: n}, total); every status and rating is present, also when 0."""
    by_status, by_rating = {s: 0 for s in STATUSES}, {r: 0 for r in RATINGS}
    with _lock, _connect() as con:
        for r in con.execute("SELECT status, COUNT(*) n FROM feedback GROUP BY status"):
            by_status[r["status"]] = r["n"]
        for r in con.execute("SELECT rating, COUNT(*) n FROM feedback GROUP BY rating"):
            by_rating[r["rating"]] = r["n"]
    return by_status, by_rating, sum(by_status.values())


def outcomes(limit=10):
    """Entries the team acted on (Action planned / Done) that have a note. Only the issue type and the note are public,
    never the user's comment or the cell."""
    with _lock, _connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT issue, status, action_note, reviewed_utc FROM feedback WHERE status IN ('Action planned','Done') "
            "AND action_note<>'' ORDER BY reviewed_utc DESC, id DESC LIMIT ?", (limit,))]


def _spreadsheet_safe(v):
    """Stop Excel or Sheets from running a comment such as =HYPERLINK(...) as a formula."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def export_csv(path):
    data = rows(limit=1_000_000)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:          # utf-8-sig so Excel reads it correctly
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in reversed(data):
            w.writerow([_spreadsheet_safe(r[c]) for c in COLUMNS])
    return len(data)


def review_code_ok(code):
    """Simple gate for the Review tab: the code in the REVIEW_CODE environment variable (default 'review')."""
    want = os.environ.get("REVIEW_CODE", "review")
    ok = hmac.compare_digest((code or "").encode(), want.encode())
    if not ok:
        time.sleep(1)                         # makes guessing the code slow
    return ok


# ---- display helpers (everything user-written is HTML-escaped) ----
def _e(v):
    return html.escape("" if v is None else str(v)).replace("\n", "<br>")


def table_html(data):
    if not data:
        return '<div style="padding:12px">No feedback to show.</div>'
    head = ["#", "Time (UTC)", "Cell", "Rating", "Issue", "Comment", "Model said", "InSAR said", "Status", "Action taken"]
    th = "".join(f'<th style="text-align:left;padding:6px 8px;border-bottom:2px solid var(--border-color-primary)">{h}</th>'
                 for h in head)
    td = 'style="padding:6px 8px;border-bottom:1px solid var(--border-color-primary);vertical-align:top"'
    body = []
    for r in data:
        conf = "" if r["confidence"] is None else f' ({r["confidence"]:.2f})'
        color = "#d1453b" if r["rating"] == "Bad" else "#2e9e5b"
        cells = [_e(r["id"]), _e(r["created_utc"]), _e(r["grid_id"]),
                 f'<b style="color:{color}">{_e(r["rating"])}</b>', _e(r["issue"]), _e(r["comment"]),
                 _e(r["predicted_class"]) + _e(conf), _e(r["observed_class"] or "none"), f'<b>{_e(r["status"])}</b>',
                 _e(r["action_note"])]
        body.append("<tr>" + "".join(f"<td {td}>{c}</td>" for c in cells) + "</tr>")
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-size:13px">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def summary_md():
    by_status, by_rating, total = counts()
    if total == 0:
        return "No feedback yet."
    bad = by_rating["Bad"]
    return (f"**{total}** entries: {by_rating['Good']} Good, {bad} Bad ({bad / total:.0%} Bad).  \n"
            + " | ".join(f"{s}: **{by_status[s]}**" for s in STATUSES))


def outcomes_md():
    by_status, _, total = counts()
    done = by_status["Done"] + by_status["Action planned"]
    lines = ["### Feedback and what we did about it",
             f"People have sent **{total}** feedback entr{'y' if total == 1 else 'ies'} so far. "
             f"The team has reviewed {total - by_status['New']} of them and acted on, or planned action for, {done}."]
    items = outcomes()
    if items:
        lines.append("")
        for r in items:
            lines.append(f"- **{html.escape(r['status'])}** ({html.escape(r['issue'])}): {html.escape(r['action_note'])}")
    else:
        lines.append("\nNothing has been changed because of feedback yet.")
    return "\n".join(lines)
