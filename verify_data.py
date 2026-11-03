"""Check the data files against the recorded SHA-256 checksums.

Run from the repository root:   python verify_data.py

Reads data/checksums.json (the checksums) and data/label_build_log.json (the checksums the label build wrote down),
hashes each file that is present, and says OK, MISSING, SKIPPED (a large file that is not in the repository) or MISMATCH.
A text file whose content is right but whose line endings were changed by git is reported separately, because that is
the usual cause of a mismatch on a fresh clone (see .gitattributes and README.md). Standard library only.
"""
import hashlib, json, os, sys

def sha(b): return hashlib.sha256(b).hexdigest()
def file_sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

spec = json.load(open("data/checksums.json", encoding="utf-8"))
bad = 0
print(f"{'file':44s} status")
for e in spec["files"]:
    p, want = e["path"], e["sha256"]
    if not os.path.exists(p):
        if e.get("committed_in_repo", True):
            print(f"{p:44s} MISSING (should be in the repository)"); bad += 1
        else:
            print(f"{p:44s} SKIPPED (large file, not in the repository; checksum on record {want[:16]}...)")
        continue
    got = file_sha(p)
    if got == want:
        print(f"{p:44s} OK"); continue
    if p.endswith((".csv", ".json", ".txt")):
        raw = open(p, "rb").read()
        lf = raw.replace(b"\r\n", b"\n"); crlf = lf.replace(b"\n", b"\r\n")
        if want in (sha(lf), sha(crlf)):
            print(f"{p:44s} LINE ENDINGS CHANGED (content is right; git rewrote the line endings. Fix: git add --renormalize . and commit)")
            bad += 1; continue
    print(f"{p:44s} MISMATCH (got {got[:16]}..., expected {want[:16]}...)"); bad += 1

# cross-check: the label build wrote its own checksums; they must agree with the table above
log = json.load(open("data/label_build_log.json", encoding="utf-8"))
byp = {e["path"]: e["sha256"] for e in spec["files"]}
pairs = [("data/subsidence_dataset.csv", log["input_csv"]["sha256"]),
         ("data/subsidence_dataset_processed.csv", log["output_csv"]["sha256"]),
         ("data/insar_study_area.csv", log["study_area_csv"]["sha256"]),
         ("data/insar_points.zip", log["extract_log"]["src_sha256"])]
for p, v in pairs:
    ok = byp.get(p) == v
    print(f"label_build_log.json agrees for {p:40s} {'yes' if ok else 'NO'}"); bad += (not ok)
print("\nAll recorded checksums match." if not bad else f"\n{bad} problem(s) found.")
sys.exit(1 if bad else 0)
