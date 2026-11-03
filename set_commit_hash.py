"""Put your commit hash into the Milestone 4 docx (both yellow placeholders) and remove the yellow highlight.
Usage:  python set_commit_hash.py  <docx>  <commit-hash>
Example: python set_commit_hash.py Milestone4_AI_Component_and_Experimentation_Rahaman.docx 3f2a9c1
Needs: pip install python-docx
"""
import sys, docx
from docx.oxml.ns import qn

path, sha = sys.argv[1], sys.argv[2].strip()
d = docx.Document(path)
n = 0
for r in d.element.body.iter(qn('w:r')):
    t = r.find(qn('w:t'))
    if t is None or not t.text: continue
    if '[paste the commit hash' in t.text:
        t.text = sha
        rpr = r.find(qn('w:rPr'))
        h = rpr.find(qn('w:highlight')) if rpr is not None else None
        if h is not None: rpr.remove(h)
        n += 1
d.save(path)
print(f"replaced {n} placeholder(s) with {sha}")   # expect 2
