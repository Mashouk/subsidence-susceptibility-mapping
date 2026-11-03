"""Insert Step 10 (label support, block edges, fold-wise cut points) into training_experiments.ipynb,
just before the '## Experiment results (Part A: Random Forest)' cell.
Usage (run from the folder that holds step10_cell.py):
    python add_step10_to_notebook.py ../training_experiments.ipynb
Then open the notebook, run the new cell (about 2 minutes), check that it prints 'macro-F1 0.3692', save with outputs, commit.
Standard library only. Safe to run twice."""
import json, sys, pathlib, uuid
nbpath = pathlib.Path(sys.argv[1]); nb = json.loads(nbpath.read_text(encoding="utf-8"))
code = pathlib.Path(__file__).with_name("step10_cell.py").read_text(encoding="utf-8")
md = """## Step 10 (Milestone 4): label support, block edges and cut points (checks prompted by the Milestone 3 feedback)

**A. Label support.** Each label is the median of the InSAR points inside a cell. How do the scores change when a label rests on one point, a few, or many? And does a rule that keeps only well-supported cells help when it is applied to the training side?

**B. Block edges.** The 5 km blocks are not separated by a buffer. How many validation cells (and how many of the real test cells) sit next to a cell on the other side of a block edge, and what happens to the score when a buffer is removed from the training side? Only the *positions* of the 967 test cells are used; no test label is read.

**C. Cut points.** The four class cut points were computed from all 2,572 development cells, so inside the folds the validation cells' labels helped set them. Here they are recomputed from the training part of each fold only.

Model: the selected forest (100 trees, `min_samples_leaf` 5), same block-grouped folds and preprocessing as Run 2. The cell runs on its own."""
reading = """**Reading of Step 10** (numbers from the cell above; check them against your own printed output).

- **A.** Errors pile up where labels are thin: macro-F1 is 0.254 to 0.259 for cells resting on 1 or 2 points (accuracy 0.28 to 0.30) and 0.429 for cells resting on 11 or more (accuracy 0.49). This is not a controlled comparison, because well-supported cells may differ in where they are and which classes they hold. Dropping thin cells from the *training* side does not help: macro-F1 falls from 0.3709 (all cells) to 0.3637 (at least 2 points), 0.3638 (at least 3) and 0.3580 (at least 5), with spreads of 0.002 to 0.005. So a support rule belongs on scoring and on the map layer, not on training.
- **B.** 22.0% of validation cells touch a training cell from another block, and 186 of the 967 test cells (19.2%) have a development cell next to them. Edge cells score higher than interior cells (0.419 against 0.355), the direction expected if neighbors help. Removing training cells within 0.75 km of the validation cells (6% of the training cells) lowers macro-F1 from 0.3709 to 0.3620 (sd 0.004). Wider buffers lower it further (0.3599 at 1.5 km, 0.3503 at 3 km) but also remove more training data, so that part of the drop is not leakage. I read the block-grouped scores as optimistic by about 0.01 at most.
- **C.** Recomputing the cut points inside each fold changes macro-F1 by -0.0004 (0.3704 against 0.3709), well inside the spread, so the shared cut points do not explain the scores."""
cells = nb["cells"]
def _txt(c): return "".join(c["source"])
already = [i for i, c in enumerate(cells) if (c["cell_type"] == "markdown" and (_txt(c).startswith("## Step 10 (Milestone 4)") or _txt(c).startswith("**Reading of Step 10**")))
           or (c["cell_type"] == "code" and _txt(c).startswith("# ---- Step 10 (Milestone 4)"))]
for i in reversed(already): del cells[i]
idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _txt(c).startswith("## Experiment results (Part A"))
def mk(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if nb.get("nbformat_minor", 0) >= 5: c["id"] = uuid.uuid4().hex[:8]
    if kind == "code": c.update(execution_count=None, outputs=[])
    return c
cells[idx:idx] = [mk("markdown", md), mk("code", code), mk("markdown", reading)]
nbpath.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("inserted 3 cells before cell", idx)
