"""Check proposal-guided high-resolution crops without changing base boxes."""
import argparse
from itertools import product
from math import ceil

from PIL import ImageOps
from pathlib import Path
from time import perf_counter

import torch

from medsyringe.detection.additive import additions, duplicates, project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import Grounder
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.baselines.additive_study import FRESH
from scripts.experiments.detector_training.optimize_detector import FINAL, counters
from scripts.experiments.detector_audits.precision_study import baseline, source
from scripts.experiments.detector_training.refine_detector import ART, OLD
from scripts.experiments.transfer.supervised_study import sets


def regions(ranked, size, cutoff):
    w, h = size
    kept = [p for p in ranked if p["score"] >= cutoff]
    # Evaluate the most promising unexplained proposals. Adjacent proposals
    # covered by a previously scheduled crop reuse that pass.
    result = []
    for p in ranked:
        l, t, r, b = p["bbox"]
        if not .30 <= p["score"] < cutoff or any(duplicates(p["bbox"], q["bbox"]) for q in kept):
            continue
        cx, cy = (l+r)/2, (t+b)/2
        if any(x <= cx*w <= X and y <= cy*h <= Y for x, y, X, Y in result):
            continue
        width, height = min(1, max(.50, 1.55*(r-l))), min(1, max(.50, 1.55*(b-t)))
        x, y = min(1-width, max(0, cx-width/2)), min(1-height, max(0, cy-height/2))
        result.append((round(x*w), round(y*h), round((x+width)*w), round((y+height)*h)))
        if len(result) == 2:
            break
    return result


def zoom(detector, image, window):
    crop = image.crop(window)
    w, h = crop.size
    # Keep enough feature-map positions for Grounding DINO's fixed query count.
    extra_x, extra_y = max(0, ceil(h/2)-w), max(0, ceil(w/2)-h)
    x, y = extra_x//2, extra_y//2
    canvas = ImageOps.expand(crop, (x, y, extra_x-x, extra_y-y), fill=(127,127,127))
    W, H = canvas.size
    for prediction in detector.propose(canvas):
        l,t,r,b = prediction["bbox"]
        box = [(l*W-x)/w, (t*H-y)/h, (r*W-x)/w, (b*H-y)/h]
        if box[0] >= 1 or box[1] >= 1 or box[2] <= 0 or box[3] <= 0:
            continue
        box = [max(0,min(1,v)) for v in box]
        restored = project(box, window, image.size)
        if restored:
            yield dict(bbox=restored, score=prediction["score"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    frozen, stage = read_json(OLD), read_json(FINAL)
    _, val, historical = sets()
    if args.audit:
        calibration_path = ART / "zoom_study.json"
        calibration = read_json(calibration_path)
        if calibration["base_sha256"] != digest(FINAL):
            raise ValueError("Frozen model changed.")
        winning = max((t for t in calibration["trials"] if t["guard"]["passed"]),
                      key=lambda t: (t["f1"], t["precision"], t["iou"], t["support"]))
        locked = {k: winning[k] for k in ("aux_cutoff", "support", "iou")}
        write_json(ART / "zoom_external_protocol.json", dict(setting=locked,
                   manifest_sha256=digest(FRESH), calibration_sha256=digest(calibration_path)))
        rows = read_json(FRESH)["rows"]
        base = baseline(frozen, stage, rows, "additive-audit")
    else:
        rows = [*val, *historical]
        base = {**baseline(frozen, stage, val, "precision-fresh"),
                **baseline(frozen, stage, historical, "test")}
    cutoff = stage["cutoff"]
    detector = Grounder(Path("."), frozen["backbone"])
    recovered = {}
    started = perf_counter()
    passes = 0
    for i, row in enumerate(rows, 1):
        image = source(row)
        windows = regions(base[row["id"]], image.size, cutoff)
        boxes = []
        for window in windows:
            passes += 1
            boxes.extend(zoom(detector, image, window))
        recovered[row["id"]] = boxes
        if i % 5 == 0:
            print("ROI study", i, "/", len(rows), flush=True)
    old = evaluate(rows, base, cutoff)
    if args.audit:
        result = {r["id"]: additions(base[r["id"]], recovered[r["id"]], cutoff, **locked)
                  for r in rows}
        stats = evaluate(rows, result, cutoff)
        write_json(ART / "zoom_external_audit.json", dict(baseline=old, experimental=stats,
                   guard=compare(old, stats), extra_passes=passes,
                   elapsed_seconds=perf_counter()-started,
                   scope="Previously consulted public photos, exploratory check only"))
        print("Zoom audit", counters(old), counters(stats), flush=True)
        return
    trials = []
    for threshold, support, overlap in product((.35, .45, .55, .65), (.25, .40, .55), (.30, .45)):
        setting = dict(aux_cutoff=threshold, support=support, iou=overlap)
        result = {r["id"]: additions(base[r["id"]], recovered[r["id"]], cutoff, **setting)
                  for r in rows}
        stats = evaluate(rows, result, cutoff)
        trials.append(dict(**setting, **counters(stats), guard=compare(old, stats)))
    eligible = [r for r in trials if r["guard"]["passed"] and r["tp"] > old["tp"]
                and r["precision"] >= old["precision"]]
    choice = max(eligible, key=lambda r: (r["f1"], r["precision"])) if eligible else None
    write_json(ART / "zoom_study.json", dict(baseline=counters(old), selected=choice, trials=trials,
                              extra_passes=passes, elapsed_seconds=perf_counter()-started,
                              scope="Calibration only, combined previously inspected sets",
                              base_sha256=digest(FINAL)))
    print("Baseline", counters(old), "selected", choice, flush=True)


if __name__ == "__main__":
    main()
