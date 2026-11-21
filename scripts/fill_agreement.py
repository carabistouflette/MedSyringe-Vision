"""Measure agreement between independently localized piston edges and audit a lumen veto."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from medsyringe.data import digest, read_json, write_json
from medsyringe.fill import SCALES, estimate
from medsyringe.graduation_ocr import supplement_graduations
from medsyringe.liquid_appearance import appearance
from medsyringe.liquid_region import HEAD
from medsyringe.piston_landmarks import LANDMARKS
from scripts.expanded_liquid import ART as EXPANDED, records
from scripts.fill_study import ART, LEGACY_ROWS, prepared, references
from scripts.graduation_ocr_study import CACHE as OCR_CACHE, LOCK as OCR_LOCK
from scripts.piston_study import LOCK as PISTON_LOCK, estimate_landmarks, frame_for, frozen, project_fill

LOCK = ART / "piston_ocr_agreement_selection.json"
AUDIT = ART / "piston_ocr_agreement_regression.json"
VETO = EXPANDED / "short_lumen_veto_regression.json"


def measurements(source, split):
    _, model, strategy = frozen()
    ocr_policy = read_json(OCR_LOCK)["shadow"]["policy"]
    patches = read_json(OCR_CACHE / f"{split}.json")["frames"]
    settings = read_json(ART / "release.json")["settings"]
    rows, _ = references()
    original = {r["id"]: r["bbox"] for r in read_json(LEGACY_ROWS)["files"]}
    output = {}
    for name, image, texts, appearance_result, bounds in prepared(source, split):
        base = estimate(image, texts, appearance_result, settings)
        if "ml" in base:
            continue
        supplemented = supplement_graduations(texts, patches[name], ocr_policy)
        reading = estimate(image, supplemented, appearance_result, settings)
        if "ml" not in reading:
            continue
        row = rows[name]
        frame = frame_for(row)
        result = estimate_landmarks(frame, model, strategy)
        physical = None
        if result:
            h,w = frame["mask"].shape
            coords = np.array([result[k] for k in LANDMARKS])/[w,h]
            if split == "holdout-60" or row["group"] == "amber-20-low-fill-dark-wall":
                x,y,r,b = row["objects"][0]["bbox"]
                ox,oy,ore,ob = original[name]
                coords = (coords*[r-x,b-y] + [x-ox,y-oy])/[ore-ox,ob-oy]
            physical = project_fill(image, supplemented, appearance_result, coords, settings)
        output[name] = dict(optical=reading["ml"], physical=physical,
                            minor=SCALES[reading["capacity_ml"]][1], reference=bounds)
    return output


def score(readings, tolerance):
    accepted = {k: r for k,r in readings.items() if r["physical"] is not None and
                abs(r["optical"]-r["physical"]) <= r["minor"]*tolerance}
    correct = sum(bool(v["reference"] and v["reference"][0] <= v["optical"] <= v["reference"][1])
                  for v in accepted.values())
    return dict(eligible=len(readings), answered=len(accepted), within_bracket=correct,
                accepted=sorted(accepted))


def select(source):
    if LOCK.exists():
        raise ValueError("The agreement thresholds are already frozen.")
    observed = measurements(source, "development")
    trials = [dict(tolerance=tolerance, outcome=score(observed, tolerance)) for tolerance in (0,1)]
    safe = [r for r in trials if r["outcome"]["answered"] == r["outcome"]["within_bracket"] > 0]
    key = lambda r: (r["outcome"]["within_bracket"], -r["tolerance"])
    selected = max(safe, key=key, default=None)
    write_json(LOCK, dict(selected=selected, shadow=selected or max(trials,key=key), trials=trials,
                          source_freezes={str(p): digest(p) for p in (PISTON_LOCK, OCR_LOCK)},
                          estimates=observed))
    print("Agreement development", trials, flush=True)


def audit(source):
    freeze = read_json(LOCK)
    if any(digest(p) != h for p,h in freeze["source_freezes"].items()):
        raise ValueError("Piston or OCR selection changed.")
    results = {}
    for split in ("holdout","holdout-60","clear"):
        found = measurements(source, split)
        results[split] = dict(observed=found, outcome=score(found, freeze["shadow"]["tolerance"]))
    outcomes = [r["outcome"] for r in results.values()]
    passed = bool(freeze["selected"] and sum(r["answered"] for r in outcomes) > 0 and
                  all(r["answered"] == r["within_bracket"] for r in outcomes))
    write_json(AUDIT, dict(passed=passed, selection_sha256=digest(LOCK), groups=results))
    print("Agreement regression", {k: v["outcome"] for k,v in results.items()}, flush=True)


def veto():
    _, model, policy = frozen()
    legacy = read_json(HEAD)
    controls = {}
    for split in ("calibration","test"):
        cases = []
        for row, frame in records(split):
            logits = cv2.resize(frame["grid"]@legacy["weights"]+legacy["bias"], frame["mask"].shape[::-1])
            before = appearance(frame["raw"], frame["mask"], logits >= legacy["cutoff"])
            if not before["lightness"]:
                continue
            local = estimate_landmarks(frame, model, policy)
            fraction = local.get("fraction") if local else None
            # Reuse the pre-existing minimum lumen size from lumen_support.
            if fraction is not None and fraction < .08:
                cases.append(dict(id=row["id"], reference_colored=frame["distance"] is not None,
                                  apparent_column_fraction=fraction, legacy=before))
        controls[split] = cases
    safe = all(not r["reference_colored"] for c in controls.values() for r in c)
    useful = bool(controls["test"])
    write_json(VETO, dict(passed=safe and useful, notes="Conservative veto only for a verified short visible segment.",
                          selection_sha256=digest(PISTON_LOCK), vetoes=controls))
    print("Short lumen veto", safe and useful, {k: [r["id"] for r in v] for k,v in controls.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select","audit","veto"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.stage == "veto":
        veto()
    elif args.source:
        (select if args.stage == "select" else audit)(args.source)
    else:
        parser.error("The fill agreement check requires --source.")
