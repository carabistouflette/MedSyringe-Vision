"""Freeze a SAM box-refinement choice and check matched-instance non-regression."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_device_scene_audit import CACHE as DEVICE_PREDICTIONS
from scripts.detection_device_scene_data import MANIFEST as DEVICE
from scripts.detection_guided_geometry import GuidedMasks, OUTPUT as DEVELOPMENT, refine
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_precision_confirmation import CAPTURE, COHORTS, cohort
from scripts.expanded_proposals import image_for
from scripts.precision_study import source
from scripts.sam3_study import LOCK as WEIGHTS, configure_cpu

LOCK = ART / "guided_geometry_audit_freeze.json"
OUTPUT = ART / "guided_geometry_regression.json"
SCOPES = (*COHORTS, "device")


def destination(scope):
    return DATA / "proposals" / f"guided-geometry-regression-{scope}.json"


def forecasts(scope):
    if scope == "device":
        rows = read_json(DEVICE)["rows"]
        values = read_json(DEVICE_PREDICTIONS)["frames"]
        ranks = {k: v["actual"] for k, v in values.items()}
        proof = DEVICE_PREDICTIONS
    else:
        rows, baseline = cohort(scope)
        proof = DATA / "proposals" / f"fixed-precision-{scope}.json"
        all_boxes = read_json(proof)["frames"]
        ranks = {k: [p for p in v["ranked"] if p["score"] >= baseline[k]["floor"]]
                 for k, v in all_boxes.items()}
    if {r["id"] for r in rows} != set(ranks):
        raise ValueError("Missing frozen accepted boxes.")
    return rows, ranks, proof


def seal():
    if LOCK.exists() or OUTPUT.exists() or any(destination(scope).exists() for scope in SCOPES):
        raise ValueError("Guided geometry qualification has already begun.")
    selected = read_json(DEVELOPMENT)
    best = selected["best"][0]
    if any(v["map"] <= selected["baseline"][k]["map"] for k, v in best["scopes"].items()):
        raise ValueError("Development did not show a useful improvement across both domains.")
    assets = [Path(__file__), Path("scripts/detection_guided_geometry.py"), DEVELOPMENT, WEIGHTS, DEVICE]
    catalog = {scope: {r["id"]: r.get("derived_sha256", r["sha256"]) for r in forecasts(scope)[0]}
               for scope in SCOPES}
    assets.extend(forecasts(scope)[2] for scope in SCOPES)
    write_json(LOCK, dict(decision=best["choice"], catalog=catalog, paths={str(p): digest(p) for p in assets},
                          protocol="Preserve every original recall match at IoU 0.50 and every automatic count. "
                                   "No false positive may be added in any historical cohort. Estimate the paired "
                                   "AP@50–95 difference among the accepted boxes. All cohorts are regression "
                                   "evidence. A separate new natural-image confirmation is needed for activation."))


def verify(scope):
    lock = read_json(LOCK)
    rows, ranks, proof = forecasts(scope)
    paths = (Path(__file__), Path("scripts/detection_guided_geometry.py"), DEVELOPMENT, WEIGHTS, proof)
    hashes = {r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows}
    if any(digest(p) != lock["paths"][str(p)] for p in paths) or hashes != lock["catalog"][scope]:
        raise ValueError("A sealed box-refinement input changed.")
    return rows, ranks, lock["decision"]


def collect(scope, capture):
    refs, ranks, choice = verify(scope)
    configure_cpu()
    path = destination(scope)
    sealed = digest(LOCK)
    saved = read_json(path) if path.exists() else dict(freeze_sha256=sealed, frames={})
    if saved["freeze_sha256"] != sealed:
        raise ValueError("This cached geometry was computed under another qualification.")
    model = None
    for row in refs:
        identifier = row["id"]
        if identifier in saved["frames"]:
            continue
        accepted = ranks[identifier]
        start = perf_counter()
        if accepted:
            if model is None:
                model = GuidedMasks()
            picture = image_for(row, capture) if scope in CAPTURE else source(row)
            masks = model.predict(picture, [p["bbox"] for p in accepted])
            candidate = refine(accepted, masks, choice)
        else:
            candidate = []
        saved["frames"][identifier] = dict(
            ranked=[dict(bbox=p["bbox"], score=p["score"]) for p in candidate],
            ms=1000 * (perf_counter() - start))
        count = len(saved["frames"])
        if count % 4 == 0 or count == len(refs):
            write_json(path, saved)
            print("Guided box regression", scope, count, "/", len(refs), flush=True)


def measure():
    if OUTPUT.exists():
        raise ValueError("The regression is already measured.")
    scopes = {}
    fields = ("images", "objects", "tp", "fp", "fn", "precision", "recall", "f1", "ap50", "ap50_95", "exact_count")
    for scope in SCOPES:
        rows, before, _ = verify(scope)
        loaded = read_json(destination(scope))
        frames = loaded["frames"]
        if loaded["freeze_sha256"] != digest(LOCK) or set(frames) != set(before):
            raise ValueError(f"Incomplete output for {scope}.")
        after = {k: v["ranked"] for k, v in frames.items()}
        same_counts = all(len(before[k]) == len(after[k]) for k in frames)
        original, candidate = [evaluate(rows, ranked, 0.) for ranked in (before, after)]
        check = compare(original, candidate)
        scopes[scope] = dict(passed=same_counts and check["passed"] and candidate["fp"] <= original["fp"],
                              counts_identical=same_counts, guard=check,
                              previous={k: original[k] for k in fields},
                              proposed={k: candidate[k] for k in fields},
                              geometry_median_ms=median(v["ms"] for v in list(frames.values())[1:]
                                                        if v["ranked"]) if any(
                                  v["ranked"] for v in list(frames.values())[1:]) else None)
    write_json(OUTPUT, dict(passed=all(v["passed"] for v in scopes.values()),
                            decision_sha256=digest(LOCK), cohorts=scopes))
    print("Guided box non-regression", {k: v["passed"] for k, v in scopes.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "collect", "measure"))
    parser.add_argument("--scope", choices=SCOPES)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    if args.action == "seal":
        seal()
    elif args.action == "measure":
        measure()
    elif args.scope and (args.scope not in CAPTURE or args.capture):
        collect(args.scope, args.capture)
    else:
        parser.error("Select the cohort and provide its original capture path if needed.")
