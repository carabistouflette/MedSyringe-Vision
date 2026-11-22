"""Rescore the fixed native crowd proposal against each independent visual reading."""
from itertools import combinations
from pathlib import Path
from statistics import median

import numpy as np
from scipy.optimize import linear_sum_assignment

from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.crowded_sam_hybrid import CACHE, POLICIES, RESULTS, baseline, reconcile
from scripts.detection_hillclimb_experiment import ART
from scripts.expanded_detection_study import short

READS = Path("data/public-syringe-dataset")
OUTPUT = ART / "crowded_sam_reference_sensitivity.json"


def agreement(first, second):
    paired = np.array([[box_iou(a, b) for b in second] for a in first])
    if not len(first) or not len(second):
        return []
    left, right = linear_sum_assignment(paired, maximize=True)
    return [float(paired[i, j]) for i, j in zip(left, right, strict=True)]


def read_reviews(paths, expected):
    reviewed = {}
    for path in paths:
        source = read_json(path)["rows"]
        if set(source) != expected:
            raise ValueError("Every reader must inspect the same unmarked photos.")
        for item in source.values():
            if not all(valid_box(box) for box in item["boxes"]):
                raise ValueError("The reader returned invalid coordinates.")
        reviewed[path.stem] = source
    return reviewed


def run():
    if OUTPUT.exists():
        raise FileExistsError("The reference sensitivity analysis is already written.")
    fixed = read_json(RESULTS)
    rows, known = baseline()["native regression"]
    native = read_json(CACHE / "native-fixed.json")["frames"]
    cases = {r["id"]: r for r in rows if len(r["objects"]) >= 2}
    paths = [READS / f"native-blind-boxes-{name}.json" for name in ("a", "b")]
    independent = read_reviews(paths, set(cases))
    comparators = {"frozen": {key: {"boxes": [o["bbox"] for o in item["objects"]]}
                              for key, item in cases.items()}, **independent}
    proposals = {"released": known, **{policy: {
        key: reconcile(result, native.get(key, {}).get("accepted", []), policy)
        for key, result in known.items()} for policy in POLICIES}}
    rows_per_version = {}
    cross_reader = {}
    for (left_name, left), (right_name, right) in combinations(comparators.items(), 2):
        key = left_name + " versus " + right_name
        cross_reader[key] = {}
        for name in cases:
            alignment = agreement(left[name]["boxes"], right[name]["boxes"])
            cross_reader[key][name] = dict(counts=[len(left[name]["boxes"]), len(right[name]["boxes"])],
                                           paired_iou=alignment)
    for label, group in comparators.items():
        local = [{**r, "objects": [dict(bbox=b) for b in group[r["id"]]["boxes"]]} if r["id"] in cases else r
                 for r in rows]
        scored = {arm: evaluate(local, guesses, 0.) for arm, guesses in proposals.items()}
        original = scored["released"]
        rows_per_version[label] = {
            arm: dict(metrics=short(result), retention=compare(original, result),
                      errors=dict(false_positive=result["false_positives"],
                                  false_negative=result["false_negatives"]))
            for arm, result in scored.items()}
    counterpart = tuple(independent)
    measurements = cross_reader[" versus ".join(counterpart)]
    corroboration = [iou for frame in measurements.values() for iou in frame["paired_iou"]]
    write_json(OUTPUT, dict(official=rows_per_version["frozen"], sensitivity=rows_per_version,
                            independent_labels=independent, geometry_agreement=cross_reader,
                            independent_iou_median=median(corroboration) if corroboration else None,
                            lock_sha256=fixed["freeze_sha256"],
                            readers={str(path): digest(path) for path in paths},
                            scope="Sensitivity analysis on new blind readings of the same five crowded rasters. "
                                  "The official release findings stay on the originally frozen labels."))
    print({label: {arm: (record["metrics"]["tp"], record["metrics"]["fp"])
                   for arm, record in versions.items()} for label, versions in rows_per_version.items()}, flush=True)


if __name__ == "__main__":
    run()
