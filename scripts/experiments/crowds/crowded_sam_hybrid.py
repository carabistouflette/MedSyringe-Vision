"""Qualify fixed crowd-only SAM policies against the current hybrid."""
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.crowds.crowded_sam_resolution import CACHE, FOCAL, OUTPUT as DEVELOPMENT
from scripts.experiments.crowds.crowded_sam_resolution import CrowdedMasks, configure_cpu, masks, rows, select
from scripts.experiments.transfer.detection_device_scene_audit import CACHE as NATIVE_PREDICTIONS, MANIFEST as NATIVE
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART
from scripts.experiments.detector_audits.detection_transparent_followup import CACHE as PUBLIC_PREDICTIONS
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.measurement.field_abstention_audit import CACHE as LIVE_CACHE

RESULTS = ART / "crowded_sam_hybrid_qualification.json"
PLAN = ART / "crowded_sam_hybrid_freeze.json"
OVERLAP_CASE = ART / "public_overlap_count_case.json"
POLICIES = ("add unmatched", "replace aligned")
MINIMUM_IOU = .40


def reconcile(known, auxiliary, mode):
    if len(known) < 2 or len(auxiliary) <= len(known):
        return known
    overlap = np.asarray([[box_iou(before["bbox"], after["bbox"]) for after in auxiliary] for before in known])
    # Maximize the number of admissible matches before preferring one by overlap.
    assignments, peers = linear_sum_assignment(overlap + (overlap >= MINIMUM_IOU) * (len(known) + 1), maximize=True)
    if any(overlap[i, j] < MINIMUM_IOU for i, j in zip(assignments, peers, strict=True)):
        return known
    if mode == "replace aligned":
        return auxiliary
    selected = set(peers.tolist())
    return [*known, *({**item, "score": 0.} for i, item in enumerate(auxiliary) if i not in selected)]


def baseline():
    public = read_json(PUBLIC_PREDICTIONS)["frames"]["crowded-commons-confirmation"]
    native = read_json(NATIVE_PREDICTIONS)["frames"]
    field = read_json(LIVE_CACHE)["frames"]
    current = read_json(OVERLAP_CASE)["results"]["public"]
    focal = [dict(bbox=box, score=.68) for box in current["boxes"]]
    return {
        "public development": (rows("development"), {
            row["id"]: focal if row["id"] == read_json(FOCAL)["id"] else field[row["id"]]["objects"]
            for row in rows("development")}),
        "public transfer": (rows("transfer"), {row["id"]: public[row["id"]]["previous"] for row in rows("transfer")}),
        "native regression": (read_json(NATIVE)["rows"], {k: row["actual"] for k, row in native.items()}),
    }


def lock(groups, configuration):
    return dict(code=digest(__file__), development=digest(DEVELOPMENT), configuration=configuration,
                plans={name: {row["id"]: [row["sha256"], len(known[row["id"]])] for row in cohort}
                       for name, (cohort, known) in groups.items()},
                acceptance="The same boxes must remain matched at IoU .50. The count must remain correct on "
                           "every already correct scene, and no cohort may gain extra false positives. "
                           "Only trigger where the shipped engine accepts at least two objects.")


def run():
    if RESULTS.exists():
        raise FileExistsError("The fixed hybrid comparison is already complete.")
    configuration = read_json(DEVELOPMENT)["selected"]["configuration"]
    edge, prompt, geometry, identity, floor = configuration
    groups = baseline()
    expected = lock(groups, configuration)
    if PLAN.exists() and read_json(PLAN) != expected:
        raise ValueError("The audit plan changed after freezing.")
    write_json(PLAN, expected)
    cache = CACHE / "native-fixed.json"
    native = read_json(cache) if cache.exists() else dict(plan=digest(PLAN), frames={})
    if native["plan"] != digest(PLAN):
        raise ValueError("Cached native proposals belong to a different plan.")
    cohort, known = groups["native regression"]
    todo = [row for row in cohort if len(known[row["id"]]) >= 2 and row["id"] not in native["frames"]]
    if todo:
        configure_cpu()
        model = CrowdedMasks(edge, "xpu")
        for row in todo:
            path = Path("data") / row["image"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"The native source {row['id']} changed.")
            picture = read_image(path)
            started = perf_counter()
            views = dict(model.candidates(picture))
            auxiliary = select(masks(views[prompt]), geometry, identity, floor)
            native["frames"][row["id"]] = dict(accepted=auxiliary, seconds=perf_counter() - started)
            write_json(cache, native)
            print("Native crowd", row["id"], len(known[row["id"]]), len(auxiliary), flush=True)
    proposals = {
        "public development": read_json(DEVELOPMENT)["selected"]["accepted"],
        "public transfer": read_json(ART / "crowded_sam_transfer.json")["selected"]["accepted"],
        "native regression": {k: v["accepted"] for k, v in native["frames"].items()},
    }
    ledger = {}
    for name, (cohort, known) in groups.items():
        earlier = evaluate(cohort, known, 0.)
        outcome = {}
        for policy in POLICIES:
            predictions = {key: reconcile(value, proposals[name].get(key, []), policy)
                           for key, value in known.items()}
            result = evaluate(cohort, predictions, 0.)
            retention = compare(earlier, result)
            counts = {row["id"]: [len(row["objects"]), len(known[row["id"]]), len(predictions[row["id"]])]
                      for row in cohort}
            spoiled = [key for key, (reference, old, new) in counts.items() if reference == old != new]
            outcome[policy] = dict(metrics=short(result), counts=counts, same_matches=retention, spoiled=spoiled,
                                   passes=retention["passed"] and result["fp"] <= earlier["fp"] and not spoiled)
        ledger[name] = dict(current=short(earlier), policies=outcome)
    qualified = [policy for policy in POLICIES if all(v["policies"][policy]["passes"] for v in ledger.values())]
    times = [v["seconds"] for v in native["frames"].values()]
    write_json(RESULTS, dict(cohorts=ledger, automatic_candidates=qualified, freeze_sha256=digest(PLAN),
                             qualification_crowded_native=len(times), median_native_comparison_seconds=median(times),
                             scope="The native set represents one previously sealed acquisition. Supplemental "
                                   "matches measured against the same images are regression outcomes."))
    print("Crowd hybrid qualification", qualified, flush=True)


if __name__ == "__main__":
    run()
