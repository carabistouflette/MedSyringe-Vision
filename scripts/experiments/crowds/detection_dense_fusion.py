"""Compare dense and detector geometry on real development observations."""
import argparse
from itertools import product

import numpy as np

from medsyringe.detection.additive import overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_context_study import CACHE as CONTEXT_CACHE, inputs as context
from scripts.experiments.crowds.detection_dense_appearance import CACHE as DENSE, OUTPUT as DENSE_RESULTS, candidates
from scripts.experiments.detector_audits.detection_guided_geometry import references
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, cache, inputs, loaded
from scripts.experiments.expanded.expanded_detection_study import short

OUTPUT = ART / "dense_appearance_fusion_development.json"


def envelope(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def extend(accepted, dense, geometry):
    patch_floor, min_iou, max_growth, strength = geometry
    if len(accepted) != 1 or len(dense) != 1:
        return accepted
    found, patch = accepted[0], dense[0]
    first, second = found["bbox"], patch["bbox"]
    joined = envelope(first, second)
    if (patch["score"] < patch_floor or box_iou(first, second) < min_iou
            or overlap_fraction(first, second) < .65 or area(joined) > max_growth * area(first)):
        return accepted
    box = [(1. - strength) * a + strength * b for a, b in zip(first, joined, strict=True)]
    return [{**found, "bbox": box}]


def recover(accepted, dense, pending, setting, floor):
    patch_floor, witness_floor, witness_iou, geometry = setting
    if accepted or len(dense) != 1 or dense[0]["score"] < patch_floor:
        return accepted
    proposed = dense[0]["bbox"]
    agreement = [p for p in pending if witness_floor <= p["score"] < floor
                 and box_iou(proposed, p["bbox"]) >= witness_iou]
    witness = max(agreement, key=lambda p: (box_iou(proposed, p["bbox"]), p["score"]), default=None)
    if witness is None:
        return accepted
    box = proposed if geometry == "patch" else witness["bbox"]
    return [dict(bbox=box, score=floor)]


def run():
    if OUTPUT.exists():
        raise ValueError("The dense fusion pilot is already measured.")
    current, dense_results = read_json(CONTEXT_CACHE), read_json(DENSE_RESULTS)
    if dense_results["cache_sha256"] != digest(DENSE):
        raise ValueError("The previously observed dense heatmaps changed.")
    heatmaps = read_json(DENSE)["frames"]
    public = "public-development"
    base = loaded(public, "baseline")
    observed = {
        "transparent": (context(), current["frames"], {
            k: [p for p in v["ranked"] if p["score"] >= v["floor"]] for k, v in current["frames"].items()}),
        "public": (inputs(public), base, references(public))
    }
    comparator = {}
    for scope, (rows, frames, accepted) in observed.items():
        ids = {r["id"] for r in rows}
        if set(frames) != ids or set(heatmaps[scope]) != ids:
            raise ValueError("Dense and baseline observations must cover both complete development cohorts.")
        comparator[scope] = (evaluate(rows, accepted, 0.), {
            threshold: {k: candidates(np.asarray(v["heat"]), threshold, 0.) for k, v in heatmaps[scope].items()}
            for threshold in (.70, .85)})
    trial = []
    for threshold in (.70, .85):
        geometry = product((.82, .92), (.20, .40), (2., 3.), (.5, 1.))
        empty = product((.82, .92), (.23, .33), (.35, .55), ("patch", "witness"))
        for kind, options in (("extension", geometry), ("recovery", empty)):
            for setting in options:
                outcomes = {}
                for scope, (rows, frames, accepted) in observed.items():
                    baseline, new = comparator[scope]
                    output = {}
                    for key, frame in frames.items():
                        patch = new[threshold][key]
                        output[key] = (extend(accepted[key], patch, setting) if kind == "extension" else
                                       recover(accepted[key], patch, frame["ranked"], setting, frame["floor"]))
                    score = evaluate(rows, output, 0.)
                    guard = compare(baseline, score)
                    spoiled = [r["id"] for r in rows if len(accepted[r["id"]]) == len(r["objects"])
                               and len(output[r["id"]]) != len(r["objects"])]
                    outcomes[scope] = dict(metrics=short(score), guard=guard, spoiled=spoiled,
                                           changes=sum(output[k] != accepted[k] for k in output),
                                           qualifies=guard["passed"] and not spoiled and score["fp"] <= baseline["fp"])
                trial.append(dict(method=kind, cutoff=threshold, parameters=setting,
                                  outcomes=outcomes, qualifies=all(v["qualifies"] for v in outcomes.values())))
    leading = sorted([r for r in trial if r["qualifies"]], key=lambda r: (
        sum(v["guard"]["recovered_matches"] for v in r["outcomes"].values()),
        sum(v["metrics"]["ap50_95"] for v in r["outcomes"].values())), reverse=True)
    write_json(OUTPUT, dict(base={k: short(v[0]) for k, v in comparator.items()}, variants=len(trial),
                            safe_development=len(leading), leading=leading[:12],
                            input_files={str(p): digest(p) for p in (CONTEXT_CACHE, DENSE, DENSE_RESULTS,
                                                                   cache(public, "baseline"), __file__)},
                            description="Compare box extension and recovery based on complementary dense appearance. "
                                        "The transparent series is one development source. No reserve is used."))
    print("Dense fusion", len(leading), "/", len(trial),
          {k: v["guard"]["recovered_matches"] for k, v in leading[0]["outcomes"].items()} if leading else {},
          flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
