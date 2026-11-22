"""Freeze active diagnostic rules on development and explain transfer errors."""
import argparse

import cv2

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate, match, scores
from scripts.native_candidate_verifier import simple_metrics
from scripts.syringe_action_verifier import ART, POLICY, apply, audit, rank, sealed

LOCK = ART / "syringe_action_diagnostic_policy.json"
REPORT = ART / "syringe_action_diagnostic_results.json"


def freeze():
    if LOCK.exists():
        raise FileExistsError(LOCK)
    p = sealed()
    targets = p["development"]["metrics"]["objects"]

    def utility(t):
        f1 = scores(t["tp"], t["fp"], targets - t["tp"])["f1"]
        return (f1, len(t["gains_at_75"]) - len(t["losses_at_75"]),
                -sum(t["actions"].values()), -t["extra_fp"], t["rule"]["overlap"] == "strict")

    diagnostic = {}
    for key in p["finalists"]:
        feature, arm = key.split("/")
        candidates = [t for t in p["trials"] if t["rule"]["arm"] == arm and
                      t["rule"]["features"] == feature and t["actions"] and
                      (arm != "combined" or len(t["actions"]) == 2)]
        if candidates:
            diagnostic[key] = max(candidates, key=utility)
    write_json(LOCK, dict(policy=digest(POLICY), code=digest(__file__), active=diagnostic))
    print("Exploratory rules frozen", {k: t["rule"] for k, t in diagnostic.items()}, flush=True)


def classify_actions(rows, original, output, events):
    expected = {r["id"]: r["objects"] for r in rows}
    assignments, _ = match(rows, output, .50)
    position = {(k, tuple(p["bbox"]), p["score"]): i for k, p, i, _ in assignments}
    _, used = match(rows, original, .50)
    report = {}
    for key, items in events.items():
        details = []
        for event in items:
            b = event["bbox"]
            target_overlap = [box_iou(b, o["bbox"]) for o in expected[key]]
            best = max(target_overlap, default=0.)
            if event["action"] == "add":
                pred = next(p for p in output[key] if p["bbox"] == b)
                matched = position[key, tuple(b), pred["score"]]
                effects = dict(matched_target=matched, already_covered=matched in used[key],
                               best_iou=best, new_target=matched is not None and matched not in used[key])
            else:
                prior = event["prior_bbox"]
                previous = [box_iou(prior, o["bbox"]) for o in expected[key]]
                slot_target = max(range(len(previous)), key=previous.__getitem__, default=None)
                effects = dict(prior_best_iou=max(previous, default=0.), best_iou=best,
                               before_target=slot_target,
                               after_iou_with_same_target=target_overlap[slot_target] if slot_target is not None else None)
            details.append({**event, **effects})
        report[key] = details
    return report


def diagnose(rows, baseline, frames, models, arms):
    ranked = rank(rows, baseline, frames, models)
    before, (_, before75) = simple_metrics(rows, baseline), match(rows, baseline, .75)
    missing = {}
    for row in rows:
        key = row["id"]
        for i, obj in enumerate(row["objects"]):
            if i in before["targets"][key]:
                continue
            candidates = [(box_iou(p["bbox"], obj["bbox"]), p) for p in ranked[key]]
            attainable = [(iou, p) for iou, p in candidates if iou >= .50]
            strongest = {}
            for model in models:
                for action in ("add", "replace"):
                    chosen = max(attainable, key=lambda t: t[1]["decisions"][model][action], default=None)
                    if chosen:
                        overlap, p = chosen
                        strongest[f"{model}/{action}"] = dict(iou=overlap, **p)
            missing.setdefault(key, []).append(dict(index=i, maximum_iou=max((i for i, _ in candidates), default=0.),
                                                    viable_boxes=len(attainable), strongest=strongest))
    results = {}
    for key, trial in arms.items():
        output, events = apply(baseline, ranked, trial["rule"])
        results[key] = dict(delta=audit(rows, before, before75, output, events),
                            metrics=evaluate(rows, output, 0.),
                            actions=classify_actions(rows, baseline, output, events))
    return dict(baseline=evaluate(rows, baseline, 0.), arms=results, baseline_misses=missing)


def score():
    from scripts.completeness_calibration_reserve import evidence as previous
    from scripts.relocalization_reserve import evidence as crowd
    from scripts.syringe_action_reserve import evidence

    p, protocol = sealed(), read_json(LOCK)
    if protocol["policy"] != digest(POLICY) or protocol["code"] != digest(__file__):
        raise ValueError("Diagnostic policy changed.")
    cv2.setNumThreads(1)
    reports, provenance = {}, {}
    for name, source in (("reserve", evidence), ("exposed_calibration", previous), ("exposed_crowd", crowd)):
        rows, base, frames, digests = source()
        if name == "exposed_calibration":
            frames = {key: dict(seeds=p, proposals=[]) for key, p in frames.items()}
        reports[name] = diagnose(rows, base, frames, p["heads"], protocol["active"])
        provenance.update(digests)
    write_json(REPORT, dict(protocol=digest(LOCK), inputs=provenance, cohorts=reports))
    for name, cohort in reports.items():
        print(name, {k: (v["delta"]["gain"], v["delta"]["extra_fp"], v["delta"]["actions"])
                     for k, v in cohort["arms"].items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score"))
    {"freeze": freeze, "score": score}[cli.parse_args().stage]()
