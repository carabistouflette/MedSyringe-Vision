"""Separate count recovery from box refinement on archived public predictions."""
import argparse
from itertools import product

from medsyringe.additive import duplicates, overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate, match
from scripts.detection_hillclimb_experiment import cache
from scripts.occlusion_detector import ART, DATA, FREEZE, RESULTS, frozen_frames, partitions

DEVELOPMENT = DATA / "development-predictions.json"
SCORED = DATA / "scored-predictions.json"
DENTAL = DATA.parent / "detection_hillclimb/proposals/dental-transfer-development-baseline.json"
POLICY = ART / "occlusion_relocalization_policy.json"
OUTPUT = ART / "occlusion_relocalization_results.json"
TILED = ("initial-tiled", "specialist-tiled")


def area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def replacements(baseline, ranked, peer, limits):
    """Propose one refined box while preserving the current instance and score."""
    if len(baseline) != 1:
        return list(baseline)
    current, = baseline
    ref = current["bbox"]
    candidates = []
    for candidate in ranked:
        if candidate["score"] < limits["floor"]:
            continue
        box = candidate["bbox"]
        if area(box) > limits["maximum_area_ratio"]*area(ref):
            continue
        if overlap_fraction(ref, box) < limits["containment"]:
            continue
        if limits["peer_floor"] is not None and not any(
            other["score"] >= limits["peer_floor"] and box_iou(box, other["bbox"]) >= .50 for other in peer
        ):
            continue
        if any(other["score"] >= limits["floor"] and duplicates(ref, other["bbox"])
               and not duplicates(box, other["bbox"]) for other in ranked):
            continue
        candidates.append(candidate)
    if not candidates:
        return list(baseline)
    best = max(candidates, key=lambda p: p["score"])
    return [{**current, "bbox": best["bbox"], "prior_bbox": ref, "refinement_score": best["score"]}]


def additions(baseline, original, ranked, floor):
    """Retain occupied regions from before refinement when checking additional instances."""
    result = list(baseline)
    for candidate in sorted(ranked, key=lambda p: -p["score"]):
        if candidate["score"] < floor:
            break
        if any(duplicates(candidate["bbox"], p["bbox"]) for p in [*original, *result]):
            continue
        result.append({**candidate, "score": min(candidate["score"], .01)})
    return result


def targets(rows, predictions, minimum):
    _, used = match(rows, predictions, minimum)
    return {(key, index) for key, indexes in used.items() for index in indexes}


def diagnostics(rows, baseline, outcomes, detailed=False):
    summary = {}
    for key, minimum in (("iou50", .50), ("iou75", .75)):
        old = targets(rows, baseline, minimum)
        new = targets(rows, outcomes, minimum)
        summary[key] = dict(matched=len(new), gained=sorted(new-old), lost=sorted(old-new))
    counts = dict(changed_boxes=sum("prior_bbox" in p for predictions in outcomes.values() for p in predictions),
                  added_boxes=sum(len(outcomes[k])-len(v) for k, v in baseline.items()))
    summary.update(counts)
    if detailed:
        summary["metrics"] = evaluate(rows, outcomes, 0.)
    return summary


def hypotheses(rows, baseline, predictions, policy, floors):
    refined, appended, combined = {}, {}, {}
    arm = policy["model"]
    companion = next(key for key in TILED if key != arm)
    for row in rows:
        key = row["id"]
        original, ranked = baseline[key], predictions[arm][key]
        refined[key] = replacements(original, ranked, predictions[companion][key], policy)
        appended[key] = additions(original, original, ranked, floors[arm])
        combined[key] = additions(refined[key], original, ranked, floors[arm])
    return {"refine": refined, "append": appended, "combined": combined}


def prepare_reference():
    _, dev, _ = partitions()
    paths = [cache(domain, "baseline") for domain in ("phlebotomy-development", "public-development")]
    paths.append(DENTAL)
    archive, systems = {}, set()
    for path in paths:
        record = read_json(path)
        systems.add(record.get("detector_sha256") or record["key"]["weights_sha256"])
        if set(archive) & set(record["frames"]):
            raise ValueError("Development baseline archives have duplicate identifiers.")
        archive.update(record["frames"])
    if len(systems) != 1 or set(archive) != {r["id"] for r in dev}:
        raise ValueError("Development baseline must cover all rows with one historical detector.")
    baseline = {key: [p for p in entry["ranked"] if p["score"] >= entry["floor"]]
                for key, entry in archive.items()}
    return dev, baseline, {str(p): digest(p) for p in paths}, systems.pop()


def select():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    rows, baseline, archives, version = prepare_reference()
    floors = read_json(FREEZE)["floors"]
    predictions = read_json(DEVELOPMENT)
    controls = {}
    trial_results = []
    for arm, score_multiplier, maximum_area, containment, corroborate in product(
        TILED, (.50, .75, 1.), (.60, .80, 1.), (.75, .90), (False, True)
    ):
        peer = next(key for key in TILED if key != arm)
        candidate = dict(model=arm, floor=floors[arm]*score_multiplier, maximum_area_ratio=maximum_area,
                         containment=containment, peer_floor=floors[peer]*.5 if corroborate else None)
        trial = hypotheses(rows, baseline, predictions, candidate, floors)
        trial_results.append(dict(policy=candidate, **diagnostics(rows, baseline, trial["refine"])))
        if arm not in controls:
            controls[arm] = diagnostics(rows, baseline, trial["append"], detailed=True)
    surviving = [trial for trial in trial_results if trial["changed_boxes"]
                 and not trial["iou50"]["lost"] and not trial["iou75"]["lost"]]
    winner = max(surviving, key=lambda t: (len(t["iou50"]["gained"]), len(t["iou75"]["gained"]),
                                          -t["changed_boxes"]), default=None)
    selected = winner if winner and (winner["iou50"]["gained"] or winner["iou75"]["gained"]) else None
    write_json(POLICY, dict(code_sha256=digest(__file__), rfdetr_selection=digest(FREEZE),
                            development_cache=digest(DEVELOPMENT), archives=archives, source_version=version,
                            development_groups=len({r["group"] for r in rows}), selected=selected,
                            trials=trial_results, additive_controls=controls,
                            selection_rule="At least one gained reference at IoU 0.50 or 0.75. Keep all existing "
                                           "matches on the 60 development photos at both IoU cutoffs."))
    print("Frozen on development:", selected, flush=True)


def measure():
    locked = read_json(POLICY)
    if (locked["code_sha256"] != digest(__file__) or locked["rfdetr_selection"] != digest(FREEZE)
            or locked["development_cache"] != digest(DEVELOPMENT)
            or any(digest(p) != checksum for p, checksum in locked["archives"].items())):
        raise ValueError("Frozen protocol or development data changed.")
    cohorts, released = frozen_frames()
    preds = read_json(SCORED)
    floors = read_json(FREEZE)["floors"]
    summaries = {}
    for role, rows in cohorts.items():
        baseline = {r["id"]: released[r["id"]] for r in rows}
        report = {"baseline": evaluate(rows, baseline, 0.)}
        if locked["selected"]:
            outputs = hypotheses(rows, baseline, preds, locked["selected"]["policy"], floors)
            report.update({name: diagnostics(rows, baseline, view, detailed=True)
                           for name, view in outputs.items()})
        else:
            report["additive_controls"] = {
                arm: diagnostics(rows, baseline, {
                    key: additions(old, old, preds[arm][key], floors[arm]) for key, old in baseline.items()
                }, detailed=True) for arm in TILED
            }
        summaries[role] = report
    write_json(OUTPUT, dict(policy_sha256=digest(POLICY), archived_rf_results=digest(RESULTS), cohorts=summaries,
                            scope="Retrospective replay on previously exposed cohorts. New release evidence is pending."))
    print("Replay finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "measure"))
    {"select": select, "measure": measure}[parser.parse_args().stage]()
