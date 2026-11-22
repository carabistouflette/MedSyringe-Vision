"""Replay RF-DETR and SAM agreement at a predeclared operating point."""
import argparse

from medsyringe.additive import duplicates
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.occlusion_detector import ART, DATA, FREEZE, FRESH, RESULTS, frozen_frames, target_ids
from scripts.occlusion_sam_tiling import CACHE, POLICY

SEAL = ART / "occlusion_hybrid_policy.json"
OUTPUT = ART / "occlusion_hybrid_results.json"


def policy():
    return dict(implementation=digest(__file__), reserve=digest(FRESH),
                arms=["initial-tiled", "specialist-tiled"], sam_floor=.50,
                rfdetr_floor_multiplier=.5, agreement_iou=.50, route_maximum=1)


def select():
    if SEAL.exists():
        raise FileExistsError(SEAL)
    write_json(SEAL, policy())


def combine(baseline, sam, detector, floor, rules):
    result = list(baseline)
    for candidate in sorted(sam, key=lambda p: -p["score"]):
        if candidate["score"] < rules["sam_floor"]:
            continue
        box = candidate["bbox"]
        if any(duplicates(box, p["bbox"]) for p in result):
            continue
        if any(p["score"] >= floor and box_iou(box, p["bbox"]) >= rules["agreement_iou"] for p in detector):
            result.append(dict(bbox=box, score=min(candidate["score"], .01)))
    return result


def score():
    rule = read_json(SEAL)
    if rule != policy():
        raise ValueError("Frozen hybrid agreement policy changed.")
    floors = read_json(FREEZE)["floors"]
    if read_json(RESULTS)["selection_sha256"] != digest(FREEZE):
        raise ValueError("RF-DETR replay was not built with the frozen operating points.")
    fd = read_json(DATA / "scored-predictions.json")
    sg = read_json(CACHE)
    if sg["policy"] != digest(POLICY):
        raise ValueError("SAM replay was not built with the frozen tiling policy.")
    cohorts, released = frozen_frames()
    report = {}
    for role, rows in cohorts.items():
        base = {r["id"]: released[r["id"]] for r in rows}
        metrics = evaluate(rows, base, 0.)
        initial_matches = target_ids(rows, base)
        variants = {}
        for arm in rule["arms"]:
            floor = floors[arm]*rule["rfdetr_floor_multiplier"]
            outputs = {}
            for key, existing in base.items():
                outputs[key] = (combine(existing, sg["frames"][key]["tiles"], fd[arm][key], floor, rule)
                                if len(existing) <= rule["route_maximum"] else existing)
            result = evaluate(rows, outputs, 0.)
            variants[arm] = dict(metrics=result, added_false_positives=result["fp"]-metrics["fp"],
                                 lost_targets=sorted(initial_matches-target_ids(rows, outputs)))
        report[role] = dict(released=metrics, arms=variants)
    reserve = report["new-public-reserve"]
    gate = {arm: reserve["arms"][arm]["metrics"]["tp"] > reserve["released"]["tp"]
            and all(item["arms"][arm]["added_false_positives"] <= 0 and not item["arms"][arm]["lost_targets"]
                    for item in report.values()) for arm in rule["arms"]}
    write_json(OUTPUT, dict(policy_sha256=digest(SEAL), sam_cache_sha256=digest(CACHE),
                            rfdetr_replay_sha256=digest(RESULTS), cohorts=report, meets_release_gate=gate))
    print(gate, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "score"))
    {"select": select, "score": score}[parser.parse_args().stage]()
