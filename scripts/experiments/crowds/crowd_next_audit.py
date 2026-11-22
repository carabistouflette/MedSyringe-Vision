"""Evaluate one fixed crowd policy on independently reviewed public acquisitions."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from medsyringe.evaluate import evaluate
from medsyringe.runtime.grounding_backend import analyze
from medsyringe.runtime.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.crowds.crowded_sam_hybrid import reconcile
from scripts.experiments.crowds.crowded_sam_resolution import OUTPUT as DEVELOPMENT, CrowdedMasks, masks, select
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.expanded.sam3_study import LOCK, configure_cpu

REFERENCE = ART / "crowd_next_reference_manifest.json"
REVIEW = ART / "crowd_next_reference_second_review.json"
FREEZE = ART / "crowd_next_audit_freeze.json"
OUTPUT = ART / "crowd_next_audit_results.json"
PROTOCOL = Path("docs/crowd_next_audit_protocol.md")
CACHE = DATA / "proposals/crowded-sam/next"
CONFIG = [756, "syringe", "bbox", "mask", .50]
POLICY = "add unmatched"


def inputs():
    rows = read_json(REFERENCE)["images"]
    review = read_json(REVIEW)
    if review["manifest_sha256"] != digest(REFERENCE):
        raise ValueError("The second count review must bind to the frozen photos.")
    seen_ids, seen_files, groups = set(), set(), set()
    eligible = []
    for row in rows:
        asset = Path("data") / row["file"]
        if (row["id"] in seen_ids or row["sha256"] in seen_files or digest(asset) != row["sha256"]
                or not all(row[field] for field in ("group", "source_url", "license"))
                or any(not valid_box(obj["bbox"]) for obj in row["objects"])):
            raise ValueError(f"Invalid or duplicate crowd reference {row['id']}.")
        seen_ids.add(row["id"])
        seen_files.add(row["sha256"])
        if row["id"] in review["ambiguous"] or (
                row["id"] in review["counts"] and review["counts"][row["id"]] != len(row["objects"])):
            continue
        if len(row["objects"]) >= 2 and row["id"] not in review["counts"]:
            raise ValueError(f"Crowd scene {row['id']} has no independent count.")
        groups.add(row["group"])
        eligible.append(row)
    if not any(not row["objects"] for row in eligible) or not any(len(row["objects"]) >= 2 for row in eligible):
        raise ValueError("The agreed references need both hard negatives and crowd scenes.")
    return eligible, sorted(seen_ids - {row["id"] for row in eligible}), sorted(groups)


def freeze():
    if read_json(DEVELOPMENT)["selected"]["configuration"] != CONFIG:
        raise ValueError("The preregistered policy differs from the previous development choice.")
    eligible, disputed, groups = inputs()
    return dict(protocol=digest(PROTOCOL), code=digest(__file__), checkpoint_lock=digest(LOCK),
                development=digest(DEVELOPMENT), references=digest(REFERENCE), review=digest(REVIEW),
                policy=POLICY, configuration=CONFIG,
                included=[row["id"] for row in eligible], disputed=disputed, acquisitions=groups)


def require_freeze():
    stamp = read_json(FREEZE)
    if stamp != freeze():
        raise ValueError("A fixed evaluation input changed. Do not rerun on a modified reference.")
    return stamp


def collect(branch):
    stamp = require_freeze()
    path = CACHE / f"{branch}.json"
    record = read_json(path) if path.exists() else dict(freeze=digest(FREEZE), frames={})
    if record["freeze"] != digest(FREEZE):
        raise ValueError("Predictions use a different reference or protocol.")
    pending = [row for row in inputs()[0] if row["id"] not in record["frames"]]
    if not pending:
        return
    configure_cpu()
    engine = None
    try:
        if branch == "released":
            root = Path.cwd()
            engine = MedSyringeEngine(root)
            configure_device(engine, root, "xpu")
        else:
            edge, prompt, geometry, suppression, floor = stamp["configuration"]
            sam = CrowdedMasks(edge, "xpu")
        for row in pending:
            photo = read_image(Path("data") / row["file"])
            start = perf_counter()
            if engine is not None:
                result = analyze(engine, photo, False)
                checks = result["detection_checks"]
                if checks["primary_backend"] != "xpu" or checks["status"] != "available":
                    raise ValueError("The released Intel inference path was unavailable.")
                objects = result["objects"]
            else:
                stream = next(items for name, items in sam.candidates(photo) if name == prompt)
                objects = select(masks(stream), geometry, suppression, floor)
            record["frames"][row["id"]] = dict(objects=objects, seconds=perf_counter() - start)
            write_json(path, record)
            print(branch, row["id"], len(objects), round(record["frames"][row["id"]]["seconds"], 2), flush=True)
    finally:
        if engine is not None:
            engine.close()


def reports(rows, predictions):
    counts = defaultdict(list)
    for row in rows:
        predicted = len(predictions[row["id"]])
        expected = len(row["objects"])
        counts[row["group"]].append(dict(id=row["id"], expected=expected, predicted=predicted,
                                         absolute_error=abs(expected - predicted), exact=expected == predicted))
    return dict(groups=dict(counts), group_macro_exact=mean(mean(r["exact"] for r in frames)
                                                           for frames in counts.values()),
                count_mae=mean(r["absolute_error"] for frames in counts.values() for r in frames),
                crowded_exact=sum(r["exact"] for frames in counts.values() for r in frames if r["expected"] >= 2))


def measure():
    stamp = require_freeze()
    rows, _, _ = inputs()
    raw = {name: read_json(CACHE / f"{name}.json") for name in ("released", "sam")}
    ids = set(stamp["included"])
    if any(r["freeze"] != digest(FREEZE) or ids != set(r["frames"]) for r in raw.values()):
        raise ValueError("Both branches must be collected on every frozen photograph.")
    released, sam = ({key: f["objects"] for key, f in raw[name]["frames"].items()} for name in ("released", "sam"))
    arms = {"released": released, "sam alone": sam,
            POLICY: {key: reconcile(objects, sam[key], POLICY) for key, objects in released.items()}}
    measured = {name: evaluate(rows, predictions, 0.) for name, predictions in arms.items()}
    counts = {name: reports(rows, predictions) for name, predictions in arms.items()}
    before, candidate = measured["released"], measured[POLICY]
    old, new = counts["released"], counts[POLICY]
    retention = compare(before, candidate)
    checks = dict(no_lost_matches=retention["passed"], no_extra_false_positives=candidate["fp"] <= before["fp"],
                  negatives_preserved=candidate["correct_empty"] >= before["correct_empty"],
                  more_exact_crowds=new["crowded_exact"] > old["crowded_exact"],
                  more_exact_by_acquisition=new["group_macro_exact"] > old["group_macro_exact"],
                  no_previous_exact_count_lost=all(not a["exact"] or b["exact"] for group, frames in old["groups"].items()
                                                   for a, b in zip(frames, new["groups"][group], strict=True)))
    outcome = dict(freeze=digest(FREEZE), reference=stamp, checks=checks, passed=all(checks.values()),
                   retention=retention,
                   arms={name: dict(**short(stats), **counts[name], false_positives=stats["false_positives"],
                                    false_negatives=stats["false_negatives"]) for name, stats in measured.items()},
                   seconds_median={name: median(f["seconds"] for f in arm["frames"].values()) for name, arm in raw.items()})
    write_json(OUTPUT, outcome)
    print({name: short(value) for name, value in measured.items()}, checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("freeze", "released", "sam", "measure"))
    job = parser.parse_args().job
    if job == "freeze":
        if FREEZE.exists():
            require_freeze()
        else:
            write_json(FREEZE, freeze())
    elif job == "measure":
        measure()
    else:
        collect(job)
