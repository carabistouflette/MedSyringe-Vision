"""Check subpixel parity and preserved evidence on separately fixed inputs."""
import argparse
import math
from pathlib import Path
from statistics import median

from medsyringe.data import digest, read_json, write_json
from scripts.checks.shared_sam_parity import ART, IMPLEMENTATION, ROOT

PREREGISTRATION = ART / "sam_residency_acceptance.json"
RESULT = ART / "sam_residency_confirmation.json"
BOX_TOLERANCE_PIXELS = .25
SCORE_TOLERANCE = .001
DETECTION_FIELDS = ("objects", "below_threshold_candidates")
SCORE_FIELDS = ("score", "original_score", "auxiliary_score", "crop_score", "semantic_margin")


def seal(images, long_side, ocr):
    if PREREGISTRATION.exists() or RESULT.exists():
        raise ValueError("The independent residency check has been sealed.")
    if len(images) < 3 or len({digest(path) for path in images}) != len(images):
        raise ValueError("Use at least three distinct source files.")
    write_json(PREREGISTRATION, dict(
        code={str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__).resolve(), *IMPLEMENTATION)},
        inputs=[dict(image=p.name, sha256=digest(p), long_side=long_side, ocr=ocr) for p in images],
        maximum_score_delta=SCORE_TOLERANCE, maximum_box_delta_pixels=BOX_TOLERANCE_PIXELS,
        protocol="The initial exact equality test remains archived. In this separate comparison, box differences "
                 "must fit within a quarter of one native pixel, and uncalibrated scores within 0.1 percentage "
                 "points. Counts, review order, evidence, attributes, and all other fields must agree exactly. "
                 "Latency is descriptive and excludes program loading. Prior examples are not confirmation data."))


def compare_reports(old, shared):
    differences = []
    deltas = dict(box_pixels=0., score=0.)
    dimensions = old["dimensions"]

    def walk(a, b, field=()):
        if a == b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences.append(list(field))
            for k in sorted(set(a) & set(b)):
                walk(a[k], b[k], (*field, k))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, (*field, index))
        elif (field and field[0] in DETECTION_FIELDS and isinstance(a, float) and isinstance(b, float)
              and math.isfinite(a) and math.isfinite(b)):
            difference = abs(a-b)
            if len(field) == 4 and field[-2] == "bbox":
                value = difference * dimensions[field[-1] % 2]
                deltas["box_pixels"] = max(value, deltas["box_pixels"])
                if value > BOX_TOLERANCE_PIXELS:
                    differences.append(list(field))
            elif len(field) == 3 and field[-1] in SCORE_FIELDS:
                deltas["score"] = max(difference, deltas["score"])
                if difference > SCORE_TOLERANCE:
                    differences.append(list(field))
            else:
                differences.append(list(field))
        else:
            differences.append(list(field))

    walk(old, shared)
    return dict(passed=not differences, deltas=deltas, differences=differences)


def measure(previous, new):
    if RESULT.exists():
        raise ValueError("The confirmation has already been measured.")
    design = read_json(PREREGISTRATION)
    if any(digest(p) != h for p, h in design["code"].items()):
        raise ValueError("Challenge inputs or code changed after registration.")
    old, shared = read_json(previous), read_json(new)
    if (not old["completed"] or not shared["completed"] or old["mode"] != "separate"
        or shared["mode"] != "shared" or old["device"] != shared["device"]):
        raise ValueError("A complete before and after run on the same device is required.")
    runs = []
    for control, treatment, inputs in zip(old["cases"], shared["cases"], design["inputs"], strict=True):
        if any(control[k] != v or treatment[k] != v for k, v in inputs.items()):
            raise ValueError("The supplied runs used a different workload.")
        marks = compare_reports(control["report"], treatment["report"])
        runs.append(dict(**marks, image=inputs["image"], counts=[
            control["report"]["accepted_count"], treatment["report"]["accepted_count"]],
            earlier_seconds=control["seconds"], candidate_seconds=treatment["seconds"]))
    warm = dict(earlier=median(v["earlier_seconds"] for v in runs[1:]),
                candidate=median(v["candidate_seconds"] for v in runs[1:]))
    write_json(RESULT, dict(passed=all(r["passed"] for r in runs), device=old["device"],
                            preregistration_sha256=digest(PREREGISTRATION), raw_sha256=[digest(previous), digest(new)],
                            warm_median_seconds=warm, checks=runs))
    print("Verified residency confirmation", [r["passed"] for r in runs], warm, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "measure"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--long-side", type=int, default=4032)
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    if args.action == "seal":
        seal(args.image, args.long_side, args.ocr)
    elif args.previous and args.candidate:
        measure(args.previous, args.candidate)
    else:
        parser.error("Pass both completed comparison runs.")
