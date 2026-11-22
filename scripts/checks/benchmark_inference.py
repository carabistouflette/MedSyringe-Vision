"""Measure paired latency and prediction parity with the released local models."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.data import digest, read_image, write_json
from medsyringe.runtime.grounding_backend import analyze
from medsyringe.runtime.hardware import configure_device
from medsyringe.measurement.liquid_fields import enrich_liquid_fields
from medsyringe.pipeline import MedSyringeEngine


BOX_TOLERANCE = .0002
SCORE_TOLERANCE = .001
DETECTOR_SCORES = {"score", "raw_score", "original_score", "crop_score",
                   "auxiliary_score", "branch_score", "verification_score"}


def prediction_digest(record):
    prediction = {k: v for k, v in record.items() if k != "elapsed_ms"}
    payload = json.dumps(prediction, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def parity(before, after):
    """Require identical decisions and allow subpixel drift in detector coordinates."""
    mismatches = []
    drift = {"bbox": 0., "score": 0.}

    def compare(first, second, path):
        if first == second:
            return
        if isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys():
            for key in sorted(first):
                compare(first[key], second[key], (*path, key))
            return
        if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
            for index, (a, b) in enumerate(zip(first, second, strict=True)):
                compare(a, b, (*path, index))
            return
        if len(path) >= 3 and path[0] in {"objects", "below_threshold_candidates"}:
            field = path[2]
            detection_score = field in DETECTOR_SCORES or (field == "detection_evidence"
                               and path[-1] in DETECTOR_SCORES)
            if field == "bbox" or detection_score:
                if type(first) in (int, float) and type(second) in (int, float):
                    metric = "bbox" if field == "bbox" else "score"
                    tolerance = BOX_TOLERANCE if metric == "bbox" else SCORE_TOLERANCE
                    drift[metric] = max(drift[metric], abs(first - second))
                    if math.isclose(first, second, abs_tol=tolerance, rel_tol=0):
                        return
        mismatches.append("/".join(map(str, path)))

    compare(before, after, ())
    return dict(passed=not mismatches, mismatch_paths=mismatches, max_absolute_drift=drift)


def detection_record(record):
    keys = ("model", "threshold", "dimensions", "accepted_count", "below_threshold_candidates",
            "sam3_status", "detection_checks")
    evidence = {key: record[key] for key in keys}
    evidence["objects"] = [
        {key: obj[key] for key in ("id", "bbox", "score", "detection_evidence", "origin")}
        for obj in record["objects"]
    ]
    return evidence


def measure(args):
    import torch

    torch.set_num_interop_threads(1)
    sources = [
        ("Fil Rouge capacity", args.capture_root / "IMG_3294.jpeg", True),
        ("Fil Rouge fill", args.capture_root / "IMG_3316.jpeg", True),
        ("Native crowded", Path("data/detection_hillclimb/device-native/fr4KaYX2NdrKAcwFd4Sh.jpg"), False),
        ("Public yellow", Path("data/public-annotated-collection/images/oi-9b1a594e0cea3393.jpg"), True),
        ("Public negative", Path("data/detection_hillclimb/public/images/23953605.jpg"), False),
    ]
    if args.cases:
        sources = [source for source in sources if source[0] in args.cases]
        if len(sources) != len(set(args.cases)):
            raise ValueError("Choose existing case names to filter the panel.")
    images = {name: (read_image(path), ocr) for name, path, ocr in sources}
    modes = ("optimized", "count-only") if args.compare_count_only else ("baseline", "optimized")
    first = args.first or "optimized"
    result = dict(device=args.device, modes=modes, fixed_primary=args.fixed_primary, cases={
        name: dict(filename=path.name, sha256=digest(path), ocr=ocr,
                   dimensions=list(images[name][0].size)) for name, path, ocr in sources
    }, runs=[])
    started = perf_counter()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), args.device)
        if engine.identity.get("primary_backend", "cpu") != args.device:
            raise RuntimeError("The requested backend is unavailable.")
        result.update(setup_seconds=perf_counter() - started, model=dict(engine.identity))
        if args.fixed_primary:
            # Isolate the changed code from run-to-run drift in the unchanged proposer.
            ranks = {id(image): engine.rank_detections(image) for image, _ in images.values()}
            engine.rank_detections = lambda image: deepcopy(ranks[id(image)])
        packed = engine.appearance.pack_crops
        original = engine.appearance.worker.run
        transport = []

        def record_transport(image, **kwargs):
            transport.append(dict(mode=kwargs.get("operation", "full scene"), dimensions=list(image.size)))
            return original(image, **kwargs)

        engine.appearance.worker.run = record_transport

        def run(case, mode, round_number, cold=False):
            engine.overlap_crop_checks = mode != "baseline"
            engine.appearance.pack_crops = packed and mode != "baseline"
            image, ocr = images[case]
            transport.clear()
            started = perf_counter()
            if mode == "count-only":
                record = analyze(engine, image, ocr, count_only=True)
                if record.get("analysis_mode") != "count only" or any(
                    obj["scale"] is not None or obj["liquid_appearance"] is not None
                    or obj["inscriptions"]["status"] != "not requested" for obj in record["objects"]
                ):
                    raise RuntimeError("Unexpected attribute from the count-only mode.")
            else:
                record = analyze(engine, image, ocr)
                enrich_liquid_fields(image, record["objects"])
            seconds = perf_counter() - started
            check = record["detection_checks"]
            if check["status"] != "available" or check["primary_backend"] != args.device:
                raise RuntimeError(f"Inference degraded for {case}: {check}")
            record.pop("elapsed_ms")
            row = dict(case=case, mode=mode, round=round_number, process_cold=cold,
                       seconds=seconds, prediction_sha256=prediction_digest(record),
                       accepted_count=record["accepted_count"], appearance_payloads=list(transport),
                       prediction=record)
            result["runs"].append(row)
            write_json(args.output, result)
            print(case, mode, round_number, round(seconds, 3), row["accepted_count"], flush=True)

        run(sources[0][0], first, -1, cold=True)
        if not args.cold_only:
            for round_number in range(args.pairs):
                for index, (case, _, _) in enumerate(sources):
                    sequence = modes[::-1] if (index + round_number) % 2 else modes
                    for mode in sequence:
                        run(case, mode, round_number)
        summaries = {}
        for case in images:
            runs = [row for row in result["runs"] if row["case"] == case and not row["process_cold"]]
            if not runs:
                continue
            times = {mode: median(row["seconds"] for row in runs if row["mode"] == mode)
                     for mode in modes}
            pairs = []
            for round_number in range(args.pairs):
                paired = {r["mode"]: r["prediction"] for r in runs if r["round"] == round_number}
                before, after = (paired[mode] for mode in modes)
                if args.compare_count_only:
                    before, after = detection_record(before), detection_record(after)
                pairs.append(parity(before, after))
            hashes = {row["prediction_sha256"] for row in runs}
            summaries[case] = dict(**times, operational_parity=all(p["passed"] for p in pairs),
                                   same_prediction=len(hashes) == 1, paired_checks=pairs,
                                   speedup=times[modes[0]] / times[modes[1]])
        result.update(completed=True, passed=all(
            case["operational_parity"] for case in summaries.values()
        ), paired_warm_results=summaries,
            tolerances=dict(detector_bbox=BOX_TOLERANCE, detector_score=SCORE_TOLERANCE))
        write_json(args.output, result)
        if summaries and not result["passed"]:
            raise SystemExit("At least one complete prediction differed.")
    finally:
        engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "xpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--first", choices=("baseline", "optimized", "count-only"))
    parser.add_argument("--case", dest="cases", action="append", help="Repeat to select a case subset.")
    parser.add_argument("--compare-count-only", action="store_true")
    parser.add_argument("--fixed-primary", action="store_true",
                        help="Replay one primary proposal list per input to isolate parity.")
    parser.add_argument("--cold-only", action="store_true")
    options = parser.parse_args()
    if options.pairs < 1:
        parser.error("Choose at least one paired measurement.")
    measure(options)
