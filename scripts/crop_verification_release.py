"""Audit the frozen verifier on the full capture and stage its live release."""
import argparse
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from medsyringe.crop_verifier_runtime import RELEASE
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.expanded_runtime import ExpandedRuntime
from medsyringe.grounding_backend import analyze, enable_intel
from medsyringe.production import ProductionEngine
from medsyringe.recall_guard import compare
from scripts.expanded_crop_verification import AUDIT, FLOOR_PATH, LOCK, examples, features, predictions
from scripts.expanded_detection_study import load, short
from scripts.expanded_proposals import ART, image_for, rows_for
from scripts.expanded_release import EVIDENCE as EXPANDED_EVIDENCE, RELEASE as EXPANDED_RELEASE, prepared_root
from scripts.expanded_runtime_pair import signature
from scripts.rfdetr_verifier import DINOV2, DINOV2_INFO, DinoEncoder
from scripts.sam3_study import configure_cpu

REGRESSION = ART / "crop_verification_full_capture.json"
PAIR = ART / "runtime_pair_cpu.json"
DRAFT = ART / "crop_verifier_candidate.json"
PROOF = ART / "crop_verifier_runtime.json"


def regress(source):
    study = read_json(LOCK)
    initial = read_json(AUDIT)
    if not initial["passed"] or initial["selection_sha256"] != digest(LOCK):
        raise ValueError("The verifier did not pass the fixed study.")
    choice = study["selected"]
    method = choice["method"]
    if method != "dino" or choice["protected_votes"] != 2:
        raise ValueError("The production verifier expects a DINO crop head and two-detector corroboration.")
    encoder = DinoEncoder()
    floor = read_json(FLOOR_PATH)["cutoff"]
    outputs = {}
    for split in ("fit", "calibration", "test", "reserve"):
        rows, frames = load(split)
        if split in ("fit", "calibration"):
            rows = [r for r in rows if "file" in r]
        baseline = {row["id"]: frames[row["id"]]["prior"] for row in rows}
        items = examples(rows, frames, floor, False)
        matrix = features(items, source, encoder)[method]
        accepted, removed = predictions(rows, items, matrix, study["heads"][method], choice["cutoff"],
                                        choice["protected_votes"])
        # Keep the baseline's subthreshold proposals when computing AP.
        after = {key: accepted[key] + [p for p in ranked if p["score"] < floor]
                 for key, ranked in baseline.items()}
        before, candidate = evaluate(rows, baseline, floor), evaluate(rows, after, floor)
        outputs[split] = dict(baseline=short(before), candidate=short(candidate),
                              instance_guard=compare(before, candidate), removed=removed)
        print(split, short(candidate), outputs[split]["instance_guard"], flush=True)
    allowed = all(o["instance_guard"]["passed"] and o["candidate"]["fp"] <= o["baseline"]["fp"]
                  for o in outputs.values())
    write_json(REGRESSION, dict(passed=allowed, results=outputs, selection_sha256=digest(LOCK),
               scope="Complete 177-image capture with cached baseline proposals plus fixed public controls. "
                     "Fit and calibration are included for regression safety. AP keeps unchanged proposals below "
                     "the operating cutoff and excludes verifier-rejected boxes."))
    dependencies = ("medsyringe/crop_verifier_runtime.py", "medsyringe/production.py", "medsyringe/inference.py",
                    "scripts/rfdetr_verifier.py", "scripts/expanded_worker.py", "scripts/expanded_worker_xpu.sh",
                    str(DINOV2_INFO))
    cached = read_json(PAIR)
    model = cached["outputs"][0]["result"]["model"]
    write_json(DRAFT, dict(id="guarded-dinov2-crop-verifier", head=study["heads"][method],
               cutoff=choice["cutoff"], model=model, code={name: digest(name) for name in dependencies},
               models={str(DINOV2 / name): checksum for name, checksum in read_json(DINOV2_INFO)["files"].items()},
               evidence=str(REGRESSION), evidence_sha256=digest(REGRESSION)))


def stable_observation(expected, actual):
    if actual["accepted_count"] != expected["accepted_count"]:
        return False
    for a, b in zip(expected["objects"], actual["objects"], strict=True):
        sa, sb = signature(a), signature(b)
        light = sa.pop("lightness"), sb.pop("lightness")
        stable_light = light[0] == light[1] or (None not in light and abs(light[0]-light[1]) <= 1.)
        if not (sa == sb and stable_light and box_iou(a["bbox"], b["bbox"]) >= .98):
            return False
    return True


def runtime(source, backend, group):
    if not read_json(REGRESSION)["passed"]:
        raise ValueError("The full capture regression did not pass.")
    cohort = {r["id"]: r for split in ("fit", "calibration", "test", "reserve") for r in rows_for(split)}
    original = read_json(PAIR)["outputs"]
    removed = [e["id"] for e in read_json(REGRESSION)["results"]["reserve"]["removed"]
               if not cohort[e["id"]]["objects"]]
    examples = [(r["id"], r["rotation"], r["result"]) for r in original]
    negative = next(k for k in removed if k not in {e[0] for e in examples})
    examples.append((negative, 0, None))
    examples = examples[:4] if group == "a" else examples[4:]
    cases = []
    with TemporaryDirectory(prefix="medsyringe-crop-release-") as scratch:
        root = Path(scratch)
        prepared_root(root)
        for path in (EXPANDED_RELEASE, EXPANDED_EVIDENCE):
            (root / path).symlink_to(path.resolve())
        write_json(root / RELEASE, read_json(DRAFT))
        engine = ProductionEngine(root)
        try:
            if backend == "xpu":
                enable_intel(engine, root)
            for name, angle, expected in examples:
                image = image_for(cohort[name], source)
                if angle:
                    image = image.transpose(Image.Transpose.ROTATE_180)
                actual = analyze(engine, image, True)
                baseline_recheck = None
                if expected is None:
                    suppressed = [p for p in actual["below_threshold_candidates"] if p.get("crop_score") is not None]
                    passed = actual["accepted_count"] == 0 and bool(suppressed)
                else:
                    passed = stable_observation(expected, actual)
                    if not passed and backend == "cpu":
                        # The paired accelerator study uses Transformers 5. The
                        # main CPU app uses the environment pinned by uv.lock.
                        installed = engine.expanded.detect
                        engine.expanded.detect = partial(ExpandedRuntime.detect, engine.expanded)
                        try:
                            baseline_recheck = analyze(engine, image, True)
                        finally:
                            engine.expanded.detect = installed
                        passed = stable_observation(baseline_recheck, actual)
                if backend == "xpu":
                    passed &= actual["model"].get("primary_backend") == "xpu"
                print(backend, name, angle, passed, actual["accepted_count"], round(actual["elapsed_ms"]), flush=True)
                cases.append(dict(id=name, orientation=angle, passed=passed,
                                  baseline_recheck=baseline_recheck, result=actual))
        finally:
            engine.close()
    write_json(ART / f"crop_verification_runtime_{backend}_{group}.json", dict(passed=all(c["passed"] for c in cases),
               candidate_sha256=digest(DRAFT), cases=cases))


def release():
    paths = [ART / f"crop_verification_runtime_{backend}_{group}.json"
             for backend in ("cpu", "xpu") for group in ("a", "b")]
    available = all(read_json(p)["candidate_sha256"] == digest(DRAFT) and read_json(p)["passed"] for p in paths)
    if not available or not read_json(REGRESSION)["passed"]:
        raise ValueError("The default verifier requires passing cached and live regressions.")
    write_json(PROOF, dict(passed=True, full_capture_sha256=digest(REGRESSION),
                           paired={str(p): digest(p) for p in paths}))
    write_json(RELEASE, {**read_json(DRAFT), "evidence": str(PROOF), "evidence_sha256": digest(PROOF)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("regress", "cpu", "xpu", "release"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--group", choices=("a", "b"), default="a")
    args = parser.parse_args()
    configure_cpu()
    if args.stage == "release":
        release()
    elif not args.source:
        parser.error("--source is required.")
    elif args.stage == "regress":
        regress(args.source)
    else:
        runtime(args.source, args.stage, args.group)


if __name__ == "__main__":
    main()
