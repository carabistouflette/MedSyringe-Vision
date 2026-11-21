"""Qualify exact scene counts and retained localizations under shared SAM residency."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.grounding_backend import analyze
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.recall_guard import compare
from medsyringe.shared_sam import share
from scripts.detection_precision_confirmation import CAPTURE
from scripts.expanded_detection_study import short
from scripts.package_detection_evidence import MANIFEST
from scripts.reinfer_native_evidence import sealed_cohort
from scripts.runtime_bundle import confined, git_revision
from scripts.sam3_study import configure_cpu


def capture_groups(bundle, originals):
    manifest = read_json(bundle / MANIFEST)
    groups = {}
    for name in CAPTURE:
        metadata = manifest["cohorts"][name]
        doc = bundle / confined(metadata["path"])
        if digest(doc) != metadata["sha256"]:
            raise ValueError("A frozen capture partition changed.")
        saved = read_json(doc)
        rows = saved["rows"]
        baseline = {row["id"]: [p for p in saved["candidate"][row["id"]] if p["score"] >= saved["floor"]]
                    for row in rows}
        files = {row["id"]: originals / confined(row["file"]) for row in rows}
        groups[name] = (rows, files, baseline)
    return groups


def native_group(bundle):
    _, data = sealed_cohort(bundle)
    rows = data["rows"]
    files = {row["id"]: bundle / "data" / confined(row["image"]) for row in rows}
    baseline = {r["id"]: data["observations"][r["id"]]["actual"] for r in rows}
    return rows, files, baseline


def audit(args):
    configure_cpu()
    candidate_code = Path(__file__).resolve()
    runtime = args.runtime.resolve()
    bundle = args.evidence.resolve()
    cohorts = capture_groups(bundle, args.original_capture.resolve()) if args.original_capture else {}
    if args.include_native:
        cohorts["native-device"] = native_group(bundle)
    groups = tuple(cohorts)
    identity = dict(revision=git_revision(runtime), code_sha256=digest(candidate_code),
                    shared_worker_sha256=digest(candidate_code.with_name("shared_sam_worker.py")),
                    evidence_sha256=digest(bundle / MANIFEST), active_release_sha256=digest(runtime / RELEASE),
                    cohorts=list(groups), device="xpu")
    saved = read_json(args.output) if args.output.exists() else dict(inputs=identity, frames={})
    if saved["inputs"] != identity:
        raise ValueError("The candidate cache belongs to a different runtime or source manifest.")
    items = [(name, row, files[row["id"]]) for name, (rows, files, _) in cohorts.items() for row in rows]
    for _, row, file in items:
        if digest(file) != row["sha256"]:
            raise ValueError(f"The native source changed: {row['id']}.")
    remaining = [(name, r, file) for name, r, file in items if r["id"] not in saved["frames"]]
    if remaining:
        engine = MedSyringeEngine(runtime)
        try:
            configure_device(engine, runtime, "xpu")
            if engine.identity.get("primary_backend") != "xpu":
                raise ValueError("The runtime did not activate the verified Intel device.")
            share(engine, runtime)
            support = engine.crop_verifier.support_worker
            support.run = engine.detection_checks.observe(support.run, "SAM corroboration")
            for group, row, file in remaining[:args.limit]:
                started = perf_counter()
                result = analyze(engine, read_image(file), False)
                if result["detection_checks"]["status"] != "available":
                    raise ValueError(f"Unavailable detector for {row['id']}: {result['detection_checks']}")
                saved["frames"][row["id"]] = dict(
                    group=group, boxes=[dict(bbox=p["bbox"], score=p["score"]) for p in result["objects"]],
                    seconds=perf_counter()-started, health=result["detection_checks"])
                write_json(args.output, saved)
                print("Shared SAM live regression", len(saved["frames"]), "/", len(items), flush=True)
        finally:
            engine.close()
    if len(saved["frames"]) == len(items):
        checked = {}
        for name, (rows, _, before) in cohorts.items():
            after = {r["id"]: saved["frames"][r["id"]]["boxes"] for r in rows}
            old, new = [evaluate(rows, forecasts, 0.) for forecasts in (before, after)]
            check = compare(old, new)
            changed = [k for k in after if len(after[k]) != len(before[k])]
            timings = [saved["frames"][r["id"]]["seconds"] for r in rows]
            checked[name] = dict(passed=check["passed"] and not changed and new["fp"] <= old["fp"],
                                 changed_counts=changed, guard=check, old=short(old), new=short(new),
                                 median_seconds=median(timings), p95_seconds=float(np.percentile(timings, 95)))
        saved.update(completed=True, passed=all(v["passed"] for v in checked.values()), cohorts=checked)
        write_json(args.output, saved)
        print("Shared SAM full live guard", {k: v["passed"] for k, v in checked.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--runtime", type=Path, default=Path.cwd())
    parser.add_argument("--original-capture", type=Path)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.original_capture and not args.include_native:
        parser.error("Choose archived native scenes, the supplied capture, or both.")
    if args.limit is not None and args.limit < 1:
        parser.error("The optional limit must be positive.")
    audit(args)
