"""Run new inference on the archived native scenes using a restored local runtime."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.grounding_backend import analyze
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from scripts.detection_device_scene_audit import score
from scripts.package_detection_evidence import MANIFEST, NATIVE, replay_native
from scripts.runtime_bundle import confined, git_revision
from scripts.sam3_study import configure_cpu


def sealed_cohort(bundle):
    manifest = read_json(bundle / MANIFEST)
    evidence = manifest["cohorts"][NATIVE]
    path = bundle / confined(evidence["path"])
    if digest(path) != evidence["sha256"]:
        raise ValueError("The archived native device cohort failed its checksum.")
    cohort = read_json(path)
    for row in cohort["rows"]:
        file = Path("data") / confined(row["image"])
        if cohort["raster_sha256"].get(str(file)) != row["sha256"] or digest(bundle / file) != row["sha256"]:
            raise ValueError(f"An archived source image changed: {row['id']}.")
    replay_native(cohort)
    return evidence, cohort


def infer(args):
    configure_cpu()
    bundle = args.evidence.resolve()
    identity, data = sealed_cohort(bundle)
    root = args.runtime.resolve()
    key = dict(revision=git_revision(root), appearance_release_sha256=digest(root / RELEASE),
               evidence_sha256=identity["sha256"], code_sha256=digest(__file__), device=args.device)
    saved = read_json(args.output) if args.output.exists() else dict(inputs=key, frames={})
    if saved.get("inputs") != key:
        raise ValueError("The saved inference used a different runtime or dataset.")
    todo = [row for row in data["rows"] if row["id"] not in saved["frames"]]
    if todo:
        engine = MedSyringeEngine(root)
        try:
            configure_device(engine, root, args.device)
            actual_device = engine.identity.get("primary_backend", "cpu")
            if actual_device != args.device:
                raise ValueError("The requested local device did not activate.")
            for row in todo[:args.limit]:
                identifier = row["id"]
                image = read_image(bundle / "data" / confined(row["image"]))
                start = perf_counter()
                report = analyze(engine, image, False)
                health = report["detection_checks"]
                if health["status"] != "available" or health["primary_backend"] != args.device:
                    raise ValueError(f"An inference worker was unavailable on {identifier}: {health}")
                saved["frames"][identifier] = dict(
                    actual=report["objects"], proposals=report["below_threshold_candidates"],
                    baseline=data["observations"][identifier]["actual"], health=health, time=perf_counter()-start)
                write_json(args.output, saved)
                print("Fresh offline inference", len(saved["frames"]), "/", len(data["rows"]), flush=True)
        finally:
            engine.close()
    if len(saved["frames"]) == len(data["rows"]):
        scopes = {
            "all": data["rows"],
            "multi": [row for row in data["rows"] if len(row["objects"]) > 1],
            "single": [row for row in data["rows"] if len(row["objects"]) == 1],
        }
        scores = {name: score(rows, saved["frames"]) for name, rows in scopes.items()}
        checks = {}
        for name, measured in scores.items():
            expected = data["expected"][name]
            same = all(c["before"] == c["actual"] for c in measured["counts"].values())
            checks[name] = dict(passed=same and measured["recall_guard"]["passed"] and
                                measured["released"]["fp"] == expected["released"]["fp"],
                                exact_counts_reproduced=same, **measured)
        saved.update(completed=True, passed=all(v["passed"] for v in checks.values()), scopes=checks,
                     warm_median_seconds=median(v["time"] for v in list(saved["frames"].values())[1:]))
        write_json(args.output, saved)
        print("Native operating results reproduced", saved["passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "xpu"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("The collection limit must be positive.")
    infer(args)
