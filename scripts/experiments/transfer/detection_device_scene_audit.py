"""Audit the delivered detector on sealed native, single- and multi-syringe scenes."""
import argparse
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from medsyringe.runtime.appearance_runtime import RELEASE
from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.evaluate import evaluate, match
from medsyringe.runtime.grounding_backend import analyze, enable_intel
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.transfer.detection_device_scene_data import MANIFEST
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, EXISTING
from scripts.experiments.transfer.detection_real_box_transfer import benchmark_sources
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

LOCK = ART / "device_scene_audit_freeze.json"
OUTPUT = ART / "device_scene_audit_results.json"
CACHE = DATA / "proposals/device-scene-release.json"


def freeze():
    if LOCK.exists() or CACHE.exists() or OUTPUT.exists():
        raise ValueError("The native scene audit has already begun.")
    refs = read_json(MANIFEST)["rows"]
    real, sim, dev = benchmark_sources()
    training = [{**r, "split": "fit"} for r in real + sim]
    if leakage([*training, *dev, *refs]):
        raise ValueError("A test source crosses the supervised folds.")
    inputs = (Path(__file__), MANIFEST, EXISTING, RELEASE, Path("medsyringe/pipeline.py"),
              Path("medsyringe/analysis/inference.py"), Path("medsyringe/detection/detection_checks.py"))
    write_json(LOCK, dict(inputs={str(p): digest(p) for p in inputs},
                          primary="Exact visible object count, overall and on multi-device frames.",
                          secondary="Matched instance precision and recall at IoU 0.50, absolute count error and "
                                    "candidate reach of the observed human-review interface.",
                          protocol="Run the shipped engine with its fixed detector operating point. Compare its "
                                   "accepted boxes with the former appearance policy before the SigLIP veto. "
                                   "This is one acquisition source. No operating point selection or tuning."))


def frozen():
    plan = read_json(LOCK)
    if any(digest(path) != expected for path, expected in plan["inputs"].items()):
        raise ValueError("The sealed device audit has modified inputs.")
    return read_json(MANIFEST)["rows"]


def collect():
    refs = frozen()
    exact = digest(LOCK)
    saved = read_json(CACHE) if CACHE.exists() else dict(freeze_sha256=exact, frames={})
    if saved["freeze_sha256"] != exact:
        raise ValueError("Previously saved frames belong to a different audit.")
    todo = [r for r in refs if r["id"] not in saved["frames"]]
    if not todo:
        return
    configure_cpu()
    root = Path.cwd()
    engine = MedSyringeEngine(root)
    try:
        enable_intel(engine, root)
        if engine.identity.get("primary_backend") != "xpu" or engine.appearance is None or engine.verifier:
            raise ValueError("Expected the delivered Intel path with its appearance verifier.")
        old = engine.appearance.detector
        observed = {}

        def observe(image):
            ranked, floor, supplemental = old(image)
            observed["baseline"] = [p for p in ranked if p["score"] >= floor]
            return ranked, floor, supplemental

        engine.appearance.detector = observe
        for row in todo:
            picture = source(row)
            begin = perf_counter()
            report = analyze(engine, picture, False)
            health = report["detection_checks"]
            if health["status"] != "available" or "baseline" not in observed:
                raise RuntimeError(f"An audit inference worker was unavailable: {row['id']}: {health}")
            prior = observed.pop("baseline")
            actual = report["objects"]
            if len(actual) != report["accepted_count"]:
                raise ValueError("The report does not agree with its accepted objects.")
            saved["frames"][row["id"]] = dict(
                baseline=prior, actual=actual, proposals=report["below_threshold_candidates"],
                health=health, time=perf_counter() - begin)
            write_json(CACHE, saved)
            print("Native scene audit", len(saved["frames"]), "/", len(refs), flush=True)
    finally:
        engine.close()


def score(rows, frames):
    before = {r["id"]: frames[r["id"]]["baseline"] for r in rows}
    accepted = {r["id"]: frames[r["id"]]["actual"] for r in rows}
    baseline, result = [evaluate(rows, predictions, 0.0) for predictions in (before, accepted)]
    missed = {}
    for item in result["false_negatives"]:
        missed.setdefault(item["id"], []).append(dict(bbox=item["box"]))
    recovery_refs = [dict(id=identifier, objects=objects) for identifier, objects in missed.items()]
    reach = {}
    for limit in (2, 5):
        potential = {k: frames[k]["proposals"][:limit] for k in missed}
        _, assignments = match(recovery_refs, potential, .50)
        reach[str(limit)] = sum(len(used) for used in assignments.values())
    return dict(baseline=short(baseline), released=short(result), recall_guard=compare(baseline, result),
                accepted_box_rank_only=True, missed_instances_in_review_proposals=reach,
                mean_absolute_count_error=mean(abs(len(accepted[r["id"]]) - len(r["objects"])) for r in rows),
                errors=dict(false_positives=result["false_positives"], false_negatives=result["false_negatives"]),
                counts={r["id"]: dict(reference=len(r["objects"]), before=len(before[r["id"]]),
                                      actual=len(accepted[r["id"]])) for r in rows})


def measure():
    refs = frozen()
    saved = read_json(CACHE)
    frames = saved["frames"]
    if OUTPUT.exists() or saved["freeze_sha256"] != digest(LOCK) or set(frames) != {r["id"] for r in refs}:
        raise ValueError("Expected a complete sealed collection and a new result path.")
    scopes = {"all": refs, "multi": [r for r in refs if len(r["objects"]) > 1],
              "single": [r for r in refs if len(r["objects"]) == 1]}
    findings = {name: score(rows, frames) for name, rows in scopes.items()}
    write_json(OUTPUT, dict(freeze_sha256=digest(LOCK), observations_sha256=digest(CACHE),
                            independent_source_groups=1, warm_median_seconds=median(
                                frames[r["id"]]["time"] for r in refs[1:]), scopes=findings))
    print({name: proof["released"] for name, proof in findings.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "collect", "measure"))
    {"freeze": freeze, "collect": collect, "measure": measure}[parser.parse_args().action]()
