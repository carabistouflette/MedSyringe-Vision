"""Test an orientation-rescue fallback on natural development inputs."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

from medsyringe.detection.additive import duplicates
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.runtime.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_context_study import CACHE as CONTEXT_CACHE, inputs as context
from scripts.experiments.detector_audits.detection_deskew_study import CACHE as PARALLEL_CACHE, OUTPUT as PARALLEL_RESULTS, aligned_detection, orientation
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, inputs
from scripts.experiments.detector_audits.detection_mosaic_study import run_baseline
from scripts.experiments.detector_audits.detection_parallel_development import OUTPUT as PARALLEL
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

CACHE = DATA / "proposals/natural-deskew-development.json"
OUTPUT = ART / "natural_deskew_development.json"


def cases():
    return dict(public=inputs("public-development"), transparent=context())


def binding():
    return dict(context_sha256=digest(CONTEXT_CACHE), parallel_sha256=digest(PARALLEL_RESULTS),
                implementation_sha256=digest(__file__),
                references={r["id"]: [r["sha256"], [obj["bbox"] for obj in r["objects"]]]
                            for rows in cases().values() for r in rows})


def collect():
    configure_cpu()
    key = binding()
    saved = read_json(CACHE) if CACHE.exists() else dict(binding=key, frames={})
    if saved["binding"] != key:
        raise ValueError("The development source or orientation method changed.")
    current = read_json(CONTEXT_CACHE)["frames"]
    root = Path.cwd()
    engine = MedSyringeEngine(root)
    try:
        configure_device(engine, root, "xpu")
        if engine.identity.get("primary_backend") != "xpu":
            raise ValueError("This prospective comparison uses the pinned Intel runtime.")
        for cohort, rows in cases().items():
            for row in rows:
                name = row["id"]
                if name in saved["frames"]:
                    continue
                picture = source(row)
                start = perf_counter()
                if cohort == "transparent":
                    baseline = [p for p in current[name]["ranked"] if p["score"] >= current[name]["floor"]]
                else:
                    baseline = run_baseline(picture, engine)
                angle = orientation(picture) if not baseline else None
                supplemental = aligned_detection(picture, angle, engine) if angle is not None else []
                saved["frames"][name] = dict(baseline=baseline, supplemental=supplemental,
                                             angle=angle, seconds=perf_counter() - start)
                write_json(CACHE, saved)
                print("Natural deskew", cohort, len(saved["frames"]), flush=True)
    finally:
        engine.close()


def stats(rows, before, revised):
    base = evaluate(rows, before, 0.)
    candidate = evaluate(rows, revised, 0.)
    spoiled = [r["id"] for r in rows if len(before[r["id"]]) == len(r["objects"])
               and len(revised[r["id"]]) != len(r["objects"])]
    return dict(baseline=short(base), prospective=short(candidate), gate=compare(base, candidate),
                spoiled_counts=spoiled)


def measure():
    if OUTPUT.exists():
        raise ValueError("The fixed natural development comparison already exists.")
    saved = read_json(CACHE)
    groups = cases()
    if saved["binding"] != binding() or set(saved["frames"]) != {r["id"] for rows in groups.values() for r in rows}:
        raise ValueError("The natural view comparison requires the complete registered development sets.")
    parallel_file = read_json(PARALLEL_RESULTS)
    if parallel_file["predictions_sha256"] != digest(PARALLEL_CACHE):
        raise ValueError("The synthetic control predictions changed after measurement.")
    parallel = read_json(PARALLEL_CACHE)["frames"]
    original = {k: v["baseline"] for k, v in parallel.items()}
    scenes = read_json(PARALLEL)["rows"]
    supplemental = {}
    for key in ("estimated", "oracle"):
        options = {k: v[key] for k, v in parallel.items()}
        empty = {k: values or options[k] for k, values in original.items()}
        additive = {k: [*values, *(p for p in options[k] if not any(
            duplicates(p["bbox"], prior["bbox"]) for prior in values))]
                    for k, values in original.items()}
        supplemental[key] = dict(empty=stats(scenes, original, empty), additive=stats(scenes, original, additive))
    observed = {}
    for scope, rows in groups.items():
        relevant = {r["id"]: saved["frames"][r["id"]] for r in rows}
        baseline = {k: v["baseline"] for k, v in relevant.items()}
        combined = {k: v["baseline"] or v["supplemental"] for k, v in relevant.items()}
        observed[scope] = stats(rows, baseline, combined)
        observed[scope]["stages"] = dict(Counter("original" if f["baseline"] else (
            "no dominant orientation" if f["angle"] is None else
            "supplemented" if f["supplemental"] else "empty after supplemental view") for f in relevant.values()))
    passed = any(item["gate"]["recovered_matches"] > 0 for item in observed.values()) and all(
        item["gate"]["passed"] and not item["spoiled_counts"]
        and item["baseline"]["fp"] >= item["prospective"]["fp"] for item in observed.values())
    write_json(OUTPUT, dict(development_gate=passed, natural=observed, synthetic=supplemental,
                            cache_sha256=digest(CACHE),
                            implementation="Use the original accepted boxes when present. Re-analyze a straightened "
                                           "view only for empty development scenes with a measured dominant axis. "
                                           "Independent reserves stay closed."))
    print("Natural deskew fallback development gate", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    (collect if parser.parse_args().action == "collect" else measure)()
