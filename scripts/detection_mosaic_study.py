"""Measure size and count stress of deployed and corroborated mosaic detections."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.recall_guard import compare
from scripts.appearance_text_encoder import TextMarginEncoder
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_mask_crowded_audit import OPTION, text_scores, wide
from scripts.detection_mask_retrieval import MaskRetriever
from scripts.detection_mask_verification import candidates, detect, support
from scripts.detection_mosaic_development import MANIFEST
from scripts.detection_open_models import OPEN_MODELS, Detector, model_lock
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.sam3_study import LOCK, configure_cpu

OUTPUT = ART / "crowded_mosaic_stress_results.json"


def path(stage):
    return DATA / "proposals" / f"crowded-mosaic-{stage}.json"


def binding(stage):
    prefix = dict(evidence=digest(MANIFEST), implementation=digest(__file__), release=digest(RELEASE))
    if stage == "supplemental":
        prefix.update(boxes=digest(path("baseline")), sam=digest(LOCK),
                      witnesses={name: model_lock(name) for name in OPEN_MODELS})
    return prefix


def stored(stage):
    file = path(stage)
    key = binding(stage)
    data = read_json(file) if file.exists() else dict(key=key, frames={})
    if data["key"] != key:
        raise ValueError("The mosaic stress run changed since its first completed frame.")
    return data


def run_baseline(image, engine):
    engine.detection_checks.begin()
    ranked, threshold, status = engine.expanded.detect(image)
    checks = dict(model=engine.identity, sam3_status=status)
    engine.detection_checks.attach(checks)
    if checks["detection_checks"]["status"] != "available":
        raise ValueError(f"Degraded detector during stress evaluation: {checks['detection_checks']}")
    return [p for p in ranked if p["score"] >= threshold]


def collect_baseline():
    configure_cpu()
    rows = read_json(MANIFEST)["rows"]
    collected = stored("baseline")
    root = Path.cwd()
    engine = MedSyringeEngine(root)
    try:
        configure_device(engine, root, "xpu")
        if engine.identity.get("primary_backend") != "xpu":
            raise ValueError("The validated Intel backend did not start.")
        for row in rows:
            key = row["id"]
            if key in collected["frames"]:
                continue
            picture = source(row)
            start = perf_counter()
            predictions = run_baseline(picture, engine)
            collected["frames"][key] = dict(boxes=predictions, seconds=perf_counter()-start)
            write_json(path("baseline"), collected)
            print("Mosaic baseline", len(collected["frames"]), "/", len(rows), flush=True)
    finally:
        engine.close()


def collect_supplemental():
    configure_cpu()
    rows = read_json(MANIFEST)["rows"]
    original = stored("baseline")["frames"]
    if set(original) != {r["id"] for r in rows}:
        raise ValueError("Run the base detector on every mosaic first.")
    collected = stored("supplemental")
    model = others = None
    for row in rows:
        key = row["id"]
        if key in collected["frames"]:
            continue
        accepted = original[key]["boxes"]
        ranked = []
        if len(accepted) >= 2:
            if model is None:
                model = MaskRetriever()
                others = [Detector(k, "xpu") for k in OPEN_MODELS]
            picture = source(row)
            selection = candidates(dict(grids=model.predict_grid(picture)), accepted)
            witnesses = [wide(m, picture) for m in others]
            selection = [{**v, **support(v["bbox"], witnesses, [])} for v in selection]
            ranked = [v for v in selection if v["open_votes"] and v["stability"] >= OPTION[-1]]
        collected["frames"][key] = ranked
        write_json(path("supplemental"), collected)
        print("Mosaic supplemental", len(collected["frames"]), "/", len(rows), flush=True)


def measure():
    if OUTPUT.exists():
        raise ValueError("The result was already measured.")
    rows = read_json(MANIFEST)["rows"]
    original, new = (stored(stage)["frames"] for stage in ("baseline", "supplemental"))
    if set(original) != set(new) or set(original) != {r["id"] for r in rows}:
        raise ValueError("The synthetic development inputs were not all evaluated.")
    configure_cpu()
    previous = {key: v["boxes"] for key, v in original.items()}
    proposed = {}
    encoder = None
    for row in rows:
        key = row["id"]
        available = new[key]
        if available:
            if encoder is None:
                encoder = TextMarginEncoder()
            available = text_scores(available, source(row), encoder)
        proposed[key] = detect(previous[key], available, OPTION)
    strata = {"all": rows, "multi": [r for r in rows if len(r["objects"]) >= 2]}
    evidence = {}
    for name, refs in strata.items():
        a, b = [evaluate(refs, ranks, 0.) for ranks in (previous, proposed)]
        evidence[name] = dict(previous=short(a), prospective=short(b), guard=compare(a, b))
    write_json(OUTPUT, dict(scopes=evidence, settings=OPTION, manifest_sha256=digest(MANIFEST),
                            median_baseline_seconds=median(v["seconds"] for v in original.values()),
                            sources="Synthetic development stress from repeated real development parents. "
                                    "Treat scene statistics as descriptive and separate from independent photographs."))
    print("Mosaic stress", evidence["all"]["previous"], evidence["all"]["prospective"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "supplemental", "measure"))
    {"baseline": collect_baseline, "supplemental": collect_supplemental,
     "measure": measure}[parser.parse_args().action]()
