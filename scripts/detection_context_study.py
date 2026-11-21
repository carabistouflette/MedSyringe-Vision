"""Measure the detection and proposal bottlenecks on the transparent-device development set."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter

from medsyringe.appearance_runtime import RELEASE as APPEARANCE
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.sam_residency_release import RELEASE as SAM
from scripts.detection_context_data import RELEASE as INPUTS
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

CACHE = DATA / "proposals/device-context-development-current.json"
OUTPUT = ART / "device_context_development_results.json"


def inputs():
    return [r for r in read_json(INPUTS)["rows"] if r["split"] == "device-context-development"]


def binding():
    return dict(manifest_sha256=digest(INPUTS), detector_sha256=digest(APPEARANCE),
                sam_sha256=digest(SAM), implementation_sha256=digest(__file__))


def collect():
    configure_cpu()
    known = binding()
    record = read_json(CACHE) if CACHE.exists() else dict(binding=known, frames={})
    if record["binding"] != known:
        raise ValueError("The contextual development inputs changed.")
    rows = inputs()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), "xpu")
        if engine.identity.get("primary_backend") != "xpu":
            raise ValueError("Run the fixed contextual study in the validated Intel environment.")
        for row in rows:
            identifier = row["id"]
            if identifier in record["frames"]:
                continue
            image = source(row)
            start = perf_counter()
            engine.detection_checks.begin()
            candidates, floor, sampler = engine.expanded.detect(image)
            health = dict(model=engine.identity, sam3_status=sampler)
            engine.detection_checks.attach(health)
            if health["detection_checks"]["status"] != "available":
                raise ValueError(f"One or more expected detector stages failed for {identifier}.")
            record["frames"][identifier] = dict(ranked=candidates, floor=floor,
                                               seconds=perf_counter() - start)
            write_json(CACHE, record)
            print("Transparent context", len(record["frames"]), "/", len(rows), flush=True)
    finally:
        engine.close()


def measure():
    if OUTPUT.exists():
        raise ValueError("Development results already exist.")
    rows = inputs()
    stored = read_json(CACHE)
    frames = stored["frames"]
    if stored["binding"] != binding() or set(frames) != {r["id"] for r in rows}:
        raise ValueError("The development frame cache is incomplete or incompatible.")
    candidates = {key: element["ranked"] for key, element in frames.items()}
    floors = {entity["floor"] for entity in frames.values()}
    if len(floors) != 1:
        raise ValueError("The operating point differed across the development collection.")
    actual = evaluate(rows, candidates, floors.pop())
    labels = {}
    for miss in actual["false_negatives"]:
        ranked = candidates[miss["id"]]
        parts = [(box_iou(miss["box"], item["bbox"]), index) for index, item in enumerate(ranked)]
        overlap, index = max(parts, default=(0, None))
        status = "localization below 0.50 IoU"
        if overlap >= .50:
            status = "proposed below operating point" if ranked[index]["score"] < actual["threshold"] else (
                "reference assignment conflict")
        miss.update(best_iou=overlap, proposal_index=index, likely_bottleneck=status)
    for row in rows:
        labels[row["id"]] = dict(reference=len(row["objects"]),
                                 accepted=sum(p["score"] >= actual["threshold"] for p in candidates[row["id"]]))
    durations = [frame["seconds"] for frame in list(frames.values())[1:]]
    write_json(OUTPUT, dict(
        metrics=short(actual), score_floor=actual["threshold"], counts=labels,
        missed_by_stage=dict(Counter(row["likely_bottleneck"] for row in actual["false_negatives"])),
        false_positives=actual["false_positives"], false_negatives=actual["false_negatives"],
        warm_seconds=dict(median=median(durations), p95=quantiles(durations, n=20, method="inclusive")[-1]),
        inputs=dict(stored["binding"]), cache_sha256=digest(CACHE),
        description="Photographs from one transparent-device source, development only. "
                    "The separate near-field bench project remains closed."))
    print("Transparent context", short(actual), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    {"collect": collect, "measure": measure}[parser.parse_args().action]()
