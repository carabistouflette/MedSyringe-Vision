"""Score the frozen part verifier on an additional source-disjoint positive photo."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.runtime.grounding_backend import analyze
from medsyringe.runtime.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision, protection
from scripts.experiments.parts.part_verifier_study import (
    FREEZE, REFERENCES, WORK, DinoEncoder, Segmenter, accept, brief, configure_cpu, crowd,
    describe, infer, load, offer,
)
from scripts.experiments.public_data.public_training_intake import raster_hashes

ASSET = WORK / "oriez.jpg"
REFERENCE = crowd.ART / "part_verifier_new_acquisition.json"
OUTPUT = crowd.ART / "part_verifier_new_acquisition_results.json"
PAGE = "https://www.dvidshub.net/image/6479379/wright-patt-phase-1-covid-vaccinations"


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    if read_image(ASSET).size != (1000, 698):
        raise ValueError("Unexpected source dimensions.")
    history, private, protected = protection()
    author = "R.J. Oriez"
    checksum = digest(ASSET)
    fits = read_json(REFERENCES)["roles"]["fit"]
    pixels = raster_hashes(ASSET)
    same_group = any("oriez" in lexical_key(a) for a in history["authors"])
    if (checksum in private | history["sha256"] | {row["sha256"] for row in fits}
            or same_group or url_key(PAGE).rstrip("/") in history["urls"]
            or fingerprint_collision(pixels, protected | history["dhash"])):
        raise ValueError("Known source exposure.")
    row = dict(id="oriez-vaccine-preparation", image=str(ASSET), sha256=checksum, source_url=PAGE, author=author,
               group="Wright-Patterson 2021-01-08 R.J. Oriez", license="Public domain (DVIDS)",
               objects=[dict(bbox=[l/1000, t/698, r/1000, b/698])
                        for l, t, r, b in ((594, 472, 627, 509), (889, 197, 916, 246))],
               review="Two partly occluded syringe bodies on one original photograph. The right-hand vial is separate.")
    write_json(REFERENCE, dict(row=row, implementation=digest(__file__), fitting_reference=digest(REFERENCES)))


def row():
    item = read_json(REFERENCE)
    if item["implementation"] != digest(__file__) or item["fitting_reference"] != digest(REFERENCES):
        raise ValueError("Prospective references changed.")
    if item["row"]["sha256"] != digest(ASSET):
        raise ValueError("Prospective raster changed.")
    return item["row"]


def infer_released():
    load()
    configure_cpu()
    root = Path.cwd()
    model = MedSyringeEngine(root)
    configure_device(model, root, "xpu")
    try:
        case = row()
        result = analyze(model, read_image(ASSET), False)
        checks = result["detection_checks"]
        if checks["status"] != "available" or checks["primary_backend"] != "xpu":
            raise ValueError("Released Intel inference is unavailable.")
        write_json(WORK / "oriez-baseline.json", dict(reference=digest(REFERENCE),
                                                      id=case["id"], objects=result["objects"]))
    finally:
        model.close()


def measure():
    heads = load()
    configure_cpu()
    frame = row()
    cached = read_json(WORK / "oriez-baseline.json")
    if cached["reference"] != digest(REFERENCE):
        raise ValueError("Baseline prediction refers to a different photo.")
    image = read_image(ASSET)
    model = Segmenter(756, "xpu")
    candidates = describe(image, infer(model, image), DinoEncoder())
    baseline = cached["objects"]
    conditions = {"unverified": lambda p: p["score"] >= .50,
                  **{arm: (lambda p, key=arm: accept(p, heads[key], key)) for arm in heads}}
    known = brief([frame], {frame["id"]: baseline})
    trials = {arm: brief([frame], {frame["id"]: offer(baseline, candidates, rule)})
              for arm, rule in conditions.items()}
    write_json(OUTPUT, dict(freeze=digest(FREEZE), reference=digest(REFERENCE), baseline=known, trials=trials,
                            retention={arm: compare(known, result) for arm, result in trials.items()},
                            scope="One additional pre-inference-annotated crowded public acquisition."))
    print({k: (v["tp"], v["fp"], v["exact_count"]) for k, v in {"baseline": known, **trials}.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("freeze", "released", "measure"))
    {"freeze": freeze, "released": infer_released, "measure": measure}[parser.parse_args().job]()
