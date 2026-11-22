"""Run the preselected crowd policy on independent reviewed photos and lookalikes."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.runtime.grounding_backend import analyze
from medsyringe.runtime.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.crowds.crowded_sam_hybrid import POLICIES, reconcile
from scripts.experiments.crowds.crowded_sam_resolution import OUTPUT as POLICY, CrowdedMasks, masks, select
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.expanded.sam3_study import LOCK, configure_cpu

REFERENCES = ART / "crowded_sam_prospective_reference.json"
OUTPUT = ART / "crowded_sam_prospective.json"
PREDICTIONS = DATA / "proposals/crowded-sam/prospective"


def inputs():
    data = read_json(REFERENCES)
    if digest(POLICY) != data["policy_sha256"]:
        raise ValueError("Prospective selection was frozen against a different policy.")
    return data["rows"]


def collect(branch):
    configure_cpu()
    cases = inputs()
    output = PREDICTIONS / f"{branch}.json"
    key = dict(code=digest(__file__), checkpoint=digest(LOCK), references=digest(REFERENCES))
    record = read_json(output) if output.exists() else dict(binding=key, frames={})
    if record["binding"] != key:
        raise ValueError("Prospective references or policy code changed.")
    engine = None
    if branch == "released":
        root = Path.cwd()
        engine = MedSyringeEngine(root)
        configure_device(engine, root, "xpu")
    else:
        edge, prompt, geometry, suppression, floor = read_json(POLICY)["selected"]["configuration"]
        sam = CrowdedMasks(edge, "xpu")
    try:
        for row in cases:
            if row["id"] in record["frames"]:
                continue
            photo = Path("data") / row["image"]
            if row["sha256"] != digest(photo):
                raise ValueError("A prospective photograph changed.")
            image = read_image(photo)
            if engine:
                result = analyze(engine, image, False)
                health = result["detection_checks"]
                if health["primary_backend"] != "xpu" or health["status"] != "available":
                    raise ValueError("Run the released detector with the complete Intel inference path.")
                detected = result["objects"]
            else:
                stream = next(candidates for name, candidates in sam.candidates(image) if name == prompt)
                detected = select(masks(stream), geometry, suppression, floor)
            record["frames"][row["id"]] = detected
            write_json(output, record)
            print(branch, row["id"], len(detected), flush=True)
    finally:
        if engine:
            engine.close()


def measure():
    rows = inputs()
    ids = {row["id"] for row in rows}
    raw = {name: read_json(PREDICTIONS / f"{name}.json") for name in ("released", "sam")}
    if any(p["binding"]["references"] != digest(REFERENCES) or set(p["frames"]) != ids for p in raw.values()):
        raise ValueError("Collect each preselected branch on all six frozen photos.")
    released, sam = (raw[name]["frames"] for name in ("released", "sam"))
    before = evaluate(rows, released, 0.)
    candidates = {"released": released, "sam standalone": sam, **{
        policy: {key: reconcile(prior, sam[key], policy) for key, prior in released.items()} for policy in POLICIES}}
    result = {}
    for arm, predictions in candidates.items():
        measured = evaluate(rows, predictions, 0.)
        result[arm] = dict(metrics=short(measured), retention=compare(before, measured),
                           counts={row["id"]: [len(row["objects"]), len(predictions[row["id"]])] for row in rows})
    write_json(OUTPUT, dict(outcomes=result, references_sha256=digest(REFERENCES),
                            scope="Three crowded and three hard-negative photos from distinct groups. "
                                  "The experiment is descriptive and no thresholds were tuned."))
    print({arm: outcome["metrics"] for arm, outcome in result.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("released", "sam", "measure"))
    settings = parser.parse_args()
    measure() if settings.job == "measure" else collect(settings.job)
