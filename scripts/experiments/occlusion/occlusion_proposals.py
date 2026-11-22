"""Measure the ceiling at each SAM proposal and acceptance stage."""
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from medsyringe.detection.additive import overlapping_tiles, project
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detection.detectors import box_iou
from scripts.experiments.parts.part_verifier_new_acquisition import row
from scripts.experiments.parts.part_verifier_study import (
    BODY_PROMPTS, WORK, DinoEncoder, Segmenter, accept, configure_cpu, crowd, describe, infer, load, offer,
)

WORK = Path("data/occlusion-study")
OUTPUT = crowd.ART / "occlusion_proposal_audit.json"


def oracle(objects, predictions, minimum=.50):
    """Maximize distinct eligible matches before maximizing total matched overlap."""
    matrix = np.array([[box_iou(obj["bbox"], p["bbox"]) for p in predictions] for obj in objects])
    if not objects or not predictions:
        return dict(candidates=len(predictions), matched=0, targets=[
            dict(best_iou=0., matched_iou=None, best_candidate=None) for _ in objects])
    eligible = matrix >= minimum
    rows, cols = linear_sum_assignment(eligible * (len(objects) + 1 + matrix), maximize=True)
    matched = {int(i): float(matrix[i, j]) for i, j in zip(rows, cols, strict=True) if eligible[i, j]}
    return dict(candidates=len(predictions), matched=len(matched), targets=[
        dict(best_iou=float(line.max()), matched_iou=matched.get(i), best_candidate=predictions[int(line.argmax())])
        for i, line in enumerate(matrix)])


def body(observations, region):
    return [dict(bbox=p["bbox"], score=p["score"], region=region, prompt=key)
            for key in BODY_PROMPTS for p in observations[key]]


def run():
    configure_cpu()
    locked = row()
    image = read_image(locked["image"])
    baseline = read_json(Path("data/part-verifier/oriez-baseline.json"))["objects"]
    model = Segmenter(756, "xpu")
    observations = infer(model, image)
    raw = body(observations, "full")
    described = describe(image, observations, DinoEncoder())
    slim = [dict(bbox=p["bbox"], score=p["score"]) for p in described]
    stages = dict(released=baseline, full_raw=raw, full_at_point_40=[p for p in raw if p["score"] >= .40],
                  full_after_mask_suppression=slim)
    for name, head in load().items():
        selected = offer(baseline, described, lambda p: accept(p, head, name))
        stages[f"full_{name}_fusion"] = [dict(bbox=p["bbox"], score=p["score"]) for p in selected]
    for fraction in (.66, .50):
        projected, interior = [], []
        for window in overlapping_tiles(image.size, fraction):
            observations = infer(model, image.crop(window))
            for p in body(observations, list(window)):
                safe = project(p["bbox"], window, image.size)
                projected.append({**p, "bbox": project(p["bbox"], window, image.size, edge=0),
                                  "passes_internal_edge_rule": bool(safe)})
                if safe:
                    interior.append({**p, "bbox": safe})
        stages[f"tile_{fraction}_projected"] = projected
        stages[f"tile_{fraction}_interior"] = interior
        print(fraction, {name: oracle(locked["objects"], p)["matched"] for name, p in stages.items()
                         if name.startswith("tile_")}, flush=True)
    diagnostic = {}
    for i, obj in enumerate(locked["objects"]):
        x, y, X, Y = obj["bbox"]
        xc, yc = (x + X) * image.width / 2, (y + Y) * image.height / 2
        rx, ry = (X - x) * image.width * 2, (Y - y) * image.height * 2
        window = (max(0, round(xc-rx)), max(0, round(yc-ry)),
                  min(image.width, round(xc+rx)), min(image.height, round(yc+ry)))
        predictions = [{**p, "bbox": project(p["bbox"], window, image.size, edge=0)}
                       for p in body(infer(model, image.crop(window)), list(window))]
        diagnostic[str(i)] = oracle([obj], predictions)
    report = {name: {str(t): oracle(locked["objects"], p, t) for t in (.30, .50, .65)}
              for name, p in stages.items()}
    write_json(WORK / "sam-proposals.json", stages)
    write_json(OUTPUT, dict(reference=locked["sha256"], implementation=digest(__file__), stages=report,
                            privileged_crop_only=diagnostic))
    print({name: values["0.5"]["matched"] for name, values in report.items()}, flush=True)


if __name__ == "__main__":
    run()
