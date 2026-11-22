"""Archive low-score RF-DETR boxes from fixed-resolution full and tiled views."""
import argparse
from pathlib import Path

from medsyringe.detection.additive import overlapping_tiles, project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from scripts.experiments.occlusion.occlusion_detector import EDGE, INITIAL, frozen_frames, partitions, raster
from scripts.experiments.occlusion.occlusion_relocalization_reserve import refs as previous_reserve
from scripts.experiments.transfer.rfdetr_compare import ranked_region

WORK = Path("data/native-candidate-verifier")
MINIMUM_SCORE = .025
NMS_IOU = .62
MAX_CANDIDATES = 28


def cohorts(role):
    if role == "evaluation":
        blocks, _ = frozen_frames()
        return [*(r for rows in blocks.values() for r in rows), *previous_reserve()]
    fit, development, _ = partitions()
    return fit if role == "fit" else development


def signature(rows):
    return dict(checkpoint=read_json(INITIAL)["checkpoint_sha256"], code=digest(__file__),
                rows={r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows},
                pool=[MINIMUM_SCORE, NMS_IOU, MAX_CANDIDATES], resolution=EDGE)


def views(model, image):
    def ranked(picture):
        return [p for p in ranked_region(model, picture, EDGE) if p["score"] >= MINIMUM_SCORE]

    merged = ranked(image)
    for window in overlapping_tiles(image.size, .66):
        for item in ranked(image.crop(window)):
            if bbox := project(item["bbox"], window, image.size):
                merged.append({**item, "bbox": bbox})
    return suppress(merged, NMS_IOU)[:MAX_CANDIDATES]


def collect(rows, role):
    import torch
    from rfdetr import RFDETRNano

    WORK.mkdir(exist_ok=True)
    cached = WORK / f"rf-{role}.json"
    key = signature(rows)
    output = read_json(cached) if cached.exists() else dict(signature=key, frames={})
    if output["signature"] != key:
        raise ValueError("Existing proposal bank belongs to different inputs.")
    pending = [r for r in rows if r["id"] not in output["frames"]]
    if not pending:
        return
    record = read_json(INITIAL)
    if record["status"] != "completed" or digest(record["checkpoint"]) != key["checkpoint"]:
        raise ValueError("RF-DETR weights changed.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = RFDETRNano.from_checkpoint(record["checkpoint"], device="cpu")
    for i, row in enumerate(pending, 1):
        output["frames"][row["id"]] = views(model, raster(row))
        if i % 5 == 0 or i == len(pending):
            write_json(cached, output)
            print(role, len(output["frames"]), "/", len(rows), flush=True)


def archive(rows, role):
    output = read_json(WORK / f"rf-{role}.json")
    if output["signature"] != signature(rows) or set(output["frames"]) != {r["id"] for r in rows}:
        raise ValueError(f"Missing or incompatible {role} proposals.")
    return output["frames"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("fit", "development", "evaluation"))
    role = parser.parse_args().role
    collect(cohorts(role), role)
