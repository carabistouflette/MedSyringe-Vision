"""Cache the same visual evidence for box ranking and pair discrimination."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou, suppress
from scripts.experiments.ranking.complete_syringe_selection import vector
from scripts.experiments.occlusion.occlusion_detector import ART, raster
from scripts.experiments.occlusion.occlusion_relocalization import prepare_reference
from scripts.experiments.relocalization.recrop_proposal_bank import read_bank
from scripts.experiments.transfer.rfdetr_verifier import DINOV2_INFO, DinoEncoder, expanded
from scripts.experiments.relocalization.relocalization_selector import pool
from scripts.experiments.expanded.sam3_study import configure_cpu

WORK = Path("data/overlap-ranking")
MINIMUM = .075
VIEWS = 6
BUDGET = 36


def candidate_pool(frame):
    valid = [p for p in pool(frame, VIEWS) if p["score"] >= MINIMUM]
    prioritized = suppress(valid, .88)[:BUDGET]
    return [dict(p, corroboration=max((box_iou(p["bbox"], q["bbox"]) for q in valid
                                     if p["view"] != q["view"]), default=0.))
            for p in prioritized]


def inputs(role):
    if role in ("fit", "development"):
        rows, frames = read_bank(role)
        base = {r["id"]: [] for r in rows}
        if role == "development":
            _, base, _, _ = prepare_reference()
        return rows, base, frames
    if role == "replay":
        from scripts.experiments.parts.part_association_reserve import evidence as crowded
        from scripts.experiments.relocalization.relocalization_reserve import evidence as neighboring
        from scripts.experiments.ranking.syringe_action_reserve import evidence as tray

        packets = [reader() for reader in (crowded, neighboring, tray)]
        chosen = {"association-aviano", "association-serological", "association-pipette",
                  "association-23655818", "refine-8670507", "action-17112252", "action-33658644"}
        return ([r for cohort, _, _, _ in packets for r in cohort if r["id"] in chosen],
                {k: v for _, baseline, _, _ in packets for k, v in baseline.items() if k in chosen},
                {k: v for _, _, bank, _ in packets for k, v in bank.items() if k in chosen})
    from scripts.experiments.ranking.overlap_ranking_reserve import evidence

    rows, base, frames, _ = evidence()
    return rows, base, frames


def binding(role, rows, base, frames):
    return dict(version=digest(__file__), backbone=digest(DINOV2_INFO), role=role,
                pixels={r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows},
                proposals={r["id"]: candidate_pool(frames[r["id"]]) for r in rows},
                incumbent={k: [[p["bbox"], p["score"]] for p in v] for k, v in base.items()})


def annotate_embedding(encoder, photo, objects):
    crops = [crop(photo, expanded(p["bbox"], pad)) for p in objects for pad in (.035, .24)]
    embedding = encoder.encode(crops).reshape(len(objects), 2, 384)
    return [dict(p, descriptor=vector(photo, p["bbox"]), features=np.round(e, 6).tolist())
            for p, e in zip(objects, embedding, strict=True)]


def collect(roles):
    WORK.mkdir(parents=True, exist_ok=True)
    configure_cpu()
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    encoder = DinoEncoder()
    for role in roles:
        rows, base, archive = inputs(role)
        path = WORK / f"features-{role}.json"
        signature = binding(role, rows, base, archive)
        state = read_json(path) if path.exists() else dict(binding=signature, frames={})
        if state["binding"] != signature:
            raise ValueError(f"Frozen feature inputs changed for {role}.")
        pending = [r for r in rows if r["id"] not in state["frames"]]
        for i, row in enumerate(pending, 1):
            k, image = row["id"], raster(row)
            seeds = signature["proposals"][k]
            incumbents = [dict(bbox=p["bbox"], score=p["score"], corroboration=0., view="released") for p in base[k]]
            encoded = annotate_embedding(encoder, image, [*incumbents, *seeds])
            state["frames"][k] = dict(base=encoded[:len(incumbents)], proposed=encoded[len(incumbents):],
                                      image_size=list(image.size))
            if i % 24 == 0 or i == len(pending):
                write_json(path, state)
                print(role, len(state["frames"]), len(rows), flush=True)


def load(role, rows, base, frames):
    state = read_json(WORK / f"features-{role}.json")
    if state["binding"] != binding(role, rows, base, frames) or set(state["frames"]) != {r["id"] for r in rows}:
        raise ValueError(f"Incomplete or changed visual evidence for {role}.")
    return state["frames"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("roles", nargs="+", choices=("fit", "development", "fresh", "replay"))
    collect(cli.parse_args().roles)
