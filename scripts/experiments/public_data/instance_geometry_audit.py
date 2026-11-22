"""Compare approved boxes against a second reading of a fixed raw-photo sample."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median

import numpy as np
from scipy.optimize import linear_sum_assignment

from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detection.detectors import box_iou
from scripts.experiments.public_data.instance_review_export import AUDIT, OUT

INDEX = OUT / "blind_geometry_index.json"
REVIEW = OUT / "blind_geometry_review.json"
REPORT = AUDIT.with_name("public_instance_blind_geometry_audit.json")


def order(row):
    return sha256(f"instance-audit-2026-v1/{row['id']}".encode()).hexdigest()


def sample(rows):
    groups = defaultdict(list)
    for row in rows:
        if len(row["objects"]) == 1:
            groups[row["group"]].append(row)
    singles = [min(members, key=order) for members in groups.values()]
    crowds = sorted((r for r in rows if len(r["objects"]) > 1), key=order)[:5]
    return dict(photos={r["id"]: r["image"] for r in sorted([*singles, *crowds], key=order)})


def score(rows, proof, index):
    if set(proof["photos"]) != set(index["photos"]):
        raise ValueError("The second reading does not cover the frozen sample.")
    inventory = {row["id"]: row for row in rows}
    paired_ious = []
    frames = []
    for key, reference in proof["photos"].items():
        row = inventory[key]
        approved = [obj["bbox"] for obj in row["objects"]]
        blind = reference["boxes"]
        if row["image"] != index["photos"][key] or not all(valid_box(box) for box in blind):
            raise ValueError(f"Malformed or changed raw-photo review for {key}.")
        overlap = np.array([[box_iou(a, b) for b in blind] for a in approved]).reshape(len(approved), len(blind))
        old, new = linear_sum_assignment(overlap, maximize=True)
        pairs = [dict(approved=int(a), blind=int(b), iou=float(overlap[a, b])) for a, b in zip(old, new, strict=True)]
        paired_ious.extend(pair["iou"] for pair in pairs)
        frames.append(dict(id=key, count_before=len(approved), count_blind=len(blind),
                           paired_boxes=pairs, notes=reference["note"]))
    summary = dict(photos=len(frames), same_count=sum(r["count_before"] == r["count_blind"] for r in frames),
                   pairs=len(paired_ious), minimum_iou=min(paired_ious, default=None),
                   median_iou=median(paired_ious) if paired_ious else None,
                   pairs_at_iou50=sum(iou >= .50 for iou in paired_ious),
                   pairs_at_iou75=sum(iou >= .75 for iou in paired_ious))
    return summary, frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("sample", "score"))
    rows = read_json(AUDIT)["rows"]
    if parser.parse_args().job == "sample":
        if INDEX.exists():
            raise FileExistsError(INDEX)
        write_json(INDEX, sample(rows))
    else:
        index = read_json(INDEX)
        if index != sample(rows):
            raise ValueError("The approved photos changed after the raw review.")
        summary, frames = score(rows, read_json(REVIEW), index)
        write_json(REPORT, dict(summary=summary, review_sha256=digest(REVIEW), sample_index_sha256=digest(INDEX),
                                frames=frames, scope="Group-stratified subset of the first approved catalog."))
