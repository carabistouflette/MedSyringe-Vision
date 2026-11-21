"""Inventory the expanded capture and freeze its acquisition-aware partitions."""
import argparse
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from medsyringe.data import dhash, digest, read_json, write_json

ART = Path("artifacts/expanded")
BLOCKS = (
    (3161, 3177, "clear-60-wall", "fit", "uncolored", "60-scale"),
    (3178, 3191, "clear-20-wall", "test", "uncolored", "20-scale"),
    (3192, 3196, "amber-60-initial", "fit", "amber", "60-scale"),
    (3197, 3202, "amber-60-surfaces", "fit", "amber", "60-scale"),
    (3203, 3213, "amber-60-wall", "calibration", "amber", "60-scale"),
    (3214, 3222, "amber-20-wall", "fit", "amber", "20-scale"),
    (3223, 3229, "amber-20-low-fill-dark-wall", "fit", "amber", "20-scale"),
    (3230, 3247, "amber-20-high-fill-dark-wall", "calibration", "amber", "20-scale"),
    (3248, 3287, "orange-60-dark-wall", "fit", "orange", "60-scale"),
    (3288, 3326, "orange-20-dark-wall", "calibration", "orange", "20-scale"),
    (3327, 3337, "warm-20-return-wall", "fit", "warm", "20-scale"),
)


def inspect(path):
    with Image.open(path) as original:
        exif = original.getexif()
        camera = exif.get(272)
        details = exif.get_ifd(34665)
        captured = datetime.strptime(details[36867], "%Y:%m:%d %H:%M:%S")
        resolution = list(original.size)
        orientation = exif.get(274, 1)
        icc = original.info.get("icc_profile", b"")
        original.draft("RGB", (340, 340))
        image = ImageOps.exif_transpose(original).convert("RGB")
        thumb = np.asarray(image.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
    return dict(id=path.stem, file=path.name, sha256=digest(path), bytes=path.stat().st_size,
                capture=captured.isoformat(), subsecond=str(details.get(37521, "")), camera=camera,
                stored_size=resolution, orientation=orientation, has_icc=bool(icc),
                dhash=dhash(image)), thumb


def audit(source):
    old = {row["id"]: row for row in read_json("artifacts/source_inventory.json") if "id" in row}
    rows, thumbs = [], {}
    files = sorted(p for p in source.iterdir() if p.suffix.casefold() in {".jpg", ".jpeg"})
    frames = {index: block for block in BLOCKS for index in range(block[0], block[1]+1)}
    if {p.stem for p in files} != {f"IMG_{i}" for i in frames} or len(files) != len(frames):
        raise ValueError("The source does not match the documented capture sequence.")
    for path in files:
        row, thumb = inspect(path)
        _, _, group, split, condition, scale = frames[int(path.stem[4:])]
        row.update(group=group, split=split, observed_condition=condition, group_scale=scale)
        if path.stem in old:
            prior = old[path.stem]
            distance = (int(prior["dhash"], 16) ^ int(row["dhash"], 16)).bit_count()
            if distance > 6:
                raise ValueError(f"Inspect the presumed original export pair for {path.stem}.")
            row.update(previous_sha256=prior["sha256"], previous_dhash_distance=distance)
        rows.append(row)
        thumbs[row["id"]] = thumb
    pairs = []
    for a, b in combinations(rows, 2):
        distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
        if distance > 6 and a["sha256"] != b["sha256"]:
            continue
        difference = float(cv2.absdiff(thumbs[a["id"]], thumbs[b["id"]]).mean())
        if difference > 15 and a["sha256"] != b["sha256"]:
            continue
        pairs.append(dict(a=a["id"], b=b["id"], dhash_distance=distance,
                          rgb_mean_difference=round(difference, 3),
                          strong=a["sha256"] == b["sha256"] or (distance <= 4 and difference <= 8),
                          cross_split=a["split"] != b["split"]))
    split_counts = dict(Counter(r["split"] for r in rows))
    heldout_overlap = [r["id"] for r in rows if r["split"] == "test" and "previous_sha256" in r]
    violations = [p for p in pairs if p["strong"] and p["cross_split"]]
    passed = not heldout_overlap and not violations
    write_json(ART / "inventory.json", dict(rows=rows, counts=split_counts, grouping=BLOCKS,
               near_pairs=pairs, leakage_check=dict(passed=passed, previous_in_test=heldout_overlap,
                                                   strong_cross_partition=violations)))
    print(len(rows), "images; partitions", split_counts, "strong cross-partition pairs", violations)
    if not passed:
        raise ValueError("The proposed partitions have an unresolved overlap.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    audit(parser.parse_args().source)


if __name__ == "__main__":
    main()
