"""Bind a visual audit of real overlapping proposals to the public fitting bank."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from medsyringe.detection.additive import overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from scripts.experiments.crowds.native_candidate_verifier import reference_index
from scripts.experiments.occlusion.occlusion_detector import raster
from scripts.experiments.ranking.overlap_candidate_features import ART, WORK, candidate_pool, inputs
from scripts.experiments.relocalization.recrop_proposal_bank import WORK as RAW

QUEUE = WORK / "pair-review.json"
REVIEW = ART / "overlap_ranking_pair_review.json"
LABELS = {"same", "different", "ambiguous"}
REVIEWED = {
    1: "same", 2: "same", 3: "same", 4: "same", 5: "different", 6: "different",
    7: "ambiguous", 8: "same", 9: "same", 10: "different", 11: "ambiguous",
    12: "ambiguous", 13: "ambiguous", 14: "ambiguous", 15: "ambiguous", 16: "ambiguous",
    17: "ambiguous", 18: "different", 19: "ambiguous", 20: "same", 21: "same",
    22: "ambiguous", 23: "ambiguous", 24: "same", 25: "ambiguous", 26: "different",
    27: "ambiguous",
}


def tags(box, objects):
    target = reference_index(box, objects)
    if target is not None:
        return "complete", target, box_iou(box, objects[target]["bbox"])
    overlaps = sorted((box_iou(box, obj["bbox"]), i) for i, obj in enumerate(objects))
    if not overlaps or overlaps[-1][0] < .10:
        return "background", None, 0.
    best, i = overlaps[-1]
    competing = overlaps[-2][0] if len(overlaps) > 1 else 0.
    if best < .50 and best - competing >= .06 and overlap_fraction(box, objects[i]["bbox"]) >= .76:
        if area(box) < area(objects[i]["bbox"]):
            return "fragment", i, best
    return "ambiguous", None, best


def area(box):
    l, t, r, b = box
    return (r - l)*(b - t)


def nearby(a, b):
    return box_iou(a, b) >= .09 or overlap_fraction(a, b) >= .20


def pair_label(one, two):
    left, first, _ = one
    right, second, _ = two
    if first is None or second is None:
        return None
    if first == second:
        return "same"
    return "different" if left == right == "complete" else None


def samples(rows, frames):
    entries = []
    for row in rows:
        if len(row["objects"]) < 2:
            continue
        ps = candidate_pool(frames[row["id"]])
        assigned = [tags(p["bbox"], row["objects"]) for p in ps]
        options = {name: [] for name in ("same", "different", "ambiguous")}
        for i, j in combinations(range(len(ps)), 2):
            a, b = ps[i]["bbox"], ps[j]["bbox"]
            if nearby(a, b):
                label = pair_label(assigned[i], assigned[j]) or "ambiguous"
                if label == "ambiguous" and min(assigned[i][2], assigned[j][2]) < .20:
                    continue
                options[label].append((i, j, box_iou(a,b)))
        for label, found in options.items():
            found.sort(key=lambda item: abs(item[2] - .42))
            for i, j, overlap in found[:2 if label != "ambiguous" else 1]:
                entries.append(dict(id=len(entries)+1, image=row["id"], sha256=row["sha256"],
                                    first=ps[i]["bbox"], second=ps[j]["bbox"], i=i, j=j,
                                    candidate_iou=overlap, provisional=label,
                                    target_first=assigned[i][1], target_second=assigned[j][1]))
    return entries


def queue():
    rows, _, proposals = inputs("fit")
    entries = samples(rows, proposals)
    indexed = {r["id"]: r for r in rows}
    write_json(QUEUE, dict(code=digest(__file__), pairs=entries))
    colors = ("#ff476a", "#2df5e5")
    for start in range(0, len(entries), 8):
        sheet = Image.new("RGB", (1200, 1080), "#202636")
        draw = ImageDraw.Draw(sheet)
        for n, entry in enumerate(entries[start:start+8]):
            row = indexed[entry["image"]]
            picture = raster(row)
            boxes = [entry["first"], entry["second"]]
            canvas = ImageOps.contain(picture, (580, 236))
            d = ImageDraw.Draw(canvas)
            for idx, box in enumerate(boxes):
                shape = tuple(q*v for q, v in zip(box, canvas.size*2, strict=True))
                d.rectangle(shape, outline=colors[idx], width=3)
                d.text((shape[0]+3, shape[1]+3), "A" if idx==0 else "B", fill=colors[idx],
                       stroke_width=1, stroke_fill="black")
            left, top = 600*(n%2), 270*(n//2)
            sheet.paste(canvas, (left+(600-canvas.width)//2, top))
            draw.text((left+8, top+241), f"{entry['id']}   {entry['image']}   candidate IoU {entry['candidate_iou']:.2f}",
                      fill="white")
        sheet.save(WORK/f"fit-pairs-{start//8}.jpg", quality=94)
    print(len(entries), Counter(e["provisional"] for e in entries), flush=True)


def freeze():
    if REVIEW.exists():
        raise FileExistsError(REVIEW)
    rows, _, frames = inputs("fit")
    pairs = samples(rows, frames)
    if not REVIEWED or set(REVIEWED) != {p["id"] for p in pairs} or set(REVIEWED.values())-LABELS:
        raise ValueError("A visual label is required for each review pair.")
    if read_json(QUEUE)["pairs"] != pairs:
        raise ValueError("Review sheet geometry changed.")
    recorded = [dict(p, reviewed=REVIEWED[p["id"]]) for p in pairs]
    write_json(REVIEW, dict(code=digest(__file__), fitting_bank=digest(RAW/"fit.json"), pairs=recorded,
                            counts=Counter(REVIEWED.values()),
                            method="Single visual reviewer. Ambiguous pairs are excluded from association supervision."))
    print(Counter(REVIEWED.values()), flush=True)


def load():
    r = read_json(REVIEW)
    if r["code"] != digest(__file__) or r["fitting_bank"] != digest(RAW/"fit.json"):
        raise ValueError("Fit review inputs changed.")
    return r["pairs"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("queue","freeze"))
    {"queue":queue,"freeze":freeze}[cli.parse_args().stage]()
