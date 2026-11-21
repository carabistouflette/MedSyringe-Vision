"""Fuse and score detector proposals without assuming a single object per image."""
from math import log

import numpy as np

from .additive import duplicates
from .detectors import box_iou, suppress

CHANNELS = ("primary", "rf", "sam")
FLOORS = {"primary": .14, "rf": .045, "sam": .18}


def candidates(branches):
    sources = {name: [p for p in suppress(branches[name], .50) if p["score"] >= FLOORS[name]][:8]
               for name in CHANNELS}
    proposals = []
    for name, boxes in sources.items():
        for box in boxes:
            position = box["bbox"]
            support = {}
            for other, options in sources.items():
                closest = max(options, key=lambda p: box_iou(position, p["bbox"]), default=None)
                if closest:
                    iou = box_iou(position, closest["bbox"])
                    if iou >= .32:
                        support[other] = (closest, iou)
            proposals.append({**box, "source": name, "support_channels": sorted(support),
                              "support": support})
    return proposals


def vectors(proposals):
    result = []
    for p in proposals:
        x, y, r, b = p["bbox"]
        width, height = r-x, b-y
        source_scores = [p["support"].get(n, ({"score": 0.}, 0.))[0]["score"] for n in CHANNELS]
        agreements = [p["support"].get(n, (None, 0.))[1] for n in CHANNELS]
        result.append([p["score"], *source_scores, *agreements,
                       *(p["source"] == n for n in CHANNELS),
                       log(max(width*height, 1e-6)), abs(log(width/height)),
                       float(p.get("mask_area_fraction", 0.))])
    return np.asarray(result, dtype=np.float32).reshape(-1, 13)


def fused_box(proposal, mode):
    anchors = [(other["bbox"], other["score"]) for other, overlap in proposal["support"].values()
               if overlap >= .60]
    if mode == "weighted" and len(anchors) >= 2:
        boxes, weights = zip(*anchors, strict=True)
        return np.average(boxes, axis=0, weights=weights).tolist()
    if mode == "mask" and proposal.get("mask_bbox"):
        mask = proposal["mask_bbox"]
        if box_iou(proposal["bbox"], mask) >= .65:
            return mask
    return proposal["bbox"]


def rank(proposals, confidence, mode="anchor", overlap=.42):
    if len(proposals) != len(confidence):
        raise ValueError("Every proposal needs a confidence value.")
    records = [{**{k: v for k, v in p.items() if k != "support"},
                "bbox": fused_box(p, mode), "score": float(score), "branch_score": p["score"]}
               for p, score in zip(proposals, confidence, strict=True)]
    return suppress(records, overlap)


def rule(proposals, method):
    results = []
    for p in proposals:
        count = len(p["support_channels"])
        scores = [prediction["score"] for prediction, _ in p["support"].values()]
        if method == "union":
            score = p["score"]
        elif method == "agreement":
            score = min(.999, p["score"] * .68 + .14 * count + .07 * max(scores, default=0.))
        else:
            raise ValueError(f"Unknown fusion rule {method!r}.")
        results.append(score)
    return np.asarray(results)


def retain_baseline(prior, candidates, cutoff, baseline_cutoff):
    """Accept nonoverlapping rescues without displacing any accepted baseline box."""
    retained = [{**p, "score": max(p["score"], cutoff)} for p in prior if p["score"] >= baseline_cutoff]
    for candidate in candidates:
        if candidate["score"] < cutoff:
            continue
        if not any(duplicates(p["bbox"], candidate["bbox"]) for p in retained):
            retained.append(candidate)
    return retained
