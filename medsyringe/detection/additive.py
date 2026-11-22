"""Recall-preserving spatial fusion for supplemental syringe proposals."""
from medsyringe.detection.detectors import box_iou


def overlap_fraction(first, second):
    x, y, r, b = first
    X, Y, R, B = second
    area = max(0, min(r, R)-max(x, X))*max(0, min(b, B)-max(y, Y))
    smaller = min((r-x)*(b-y), (R-X)*(B-Y))
    return area / smaller if smaller else 0.0


def duplicates(first, second):
    return box_iou(first, second) >= .35 or overlap_fraction(first, second) >= .75


def overlapping_tiles(size, fraction=.66):
    w, h = size
    tw, th = max(1, round(w*fraction)), max(1, round(h*fraction))
    return sorted({(x, y, x+tw, y+th) for x in (0, w-tw) for y in (0, h-th)})


def project(box, window, full_size, edge=.025):
    l, t, r, b = box
    x, y, X, Y = window
    w, h = full_size
    if ((x > 0 and l < edge) or (y > 0 and t < edge) or
        (X < w and r > 1-edge) or (Y < h and b > 1-edge)):
        return None
    return [(x+l*(X-x))/w, (y+t*(Y-y))/h, (x+r*(X-x))/w, (y+b*(Y-y))/h]


def additions(ranked, auxiliary, cutoff, *, support, aux_cutoff, iou):
    """Keep base proposals intact; append accepted boxes at the fixed score floor."""
    base = [p for p in ranked if p["score"] >= cutoff]
    result = list(base)
    missed = [p for p in ranked if support is not None and support <= p["score"] < cutoff]
    for candidate in sorted(auxiliary, key=lambda p: p["score"], reverse=True):
        if candidate["score"] < aux_cutoff:
            continue
        box = candidate["bbox"]
        if any(duplicates(box, p["bbox"]) for p in result):
            continue
        if support is not None and not any(box_iou(box, p["bbox"]) >= iou for p in missed):
            continue
        result.append({**candidate, "auxiliary_score": candidate["score"], "score": cutoff,
                       "source": "auxiliary"})
    return result + [p for p in ranked if p["score"] < cutoff]
