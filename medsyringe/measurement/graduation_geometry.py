"""Experimental fill fallback using a rectified strip of printed graduations."""
from collections import defaultdict
from itertools import combinations
from math import ceil, cos, floor, radians, sin

import numpy as np
from PIL import ImageOps

from medsyringe.measurement.fill import SCALES, calibration, center, find_piston, liquid_box, periodic_ticks
from medsyringe.measurement.liquid_color import display_rgb, lab_pixels


def paired_references(tokens, maximum, size, score, span):
    step, _ = SCALES[maximum]
    valid_numbers = {str(n): n for n in range(step, maximum+1, step)}
    views = defaultdict(dict)
    for token in sorted(tokens, key=lambda t: -t["score"]):
        amount = valid_numbers.get(token["text"].strip())
        if amount is not None and token["score"] >= score:
            views[token["rotation"], token["family"]].setdefault(amount, token)
    for (angle, _), view in views.items():
        theta = radians(angle)
        down = np.array([-sin(theta), cos(theta)])
        for low, high in combinations(sorted(view), 2):
            if high-low < span*maximum:
                continue
            top, bottom = (center(view[n]["crop_bbox"], size) for n in (low, high))
            travel = (bottom-top)/(high-low)
            unit = np.linalg.norm(travel)
            if unit*step < 8 or travel@down < .88*unit:
                continue
            records = [dict(ml=n, crop_bbox=view[n]["crop_bbox"], text=view[n]["text"]) for n in (low, high)]
            yield dict(axis=travel/unit, zero=top-travel*low, pixels_per_ml=unit,
                       limits=(low, high), records=records, error=0.)


def estimate(image, inscriptions, appearance, settings, strategy):
    """Require a visible tick train and a bracketed, corroborated piston edge."""
    capacity = inscriptions.get("capacity") or {}
    scale = inscriptions.get("scale") or {}
    maximum = capacity.get("value") if capacity.get("unit") == "mL" else None
    capacity_basis = "Read nominal capacity"
    if maximum is None and strategy["ordered_scale"]:
        for size in SCALES:
            if scale.get("value") == f"{size}-scale" and scale.get("method") == "ordered terminal graduations":
                maximum = size
                capacity_basis = "Visible terminal graduation"
    region = liquid_box(appearance)
    if maximum not in SCALES or not region:
        return None
    maximum = int(maximum)
    thumbnail = ImageOps.contain(image, (600, 600))
    rgb, _ = display_rgb(thumbnail)
    lab = lab_pixels(np.asarray(rgb))
    tokens = inscriptions.get("candidates", [])
    reference = calibration(tokens, maximum, thumbnail.size, settings["ocr_min"])
    candidates = [reference] if reference else []
    if capacity_basis == "Read nominal capacity":
        candidates.extend(paired_references(tokens, maximum, thumbnail.size,
                                           max(.68, settings["ocr_min"]), strategy["min_span"]))
    step, minor = SCALES[maximum]
    valid = []
    for ref in candidates:
        if not periodic_ticks(lab, ref, maximum):
            continue
        piston = find_piston(lab, ref, region, maximum, settings["min_drop"], settings["continuity"])
        if piston and ref["limits"][0] <= piston["ml"] <= ref["limits"][1]:
            valid.append((ref, piston))
    if not valid or np.ptp([p["ml"] for _, p in valid]) > minor:
        return None
    ref, piston = max(valid, key=lambda pair: (len(pair[0]["records"]), np.ptp(pair[0]["limits"]),
                                              pair[1]["strength"]))
    value = round(piston["ml"]/minor)*minor
    bound = np.array(thumbnail.size)
    line = np.array([piston["point"]-piston["lateral"]*piston["half_span"],
                     piston["point"]+piston["lateral"]*piston["half_span"]]) / bound
    tolerance = max(.45*step, ref["error"] + piston["ambiguity"] + .5*minor)
    low = max(0, floor((piston["ml"]-tolerance)/minor)*minor)
    high = min(maximum, ceil((piston["ml"]+tolerance)/minor)*minor)
    return dict(status="visual estimate", method="Rectified graduation strip and nearest piston edge",
                ml=value, interval_ml=[low, high], capacity_ml=maximum, capacity_basis=capacity_basis,
                percent=round(100*value/maximum), interval_percent=[floor(100*low/maximum), ceil(100*high/maximum)],
                piston_line=line.clip(0., 1.).tolist(), graduations=ref["records"],
                localization=dict(tick_train_verified=True, contrast=round(piston["strength"], 1),
                                  compatible_fits=len(valid)))
