"""Estimate apparent graduated fill from a visible liquid column and piston."""
from collections import defaultdict
from itertools import combinations
from math import ceil, cos, floor, radians, sin

import cv2
import numpy as np
from PIL import ImageOps

from medsyringe.measurement.liquid_color import display_rgb, lab_pixels

SCALES = {20: (5, 1), 60: (10, 2)}


def unavailable(reason):
    return dict(status="insufficient evidence", reason=reason)


def center(box, size):
    left, top, right, bottom = box
    return np.array([(left+right)*size[0]/2., (top+bottom)*size[1]/2.])


def calibration(markings, maximum, size, minimum_score, check_ticks=False):
    """Fit the printed label centers to an affine mapping in image coordinates."""
    step, _ = SCALES[maximum]
    numbers = {str(n) for n in range(step, maximum+1, step)}
    by_view = defaultdict(dict)
    for token in markings:
        text = token["text"].strip()
        if text not in numbers or token["score"] < minimum_score:
            continue
        key = (token["rotation"], token["family"])
        value = int(text)
        if value not in by_view[key] or by_view[key][value]["score"] < token["score"]:
            by_view[key][value] = token

    matches = []
    for (angle, _), tokens in by_view.items():
        if len(tokens) < (2 if check_ticks else 3):
            continue
        amounts = np.array(sorted(tokens))
        points = np.array([center(tokens[n]["crop_bbox"], size) for n in amounts])
        theta = radians(angle)
        down = np.array([-sin(theta), cos(theta)])
        for first, last in combinations(range(len(amounts)), 2):
            travel = (points[last]-points[first])/(amounts[last]-amounts[first])
            length = np.linalg.norm(travel)
            if length*step < 8 or travel@down < .88*length:
                continue
            origin = points[first]-travel*amounts[first]
            distances = np.linalg.norm(points-origin-amounts[:, None]*travel, axis=1)/length
            selected = distances <= .27*step
            levels = amounts[selected]
            if len(levels) == 2 and (not check_ticks or levels[-1] != maximum or np.ptp(levels) < maximum*.49):
                continue
            if np.ptp(levels) < maximum*.38:
                continue
            travel, origin = np.linalg.lstsq(np.column_stack((levels, np.ones(len(levels)))),
                                           points[selected], rcond=None)[0]
            residuals = np.linalg.norm(points[selected]-origin-levels[:, None]*travel, axis=1)
            unit = np.linalg.norm(travel)
            if unit*step < 8 or travel@down < .88*unit:
                continue
            error = float(np.max(residuals)/unit)
            rank = (len(levels), int(np.ptp(levels)), -error, np.mean([tokens[n]["score"] for n in levels]))
            records = [dict(ml=int(n), crop_bbox=tokens[n]["crop_bbox"], text=tokens[n]["text"]) for n in levels]
            matches.append((rank, dict(axis=travel/unit, zero=origin, pixels_per_ml=unit,
                                        error=error, records=records, limits=(int(levels[0]), int(levels[-1])))))
    if not matches:
        return None
    return max(matches, key=lambda result: result[0])[1]


def pixels_at(lab, positions):
    rows = positions[..., 1].astype(np.float32)
    cols = positions[..., 0].astype(np.float32)
    return cv2.remap(lab, cols, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def periodic_ticks(lab, reference, maximum):
    """Check for a visible tick train next to two otherwise insufficient labels."""
    step, _ = SCALES[maximum]
    unit = reference["pixels_per_ml"]
    axis = reference["axis"]
    flank = np.array([axis[1], -axis[0]])
    start, end = reference["limits"]
    heights = np.arange(start+.25, end-.25, .05)
    anchors = reference["zero"] + heights[:, None]*unit*axis
    height, width = lab.shape[:2]
    for sign in (-1., 1.):
        offsets = sign*np.linspace(.25, .85, 13)*step*unit
        pixels = anchors[:, None, :] + offsets[None, :, None]*flank
        x, y = np.moveaxis(pixels, -1, 0)
        if np.mean((x >= 1) & (x < width-1) & (y >= 1) & (y < height-1)) < .9:
            continue
        profile = np.median(pixels_at(lab, pixels)[..., 0], axis=1)
        small = cv2.GaussianBlur(profile[:, None], (1, 0), 0, sigmaY=1.4).ravel()
        for spacing in (1., 2.):
            long = cv2.GaussianBlur(profile[:, None], (1, 0), 0, sigmaY=spacing*4.5).ravel()
            dips = long-small
            phase = (heights-start) % spacing
            for shift in np.arange(0., spacing, spacing/8):
                distance = np.abs((phase-shift+spacing/2.) % spacing-spacing/2.)
                at_ticks = dips[distance <= .16*spacing]
                off_ticks = dips[distance >= .32*spacing]
                hits = [np.max(dips[(heights >= start+q) & (heights < start+q+spacing) &
                                    (distance <= .18*spacing)], initial=0.)
                        for q in np.arange(.0, end-start-.5, spacing)]
                if (len(hits) >= 5 and np.mean(np.array(hits) >= 1.9) >= .70 and
                        np.median(at_ticks)-np.median(off_ticks) >= 1.3):
                    return True
    return False


def liquid_box(appearance):
    region = appearance.get("region") if appearance else None
    if appearance and appearance.get("lightness") and region:
        return region
    return None


def find_piston(lab, reference, region, maximum, min_drop, allow_gaps):
    height, width = lab.shape[:2]
    axis = reference["axis"]
    across = np.array([axis[1], -axis[0]])
    seed = center(region, (width, height))
    zero = reference["zero"]
    unit = reference["pixels_per_ml"]
    step, minor_step = SCALES[maximum]
    gap = unit*step
    # Transfer the longitudinal position of the printed zero to a strip through
    # the segmented liquid. The graduation text can sit off the barrel center.
    zero = zero + across*((seed-zero)@across)
    location = float((seed-zero)@axis/unit)
    if not .8 <= location <= .77*maximum:
        return None
    l, t, r, b = region
    corners = np.array([[x*width, y*height] for x in (l, r) for y in (t,b)])
    last = float(np.max((corners-zero)@axis)/unit)
    first_label, last_label = reference["limits"]
    low = max(location+.12*step, first_label-step, .35)
    high = min(last+.30*step, last_label+.45*step, maximum)
    if high <= low:
        return None
    levels = np.arange(-.25*step, maximum+.6*step, minor_step/12.)
    anchors = zero + levels[:, None]*unit*axis
    lateral = np.linspace(-.44, .44, 15)*gap
    points = anchors[:, None, :] + lateral[None, :, None]*across
    valid = ((points[..., 0] >= 1) & (points[..., 0] < width-1) &
             (points[..., 1] >= 1) & (points[..., 1] < height-1))
    profile = pixels_at(lab, points)
    chroma = np.linalg.norm(profile[..., 1:], axis=2)
    warm = (chroma >= 14.) & (profile[..., 2] >= 11.) & (profile[..., 1] >= -4.)
    brightness = profile[..., 0]
    # Compare an interior strip just before the front rim with a sustained dark
    # strip inside the rubber head. Three strips must see the same transition.
    length = max(3, round(.16*step/(minor_step/12.)))
    small = max(2, round(.045*step/(minor_step/12.)))
    candidates = []
    for j in np.flatnonzero((levels >= low) & (levels <= high)):
        preceding = np.s_[j-length:j-small+1]
        following = np.s_[j+small:j+length+1]
        core = np.s_[:, 3:12]
        if j < length or j+length >= len(levels):
            continue
        if np.mean(valid[j-length:j+length+1, 3:12]) < .90:
            continue
        before = np.median(brightness[preceding][core], axis=0)
        after = np.median(brightness[following][core], axis=0)
        drop = before-after
        contrast = float(np.median(drop))
        if contrast < min_drop or np.min([np.median(q) for q in np.split(drop, 3)]) < .58*min_drop:
            continue
        if np.median(after) > 42. or np.mean(warm[preceding][core]) < .56:
            continue
        forward = (levels >= .15*step) & (levels <= levels[j]-.12*step)
        continuity = np.mean(warm[forward, 3:12], axis=1)
        if not len(continuity) or np.mean(continuity >= .50) < allow_gaps:
            continue
        if np.mean(continuity[:max(2, len(continuity)//4)]) < .54:
            continue
        candidates.append((contrast, j))
    if not candidates:
        return None
    strength, best = max(candidates)
    alternatives = [j for value, j in candidates if value >= .86*strength]
    if np.ptp(levels[alternatives]) > .40*step:
        return None
    return dict(ml=float(levels[best]), ambiguity=float(np.ptp(levels[alternatives])/2),
                point=anchors[best], lateral=across, half_span=.40*gap, strength=strength)


def estimate(image, inscriptions, appearance, settings):
    """Return a guarded estimate, in the crop's coordinates, or a reason to abstain."""
    capacity = inscriptions.get("capacity") or {}
    maximum = capacity.get("value")
    if maximum not in SCALES or capacity.get("unit") != "mL":
        return unavailable("A visible 20 mL or 60 mL capacity is required.")
    maximum = int(maximum)
    region = liquid_box(appearance)
    if not region:
        return unavailable("A continuous visible liquid region could not be established.")
    thumbnail = ImageOps.contain(image, (600, 600))
    reference = calibration(inscriptions.get("candidates", []), maximum, thumbnail.size, settings["ocr_min"],
                            settings.get("tick_check", False))
    if reference is None:
        return unavailable("Three consistent numbered graduations are required.")
    rgb, _ = display_rgb(thumbnail)
    lab = lab_pixels(np.asarray(rgb))
    only_pair = len(reference["records"]) == 2
    if only_pair and not periodic_ticks(lab, reference, maximum):
        return unavailable("The two visible numbers have no corroborating graduation pattern.")
    piston = find_piston(lab, reference, region, maximum, settings["min_drop"], settings["continuity"])
    if piston is None:
        return unavailable("The nearest piston edge or the uninterrupted liquid column is ambiguous.")
    step, minor_step = SCALES[maximum]
    ml = piston["ml"]
    tolerance = max(.45*step if only_pair else minor_step,
                    reference["error"]+piston["ambiguity"]+.5*minor_step)
    bound = np.array(thumbnail.size)
    endpoints = np.array([piston["point"]-piston["lateral"]*piston["half_span"],
                          piston["point"]+piston["lateral"]*piston["half_span"]]) / bound
    endpoints = endpoints.clip(0., 1.)
    low = max(0., floor((ml-tolerance)/minor_step)*minor_step)
    high = min(maximum, ceil((ml+tolerance)/minor_step)*minor_step)
    value = round(ml/minor_step)*minor_step
    return dict(status="visual estimate", method="Numbered graduations and nearest piston edge",
                ml=value, interval_ml=[low, high],
                percent=round(100*value/maximum), interval_percent=[floor(100*low/maximum), ceil(100*high/maximum)],
                capacity_ml=maximum, piston_line=endpoints.tolist(), graduations=reference["records"],
                localization=dict(contrast=round(piston["strength"], 1), scale_residual_ml=round(reference["error"], 2),
                                  tick_train_verified=only_pair))
