"""Inspect visible stopper and barrel evidence alongside a legible small scale."""
import cv2
import numpy as np

from .liquid_color import describe_region
from .liquid_fields import visible_hue


def spans(row_support):
    edges = np.flatnonzero(np.diff(np.r_[False, row_support, False]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def unknown(reason, seal=None):
    return dict(presence="unknown", fill_ml=None, interval_ml=None, appearance=None,
                relative_opacity=None, reason=reason, seal=seal)


def inspect_barrel(view, track, pitch, sign, capacity):
    """Inspect only barrel pixels away from the printed graduations."""
    end = track[-1][1]
    zero = end - capacity * pitch
    track_width = float(np.median([right - left for left, _, right in track]))
    edge = float(np.median([left if sign == 1 else right for left, _, right in track]))
    left, right = sorted([round(edge - sign * .86 * track_width), round(edge - sign * .20 * track_width)])
    top = round(zero - .24 * pitch)
    bottom = round(end + .60 * pitch)
    h, w = view.shape[:2]
    if not (0 <= top < bottom <= h and 0 <= left < right <= w and right - left >= 9):
        return unknown("Barrel interior is clipped or too narrow.")
    rgb = cv2.cvtColor(view[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    hue, saturation, value = np.moveaxis(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), -1, 0)
    colored = (saturation >= 108) & (value >= 42) & (value <= 200)
    dark = (saturation < 108) & (value <= 78)
    strong = colored | dark
    flat = np.mean(strong, axis=1) >= .78
    candidates = []
    margin = max(2, round(pitch * .09))
    for start, stop in spans(flat):
        level = (top + start - zero) / pitch
        if not (-.19 <= level <= capacity + .12 and .12 * pitch <= stop - start <= .66 * pitch):
            continue
        frontal = slice(max(0, start - 2 * margin), start)
        interior = slice(start + 1, min(stop, start + margin))
        if start < 2 * margin or np.mean(strong[frontal]) > .33 or np.mean(strong[interior]) < .75:
            continue
        candidates.append((start, stop, level))
    if not candidates:
        return unknown("No separate, coherent stopper front was visible beside the scale.")
    first, last, level = candidates[0]
    if len(candidates) > 1 and candidates[1][0] - last > .45 * pitch:
        return unknown("Several separated bands could be the stopper.")
    seal = dict(level_ml=round(float(level), 3), oriented_box=[left, top + int(first), right, top + int(last)])
    if level <= .13:
        return dict(presence="empty", fill_ml=0., interval_ml=[0., .2], appearance=None,
                    relative_opacity=None, reason="A distinct stopper front is seated at the scale origin.", seal=seal)
    first_fluid, last_fluid = max(0, round(zero - top + .14 * pitch)), first - margin
    if last_fluid - first_fluid < .55 * pitch:
        return unknown("The gap before the stopper is too short to distinguish a liquid column.", seal)
    rows = slice(first_fluid, last_fluid)
    sample = rgb[rows]
    chromatic = (saturation[rows] >= 32) & (saturation[rows] <= 122) & (value[rows] >= 100)
    if (np.mean(chromatic) < .65 or np.mean(np.mean(chromatic, axis=1) >= .55) < .80
            or np.mean(strong[rows]) >= .20):
        return unknown("The stopper is visible, but the chamber lacks a distinct colored liquid column.", seal)
    observed_hue = hue[rows][chromatic]
    radians = observed_hue * 2 * np.pi / 180
    common_hue = abs(np.mean(np.exp(1j * radians)))
    if common_hue < .94:
        return unknown("Color varies across the chamber and cannot establish a single liquid column.", seal)
    # Exclude pigmentation that also spans the plunger-side chamber.
    rear = saturation[last + margin:last + margin + len(sample)]
    if rear.size < .6 * sample.shape[1] * pitch or np.mean(rear >= 32) > .42:
        return unknown("The color also appears behind the stopper, or the rear chamber is obscured.", seal)
    appearance = describe_region(sample, chromatic, "source RGB assumed sRGB")
    color = appearance["apparent_color"]
    color["family"] = visible_hue(color["srgb"], color["lab"][0])
    error = .25
    return dict(presence="present", fill_ml=None,
                interval_ml=[max(0., round(float(level - error), 2)), min(float(capacity), round(float(level + error), 2))],
                appearance=appearance, relative_opacity=None, seal=seal,
                reason="A coherent colored column lies between the scale origin and the stopper front.")
