"""Extract apparent color from an existing localized liquid region."""
from colorsys import rgb_to_hsv

import cv2
import numpy as np
from PIL import ImageOps

from .attributes import crop
from .data import valid_box
from .liquid_color import describe_region, display_rgb, lab_pixels


def visible_hue(srgb, lstar):
    hue, saturation, _ = rgb_to_hsv(*(component / 255 for component in srgb))
    degrees = hue * 360
    if saturation < .12:
        return "neutral"
    if degrees < 15 or degrees >= 345:
        return "red"
    if degrees < 44:
        return "amber-brown" if lstar < 54 else "orange"
    if degrees < 68:
        return "yellow"
    if degrees < 160:
        return "green"
    if degrees < 195:
        return "cyan"
    if degrees < 260:
        return "blue"
    if degrees < 300:
        return "purple"
    return "magenta"


def contrasting_barrel_color(image, color):
    if max(image.size) < 2 * min(image.size):
        return False
    pixels = np.asarray(image)
    if image.width > image.height:
        pixels = pixels.transpose(1, 0, 2)
    h, w = pixels.shape[:2]
    strip = pixels[round(.12 * h):round(.88 * h), round(.20 * w):round(.80 * w)]
    hue, saturation, value = np.moveaxis(cv2.cvtColor(strip, cv2.COLOR_RGB2HSV), -1, 0)
    strong = (hue >= 5) & (hue <= 39) & (saturation >= 110) & (value >= 80)
    if strong.sum() < max(80, .08 * strong.size):
        return False
    low, middle, high = np.quantile(hue[strong], [.25, .50, .75])
    observed, _, _ = rgb_to_hsv(*(component / 255 for component in color["srgb"]))
    return high - low <= 8 and abs(middle - observed * 180) >= 8


def color_from_region(image, evidence):
    bounds = evidence.get("region")
    lightness = evidence.get("lightness")
    if not bounds or not valid_box(bounds) or not lightness or evidence.get("status") != "appearance measured":
        return None
    width, height = image.size
    box = tuple(round(v * n) for v, n in zip(bounds, (width, height) * 2, strict=True))
    patch = image.crop(box).convert("RGB")
    if min(patch.size) < 4:
        return None
    raw = np.asarray(patch)
    hue, saturation, value = np.moveaxis(cv2.cvtColor(raw, cv2.COLOR_RGB2HSV), -1, 0)
    chromatic = (hue >= 5) & (hue <= 39) & (saturation >= 45) & (value >= 25)
    lstar = lab_pixels(raw)[..., 0]
    if chromatic.sum() < max(50, .12 * chromatic.size):
        return None
    median = float(np.median(lstar[chromatic]))
    if abs(median - lightness["lstar"]) > 5.:
        return None
    low, high = lightness["iqr"]
    support = chromatic & (lstar >= low - 4.) & (lstar <= high + 4.)
    if support.sum() < max(40, .10 * chromatic.size):
        return None
    # Measure in the embedded display profile when available.
    screen, profile = display_rgb(patch)
    color = describe_region(np.asarray(screen), support, profile)["apparent_color"]
    if color["chroma"] < 13. or contrasting_barrel_color(image, color):
        return None
    return {**color, "family": visible_hue(color["srgb"], color["lab"][0])}


def enrich_liquid_fields(image, objects):
    for obj in objects:
        evidence = obj.get("liquid_appearance")
        if not evidence or evidence.get("apparent_color"):
            continue
        region = ImageOps.contain(crop(image, obj["bbox"]), (420, 420))
        measured = color_from_region(region, evidence)
        if measured is not None:
            evidence["apparent_color"] = measured
            evidence["color_basis"] = "Warm interior pixels within the localized region, checked against its lightness"
        elif evidence.get("lightness"):
            evidence["color_reason"] = "The localized color did not pass the pixel and barrel consistency checks"


def reviewed_liquid_color(image, body, sample):
    if not valid_box(sample) or not (body[0] <= sample[0] < sample[2] <= body[2]
                                    and body[1] <= sample[1] < sample[3] <= body[3]):
        raise ValueError("The liquid sample must fit inside the reviewed syringe box.")
    width, height = image.size
    pixels = image.crop(tuple(round(v * n) for v, n in zip(sample, (width, height) * 2, strict=True)))
    if min(pixels.size) < 4:
        raise ValueError("Choose a liquid region at least four source pixels wide and tall.")
    view, profile = display_rgb(pixels)
    rgb = np.asarray(view)
    appearance = describe_region(rgb, np.ones(rgb.shape[:2], dtype=bool), profile)
    appearance["apparent_color"]["family"] = visible_hue(appearance["apparent_color"]["srgb"],
                                                         appearance["apparent_color"]["lab"][0])
    l, t, r, b = body
    appearance.update(region=[(sample[0] - l) / (r - l), (sample[1] - t) / (b - t),
                              (sample[2] - l) / (r - l), (sample[3] - t) / (b - t)],
                      color_basis="Reviewer-selected liquid sample from the native image",
                      reason="Apparent color measured in the reviewer-selected liquid sample")
    return appearance
