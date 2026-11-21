"""Retry an unsupported colored-region estimate in a scale-confirmed orientation."""
from PIL import Image, ImageOps

from .fill import SCALES, calibration, liquid_box


def upright_liquid_for_fill(image, inscriptions, settings, measure):
    reading = inscriptions.get("capacity") or {}
    capacity = reading.get("value")
    if capacity not in SCALES or reading.get("unit") != "mL":
        return None
    reference = calibration(inscriptions.get("candidates", []), int(capacity), image.size, settings["ocr_min"])
    if reference is None or reference["axis"][1] >= -.75:
        return None
    upright = ImageOps.contain(image, (420, 420)).transpose(Image.Transpose.ROTATE_180)
    result = measure(upright)
    region = liquid_box(result)
    if not region:
        return None
    left, top, right, bottom = region
    return {**result, "region": [1-right, 1-bottom, 1-left, 1-top]}
