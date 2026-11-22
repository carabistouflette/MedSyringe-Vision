"""Optional local contrast views for inscriptions with evidence in the source crop."""
import cv2
import numpy as np
from PIL import Image


def contrast_view(image, method):
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if method == "clahe":
        enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
    elif method == "enhanced":
        smooth = cv2.GaussianBlur(gray, (0, 0), 1.1)
        sharpened = cv2.addWeighted(gray, 1.6, smooth, -.6, 0)
        low, high = np.percentile(sharpened, (2, 98))
        enhanced = np.clip((sharpened.astype(float)-low)*255/max(high-low, 10), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown marking view: {method}")
    return Image.fromarray(enhanced).convert("RGB")


def agreement(first, second):
    """Merge supported answers only when both variants agree on value and unit."""
    result = {**first, "candidates": [*first["candidates"], *second["candidates"]]}
    for field in ("capacity", "scale"):
        a, b = first.get(field), second.get(field)
        if a and b and (a["value"], a.get("unit")) == (b["value"], b.get("unit")):
            result[field] = {**a, "score": min(a["score"], b["score"]),
                             "method": "Agreement between two contrast views"}
        else:
            result[field] = None
    return result


def supplement(baseline, alternate):
    result = dict(baseline)
    changed = False
    for field in ("capacity", "scale"):
        proposal = alternate.get(field)
        if not proposal or result.get(field):
            continue
        other = result.get("scale" if field == "capacity" else "capacity")
        if other:
            capacity = proposal["value"] if field == "capacity" else other["value"]
            scale = other["value"] if field == "capacity" else proposal["value"]
            if scale in {"20-scale", "60-scale"} and capacity != float(scale.split("-")[0]):
                continue
        result[field] = proposal
        changed = True
    if changed:
        result["status"] = "visible marking"
        result["candidates"] = [*baseline["candidates"], *alternate["candidates"]]
    return result
