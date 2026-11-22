"""Conservative crop suggestions and optional local OCR with visible evidence."""
import argparse
import csv
import io
import math
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageOps

from medsyringe.data import read_image, read_json, train_rows, write_json

FIELDS = {"scale": ("20-scale", "60-scale"), "color": ("amber-brown", "orange")}


def crop(image, box):
    w, h = image.size
    l, t, r, b = box
    return image.crop((math.floor(l*w), math.floor(t*h), math.ceil(r*w), math.ceil(b*h)))


def descriptor(image, field):
    small = np.asarray(ImageOps.contain(image, (256, 256)))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    amber = (hue >= 4) & (hue <= 35) & (saturation >= 85) & (value >= 40)
    h, w = small.shape[:2]
    if field == "scale":
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        edges = np.hypot(gx, gy)
        vector = np.array([abs(math.log(w/h)), *np.percentile(edges, [50, 75, 90, 95]),
                           float(amber.mean())])
    else:
        if amber.sum() < max(20, 0.03*w*h):
            return None
        vector = np.array([*np.percentile(hue[amber], [25, 50, 75]),
                           *np.percentile(saturation[amber], [25, 50, 75]),
                           *np.percentile(value[amber], [25, 50, 75])])
    return vector


def fit_centroids(samples, field):
    values = np.array([x for x, _ in samples])
    center, scale = values.mean(axis=0), np.maximum(values.std(axis=0), 1e-3)
    normalized = (values-center)/scale
    prototypes, radii = {}, {}
    for label in FIELDS[field]:
        points = normalized[[y == label for _, y in samples]]
        if len(points) < 2:
            continue
        prototype = np.median(points, axis=0)
        prototypes[label] = prototype.tolist()
        radii[label] = float(np.percentile(np.linalg.norm(points-prototype, axis=1), 80))
    return {"center": center.tolist(), "scale": scale.tolist(), "prototypes": prototypes,
            "radii": radii, "method": "development-only robust nearest centroid"}


def suggest(vector, model):
    if vector is None or len(model["prototypes"]) != 2:
        return dict(value=None, reason="Insufficient visible support")
    normalized = (vector - np.array(model["center"])) / np.array(model["scale"])
    ranked = sorted((float(np.linalg.norm(normalized-p)), name)
                    for name, p in model["prototypes"].items())
    distance, label = ranked[0]
    margin = ranked[1][0] - distance
    if distance > model["radii"][label] or margin < 0.85:
        return dict(value=None, reason="Outside development support or ambiguous",
                    distance=distance, margin=margin)
    return dict(value=label, distance=distance, margin=margin,
                reason="Experimental suggestion from the object crop")


def describe(image, model):
    return {field: suggest(descriptor(image, field), model[field]) for field in FIELDS}


def read_inscriptions(image):
    """Return positioned OCR candidates. A number alone never establishes capacity."""
    if max(image.size) > 1500:
        image = ImageOps.contain(image, (1500, 1500))
    image = ImageOps.autocontrast(image.convert("L"), cutoff=1)
    output = []
    inverse = {
        0: lambda u, v: (u, v), 90: lambda u, v: (1-v, u),
        180: lambda u, v: (1-u, 1-v), 270: lambda u, v: (v, 1-u),
    }
    try:
        for angle, back in inverse.items():
            rotated = image.rotate(angle, expand=True)
            blob = io.BytesIO()
            rotated.save(blob, format="PNG")
            result = subprocess.run(["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "11",
                                     "tsv"], input=blob.getvalue(), capture_output=True,
                                    timeout=8, check=True)
            for row in csv.DictReader(io.StringIO(result.stdout.decode("utf-8")), delimiter="\t"):
                token = (row.get("text") or "").strip()
                confidence = float(row["conf"])
                if confidence < 65 or not re.fullmatch(r"(?:[0-9][0-9.,]{0,3}|m[lL])", token):
                    continue
                x, y, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
                corners = [back(x/rotated.width, y/rotated.height),
                           back((x+w)/rotated.width, (y+h)/rotated.height)]
                bbox = [min(p[0] for p in corners), min(p[1] for p in corners),
                        max(p[0] for p in corners), max(p[1] for p in corners)]
                output.append(dict(text=token, score=confidence/100, crop_bbox=bbox, rotation=angle))
    except (OSError, subprocess.SubprocessError):
        return dict(status="unavailable", candidates=[], capacity=None, apparent_volume=None)
    return dict(status="review" if output else "unreadable",
                candidates=sorted(output, key=lambda item: -item["score"])[:20],
                capacity=None, apparent_volume=None)


def train(root):
    rows = train_rows(root)
    designs = {}
    results = {}
    for field, labels in FIELDS.items():
        for mode in ("scene", "crop"):
            frames = []
            for row in rows:
                im = read_image(root / "data" / row["image"])
                for obj in row["objects"]:
                    label = obj[field]
                    if label in labels:
                        frame = crop(im, obj["bbox"]) if mode == "crop" else im
                        frames.append((descriptor(frame, field), label, obj["burst"]))
            samples = [(x,y) for x,y,_ in frames if x is not None]
            if mode == "crop":
                designs[field] = fit_centroids(samples,field)
            # Diagnostic only. Burst holdout does not establish independent physical units.
            confusion = {label: {name: 0 for name in (*labels, "unknown")} for label in labels}
            for burst in sorted({burst for _,_,burst in frames}):
                fitting = [(x,y) for x,y,b in frames if b != burst and x is not None]
                if not fitting:
                    continue
                model = fit_centroids(fitting, field)
                for x, label, b in frames:
                    if b == burst:
                        guess = suggest(x, model)["value"] or "unknown"
                        confusion[label][guess] += 1
            total = sum(sum(row.values()) for row in confusion.values())
            answered = sum(sum(row[k] for k in labels) for row in confusion.values())
            correct = sum(confusion[k][k] for k in labels)
            results[f"{field}-{mode}"] = dict(total=total, answered=answered, correct=correct,
                                               confusion=confusion,
                                               design="same-session leave-burst-out diagnostic")
    design_path = root / "artifacts" / "attribute_model.json"
    write_json(design_path, designs)
    write_json(root / "artifacts" / "attribute_diagnostics.json", results)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    train(parser.parse_args().root)
