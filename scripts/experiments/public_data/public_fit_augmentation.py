"""Seeded geometry and capture augmentation for reviewed public training records."""
from copy import deepcopy
from io import BytesIO
from math import cos, radians, sin

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

FAMILIES = ("geometry", "capture", "combined")


def transform(size, rng):
    width, height = size
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    theta = radians(rng.uniform(-180, 180))
    scale = rng.uniform(.63, 1.12)
    matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]) * scale
    center = np.array([width, height]) * .5
    translation = rng.uniform(-.10, .10, size=2) * np.array([width, height])
    jitter = rng.uniform(-.035, .035, size=(4, 2)) * np.array([width, height])
    targets = (corners - center) @ matrix.T + center + translation + jitter
    return cv2.getPerspectiveTransform(corners, targets.astype(np.float32))


def points(pixels, matrix):
    return cv2.perspectiveTransform(np.asarray(pixels, np.float32)[None], matrix)[0]


def geometry(image, objects, matrix):
    width, height = image.size
    scale = np.asarray([width, height])
    output = []
    for obj in objects:
        left, top, right, bottom = obj["bbox"]
        polygon = obj.get("polygon")
        if polygon is not None:
            vertices = np.asarray(polygon).reshape(-1, 2)
        else:
            vertices = [(left, top), (right, top), (right, bottom), (left, bottom)]
        projected = points(np.asarray(vertices) * scale, matrix) / scale
        envelope = np.concatenate((projected.min(axis=0), projected.max(axis=0)))
        bounds = envelope.clip(0, 1)
        areas = (envelope[2:] - envelope[:2]).prod(), (bounds[2:] - bounds[:2]).prod()
        if areas[0] <= 0 or areas[1] < .0005 or areas[1] / areas[0] < .72:
            # Skip the derivative if a target would become too ambiguous to annotate.
            return None
        result = {**obj, "bbox": bounds.tolist()}
        if polygon is not None:
            # A partial polygon needs true clipping, not coordinate-wise vertex clipping.
            if np.any((projected < 0) | (projected > 1)):
                return None
            result["polygon"] = projected.ravel().tolist()
        if obj.get("landmarks"):
            result["landmarks"] = {
                name: position.tolist() if ((position >= 0) & (position <= 1)).all() else None
                for name, raw in obj["landmarks"].items() if raw is not None
                for position in [points([np.asarray(raw) * scale], matrix)[0] / scale]
            }
        output.append(result)
    data = np.asarray(image.convert("RGB"))
    border = tuple(float(n) for n in np.median(np.concatenate((data[0], data[-1], data[:, 0], data[:, -1])), axis=0))
    projected = cv2.warpPerspective(data, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=border)
    return Image.fromarray(projected), output


def capture(image, rng):
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.63, 1.36))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.66, 1.28))
    width, height = image.size
    rgb = np.asarray(image, dtype=np.float32).copy()
    yy, xx = np.ogrid[0:1:height * 1j, 0:1:width * 1j]
    center = rng.uniform(0, 1)
    glare = 0.22 * np.exp(-((xx + .28 * yy - center) / rng.uniform(.04, .12))**2)
    rgb += (255 - rgb) * glare[:, :, None]
    blurred = Image.fromarray(np.uint8(rgb.clip(0, 255))).filter(
        ImageFilter.GaussianBlur(rng.uniform(.35, 2.2)))
    buffer = BytesIO()
    blurred.save(buffer, "JPEG", quality=int(rng.integers(38, 89)))
    buffer.seek(0)
    with Image.open(buffer) as loaded:
        return loaded.convert("RGB")


def derivative(image, row, family, seed):
    if family not in FAMILIES or row["split"] != "public-fit":
        raise ValueError("Augmentation is restricted to accepted public acquisition fit records.")
    rng = np.random.default_rng(seed)
    objects = deepcopy(row["objects"])
    worn = family in ("capture", "combined")
    matrix = np.eye(3)
    if family in ("geometry", "combined"):
        matrix = transform(image.size, rng)
        transformed = geometry(image, objects, matrix)
        if transformed is None:
            return None
        image, objects = transformed
    if worn:
        image = capture(image, rng)
        for item in objects:
            # Blur and glare can hide marks and alter colour or volume cues. Box labels survive.
            for key in ("syringe_type", "capacity_ml", "volume_ml", "liquid_colour",
                        "apparent_opacity", "transcription", "landmarks"):
                if key in item:
                    item[key] = None
    return image, objects, dict(family=family, seed=seed, matrix=matrix.tolist(),
                                attribute_supervision="masked pending legibility review" if worn else
                                "legibility must be separately reviewed")
