"""Check frozen chamber appearance on axis-aligned crops from a public staged photograph."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from medsyringe.measurement.chamber_appearance import inspect_chamber
from medsyringe.data import digest, read_image, read_json, write_json
from scripts.experiments.measurement.small_field_study import ART

SOURCE = Path("data/public-web-collection/images/openverse-67ec0d65-685b-433e-9226-4830206764d0.jpg")
SOURCE_MANIFEST = Path("data/public-web-collection/manifest.json")
REFERENCE = ART / "chamber_public_transfer_reference.json"
RESULT = ART / "chamber_public_transfer_results.json"
ALGORITHM = Path("medsyringe/measurement/chamber_appearance.py")
OBJECTS = (
    ("left-green", (88, 178), (198, 251), 30, "green"),
    ("left-orange", (142, 138), (222, 229), 27, "orange"),
    ("left-red", (175, 86), (245, 212), 28, "red"),
    ("center-yellow", (244, 61), (274, 217), 25, "yellow"),
    ("center-blue", (336, 81), (308, 220), 25, "blue"),
    ("right-orange", (405, 135), (333, 227), 28, "orange"),
    ("right-red", (456, 172), (352, 248), 28, "red"),
)


def inputs():
    item = next(row for row in read_json(SOURCE_MANIFEST)["rows"] if row["image"].endswith(SOURCE.name))
    if digest(SOURCE) != item["sha256"]:
        raise ValueError("The published staged photo changed.")
    return dict(algorithm=digest(ALGORITHM), raster=item["sha256"],
                source=item["source_url"], creator=item["author"], license="https://creativecommons.org/licenses/by/2.0/",
                acquisition=item["group"],
                coordinates="Approximate axis endpoints and crop width in source pixels, reviewed visually",
                rows=[dict(id=key, endpoints=[list(a), list(b)], width=width, apparent_hue=hue)
                      for key, a, b, width, hue in OBJECTS])


def strip(rgb, start, end, width):
    start, end = np.asarray(start), np.asarray(end)
    axis = end - start
    length = np.linalg.norm(axis)
    flank = np.array([axis[1], -axis[0]]) / length
    longitudinal = np.linspace(0, 1, round(length))
    lateral = np.linspace(-width / 2., width / 2., width)
    coords = start + longitudinal[:, None, None] * axis + lateral[None, :, None] * flank
    patch = cv2.remap(rgb, coords[..., 0].astype(np.float32), coords[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    return Image.fromarray(patch)


def run():
    frozen = inputs()
    if REFERENCE.exists() and read_json(REFERENCE) != frozen:
        raise ValueError("The candidate or acquisition changed after freezing.")
    if not REFERENCE.exists():
        write_json(REFERENCE, frozen)
    pixels = np.asarray(read_image(SOURCE))
    observations = []
    for row in frozen["rows"]:
        view = strip(pixels, *row["endpoints"], row["width"])
        outcome = inspect_chamber(view)
        print(row["id"], row["apparent_hue"], outcome["presence"], flush=True)
        observations.append(dict(**row, result=outcome))
    write_json(RESULT, dict(reference=digest(REFERENCE), rows=observations,
               scope="Seven manually aligned objects from a single staged acquisition. "
                     "The target is visible color. Fill in mL and opacity are unknown."))


if __name__ == "__main__":
    run()
