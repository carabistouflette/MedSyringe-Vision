"""Evaluate one-factor capture perturbations on held reference crops."""
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.measurement.small_capacity import SmallCapacity
from scripts.experiments.measurement.small_field_study import REFERENCE, render

OUTPUT = Path("artifacts/detection_hillclimb/small_barrel_robustness.json")


def jpeg(photo):
    buffer = BytesIO()
    photo.save(buffer, format="JPEG", quality=36)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def perspective(photo):
    w, h = photo.size
    source = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dest = np.float32([[.048*w, .028*h], [.975*w, 0], [0, .96*h], [w-1, h-1]])
    warp = cv2.getPerspectiveTransform(source, dest)
    pixels = cv2.warpPerspective(np.asarray(photo), warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(pixels)


def run():
    row = read_json(REFERENCE)["rows"][0]
    if digest(row["image"]) != row["sha256"]:
        raise ValueError("The public reference changed.")
    real = crop(read_image(row["image"]), row["bbox"])
    synth, _, _ = render(2.7, (49, 49, 51), ((25, 181, 230), .43))
    manipulations = {
        "native": lambda photo: photo,
        "jpeg_q36": jpeg,
        "gaussian_blur_0.9px": lambda photo: photo.filter(ImageFilter.GaussianBlur(.9)),
        "brightness_0.78": lambda photo: ImageEnhance.Brightness(photo).enhance(.78),
        "brightness_1.15": lambda photo: ImageEnhance.Brightness(photo).enhance(1.15),
        "perspective": perspective,
    }
    reader = SmallCapacity()
    observations = []
    for frame, picture, state, volume in (("public development photo", real, "empty", 0.),
                                          ("rendered blue column", Image.fromarray(synth), "present", 2.7)):
        for name, transform in manipulations.items():
            test = reader.read(transform(picture))
            accepted = test["capacity"]
            status = accepted["barrel"] if accepted else {}
            observed = dict(image=frame, variant=name, expected_capacity=5, expected_presence=state,
                            expected_volume_ml=volume, result=accepted,
                            correct_capacity=bool(accepted and accepted["value"] == 5),
                            correct_presence=status.get("presence") == state,
                            volume_in_interval=(status["interval_ml"][0] <= volume <= status["interval_ml"][1]
                                                if status.get("interval_ml") else None))
            observations.append(observed)
            print(frame, name, observed["correct_capacity"], observed["correct_presence"],
                  observed["volume_in_interval"], flush=True)
    write_json(OUTPUT, dict(cases=observations, reader_sha256=digest("medsyringe/measurement/small_capacity.py"),
                            scope="Two parent photographs or renders. Perturbations share their parent. "
                                  "No training or physical opacity measured."))


if __name__ == "__main__":
    run()
