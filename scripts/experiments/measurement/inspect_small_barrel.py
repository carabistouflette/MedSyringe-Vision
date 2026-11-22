"""Inspect a local syringe photograph or one normalized syringe crop."""
import argparse
from pathlib import Path

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_image, valid_box, write_json
from medsyringe.measurement.small_capacity import SmallCapacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--box", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    bounds = options.box or [0., 0., 1., 1.]
    if not valid_box(bounds):
        parser.error("Choose a normalized xyxy box that has a positive area.")
    picture = crop(read_image(options.image), bounds)
    result = SmallCapacity().read(picture)
    match = result["capacity"]
    report = dict(source_sha256=digest(options.image), source_bbox=bounds, experimental=True,
                  nominal_capacity=match, liquid=match["barrel"] if match else dict(
                      presence="unknown", fill_ml=None, interval_ml=None, appearance=None, relative_opacity=None,
                      reason="The localized small-barrel scale could not anchor this liquid inspection."),
                  ocr_diagnostics=result["views"])
    write_json(options.output, report)
    print(f"Saved experimental evidence to {options.output}")


if __name__ == "__main__":
    main()
