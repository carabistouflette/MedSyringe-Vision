"""Exercise new clear 60-scale images upright and after physical 180° rotation."""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

from medsyringe.measurement.attributes import crop
from medsyringe.measurement.capacity_ocr import SceneOCR
from medsyringe.data import digest, read_json, write_json
from medsyringe.measurement.marking_views import agreement, contrast_view, supplement
from medsyringe.measurement.markings_guard import confirm_untyped_capacity
from scripts.experiments.expanded.expanded_annotations import ANNOTATIONS, ART
from scripts.experiments.expanded.expanded_markings import LOCK, RELEASE, read
from scripts.experiments.expanded.expanded_proposals import image_for

CONTROL_IDS = {"IMG_3167", "IMG_3168", "IMG_3170", "IMG_3172", "IMG_3175", "IMG_3176"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    source = parser.parse_args().source
    if read_json(LOCK)["selected"] != "both":
        raise ValueError("The control requires the frozen two-view policy.")
    rows = [r for r in read_json(ANNOTATIONS)["rows"] if r["id"] in CONTROL_IDS]
    settings = read_json(RELEASE)["options"]
    reader = SceneOCR()
    results = []
    for row in rows:
        image = ImageOps.contain(crop(image_for(row, source), row["objects"][0]["bbox"]), (1900, 1900))
        for angle, view in ((0, image), (180, image.transpose(Image.Transpose.ROTATE_180))):
            original = read(reader, view, settings)
            final = original
            if not original.get("capacity") or not original.get("scale"):
                alternative = agreement(*(read(reader, contrast_view(view, name), settings)
                                           for name in ("clahe", "enhanced")))
                final = confirm_untyped_capacity(supplement(original, alternative), alternative)
            observed = final.get("capacity")
            results.append(dict(id=row["id"], orientation=angle, expected=60.,
                                original=original.get("capacity"), guarded=observed))
            print(row["id"], angle, observed and observed["value"], flush=True)
    write_json(ART / "markings_rotation_controls.json", dict(cases=results, freeze_sha256=digest(LOCK),
               scope="Real 60-scale JPEG crops with raster rotation. These were not used for marking selection."))


if __name__ == "__main__":
    main()
