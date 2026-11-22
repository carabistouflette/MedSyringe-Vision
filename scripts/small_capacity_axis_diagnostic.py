"""Diagnose whether a tighter barrel-axis crop restores legibility on oblique public photographs."""
from pathlib import Path

import cv2
from PIL import Image

from medsyringe.attributes import crop
from medsyringe.capacity_ocr import barrel_angle, bounded_image, rotated_image
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.small_capacity import SmallCapacity
from scripts.small_field_public_transfer import FREEZE, OUTPUT

RESULT = Path("artifacts/detection_hillclimb/small_capacity_axis_diagnostic.json")


def run():
    reader = SmallCapacity()
    outcomes = []
    for case in read_json(FREEZE)["cases"]:
        photo = crop(read_image(case["image"]), case["bbox"])
        bgr = bounded_image(photo, edge=1500)
        rotation = barrel_angle(bgr)
        oriented, inverse = rotated_image(bgr, rotation)
        height, width = oriented.shape[:2]
        trials = []
        for ratio in (.30, .40, .50):
            span = min(width, round(ratio * height))
            left = (width - span) // 2
            view = oriented[:, left:left+span]
            if min(view.shape[:2]) < 8:
                continue
            findings = reader.read(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
            trials.append(dict(rotation=rotation, oriented_box=[left, 0, left+span, height],
                               inverse_transform=inverse.tolist(), ratio=ratio,
                               candidate=findings["capacity"], views=findings["views"]))
        outcomes.append(dict(id=case["id"], reference=case["source_capacity_ml"], trials=trials))
        print(case["id"], rotation, [None if not r["candidate"] else r["candidate"]["value"] for r in trials],
              flush=True)
    write_json(RESULT, dict(transfer=digest(OUTPUT), references=digest(FREEZE), cases=outcomes,
                            scope="Post-evaluation diagnostic on public layouts already observed. No release claim."))


if __name__ == "__main__":
    run()
