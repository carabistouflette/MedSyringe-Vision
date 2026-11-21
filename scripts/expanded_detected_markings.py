"""Verify the six corroborated recoveries on actual detector crops."""
import argparse
from pathlib import Path

from PIL import ImageOps

from medsyringe.attributes import crop
from medsyringe.capacity_ocr import SceneOCR
from medsyringe.data import read_json, write_json
from medsyringe.marking_views import agreement, contrast_view, supplement
from medsyringe.markings_guard import confirm_untyped_capacity
from scripts.expanded_annotations import ANNOTATIONS, ART
from scripts.expanded_markings import RELEASE, read
from scripts.expanded_proposals import image_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    source = parser.parse_args().source
    readings = read_json(ART / "markings_calibration.json")["evidence"]
    selected = {k for k, v in readings["both"].items() if v["capacity"] and not readings["baseline"][k]["capacity"]}
    # These six frames already had accepted pre-SAM detections. The stored XPU
    # Grounding DINO replay uses the same geometric branch as the CPU release.
    boxes = read_json("data/expanded/calibration-xpu.json")["frames"]
    options = read_json(RELEASE)["options"]
    floor = read_json("artifacts/detector_release.json")["cutoff"]
    results = []
    reader = SceneOCR()
    for row in read_json(ANNOTATIONS)["rows"]:
        identifier = row["id"]
        if identifier not in selected:
            continue
        baselines = [p for p in boxes[identifier]["prior"] if p["score"] >= floor]
        if len(baselines) != 1:
            results.append(dict(id=identifier, objects=len(baselines), original=None, guarded=None))
            continue
        photo = crop(image_for(row, source), baselines[0]["bbox"])
        im = ImageOps.contain(photo, (1900, 1900))
        base = read(reader, im, options)
        final = base
        if not base.get("capacity") or not base.get("scale"):
            joint = agreement(*(read(reader, contrast_view(im, name), options) for name in ("clahe", "enhanced")))
            final = confirm_untyped_capacity(supplement(base, joint), joint)
        results.append(dict(id=identifier, objects=1, original=base["capacity"], guarded=final["capacity"]))
        print(identifier, base["capacity"] and base["capacity"]["value"],
              final["capacity"] and final["capacity"]["value"], flush=True)
    write_json(ART / "markings_detected_recovery_check.json", dict(results=results,
               scope="Rescue-enriched calibration cases using detector-proposed boxes. Expected capacity 20 mL."))


if __name__ == "__main__":
    main()
