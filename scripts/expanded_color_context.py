"""Freeze a context check and audit it on separately reviewed public color controls."""
import argparse
from pathlib import Path
from statistics import median

import numpy as np
from PIL import ImageOps

from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.liquid_color import describe_region, display_rgb, lab_pixels
from medsyringe.liquid_context import contextual_support
from medsyringe.liquid_region import RegionFeatures
from scripts.expanded_liquid import ART, LOCK as FIRST_LOCK, mask_syringe, records, support
from scripts.expanded_proposals import image_for

LOCK = ART / "color_context_freeze.json"
CONTROL_IDS = {"163322504", "178849232", "201441851", "146786108", "40322872", "19682206"}


def controls():
    source = [r for name in ("sam3_reserve", "rfdetr_syringe_reserve")
              for r in read_json(Path("artifacts") / (name+".json"))["rows"]]
    return [r for r in source if r["id"] in CONTROL_IDS and len(r["objects"]) == 1]


def retain(frame, model, cutoff, threshold):
    candidate = support(frame, model, cutoff)
    return contextual_support(frame, candidate, threshold)


def select():
    if LOCK.exists():
        raise ValueError("The contextual variant is already frozen.")
    win = read_json(FIRST_LOCK)["winner"]
    calibration = list(records("calibration"))
    original = read_json(ART / "liquid_calibration.json")["selected_frames"]
    outputs = []
    for threshold in (5., 8., 12., 16.):
        ids = [r["id"] for r, f in calibration if retain(f, win["model"], win["cutoff"], threshold).any()]
        seen = [original[i] for i in ids if original[i]["supported"]]
        scores = dict(covered=len(seen), measured=len(ids), median_lstar=median(r["lstar_error"] for r in seen),
                      median_delta_e=median(r["lab_delta_e76"] for r in seen))
        outputs.append(dict(separation=threshold, scores=scores, answered=ids))
    result = max(outputs, key=lambda r: (r["scores"]["covered"], -r["scores"]["median_lstar"], r["separation"]))
    write_json(LOCK, dict(selection=result, tried=outputs,
                initial_freeze_sha256=digest(FIRST_LOCK), code_sha256=digest("medsyringe/liquid_context.py"),
                public_rows=controls(), scope="Public controls previously used for detection and newly labeled for color."))
    print("Context calibration", [(t["separation"], t["scores"]) for t in outputs], flush=True)


def audit():
    from scripts.sam3_study import Segmenter, configure_cpu

    configure_cpu()
    config = read_json(LOCK)
    if config["initial_freeze_sha256"] != digest(FIRST_LOCK) or config["code_sha256"] != digest("medsyringe/liquid_context.py"):
        raise ValueError("The contextual variant changed since calibration.")
    win = read_json(FIRST_LOCK)["winner"]
    threshold = config["selection"]["separation"]
    inside = {r["id"]: bool(retain(f, win["model"], win["cutoff"], threshold).any()) for r, f in records("test")}
    model = Segmenter(504, "xpu")
    encoder = RegionFeatures("xpu")
    outside = {}
    for row in config["public_rows"]:
        source = image_for(row, Path("."))
        region = ImageOps.contain(crop(source, row["objects"][0]["bbox"]), (420, 420))
        display, profile = display_rgb(region)
        rgb = np.asarray(display)
        frame = dict(lab=lab_pixels(rgb), mask=mask_syringe(model, region), grid=encoder.extract(region))
        base = support(frame, win["model"], win["cutoff"])
        refined = contextual_support(frame, base, threshold)
        outside[row["id"]] = dict(baseline=bool(base.any()), candidate=bool(refined.any()),
                                 candidate_observation=describe_region(rgb, refined, profile))
        print(row["id"], outside[row["id"]]["baseline"], outside[row["id"]]["candidate"], flush=True)
    write_json(ART / "color_context_audit.json", dict(freeze_sha256=digest(LOCK), clear_regression=inside,
               public_controls=outside, false_regression=sum(inside.values()),
               false_public=sum(o["candidate"] for o in outside.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "audit"))
    select() if parser.parse_args().stage == "select" else audit()


if __name__ == "__main__":
    main()
