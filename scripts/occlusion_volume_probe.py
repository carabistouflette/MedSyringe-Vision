"""Audit apparent volume on a manually read public scale with frozen seal heads."""
from pathlib import Path

import numpy as np

from medsyringe.data import digest, read_json, write_json
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.occlusion_detector import ART, DATA
from scripts.occlusion_seal import FROZEN, REF, convert_to_volume, pick, supported
from scripts.public_training_intake import raster_hashes

SOURCE = "https://commons.wikimedia.org/wiki/File:Syringe_details_raw.jpeg"
IMAGE = DATA / "fifo-2ml.jpg"
REFERENCE = ART / "occlusion_volume_probe_reference.json"
RESULT = ART / "occlusion_volume_probe_results.json"


def run():
    if not REFERENCE.exists():
        history, private, private_visual = protection()
        checksum = digest(IMAGE)
        if checksum in private or fingerprint_collision(raster_hashes(IMAGE), private_visual):
            raise ValueError("This probe appears in the protected Fil Rouge references.")
        fit = [row for row in read_json(REF)["rows"] if row["role"] == "fit"]
        if any("fifo" in row["group"].casefold() or checksum == row["sha256"] for row in fit):
            raise ValueError("Probe overlaps a fitting acquisition.")
        historical = ("fifo" in history["authors"] or checksum in history["sha256"]
                      or fingerprint_collision(raster_hashes(IMAGE), history["dhash"]))
        write_json(REFERENCE, dict(freeze_sha256=digest(FROZEN), image=str(IMAGE), sha256=checksum,
                                   source=SOURCE, author="Fifo", license="CC BY-SA 2.5",
                                   prior_project_exposure=historical, shoulder=[756, 224], rear=[1347, 219],
                                   radius=49., rim=[1018., 222.],
                                   marked_scale=[dict(ml=ml, point=[x, y]) for ml, x, y in
                                                 ((0, 789, 263), (.5, 884, 262), (1, 981, 261),
                                                  (1.5, 1076, 260), (2, 1172, 259))],
                                   tick_spacing_ml=.1,
                                   notes="Image shows a dark liquid-facing seal transition. Author describes the 2 mL "
                                         "syringe as half full approximately. The visual rim reads roughly 1.2 mL. "
                                         "There is no physical volume measurement."))
    row = read_json(REFERENCE)
    if row["freeze_sha256"] != digest(FROZEN):
        raise ValueError("The probe is bound to a different frozen seal model.")
    frame = supported(row)
    policies = read_json(FROZEN)["policies"]
    tolerance = max(2., min(.22*frame["radius"], .045*frame["length"]))
    observed = {"rim_status": "estimated", "location": frame["target"]}
    visual = convert_to_volume(row, frame, observed, tolerance)
    results = {}
    for name, policy in policies.items():
        candidate = pick(frame, policy["head"])
        accepted = candidate["score"] >= policy["floor"] and candidate["margin"] >= policy["margin"]
        candidate["rim_status"] = "estimated" if accepted else "unknown"
        volume = convert_to_volume(row, frame, candidate, tolerance)
        results[name] = dict(rim=candidate, volume=volume,
                             interval_covers_visual_midpoint=(
                                 volume["low_ml"] <= float(np.mean([visual["low_ml"], visual["high_ml"]])) <= volume["high_ml"]
                                 if volume["status"] == "visual interval" else None))
    write_json(RESULT, dict(reference_sha256=digest(REFERENCE), visual_reference=visual, trials=results,
                            scope="One loose visual volume reference for calibration. A single scene cannot validate coverage."))
    print("Reviewed visual interval:", visual)
    print("Frozen automatic arms:", {k: v["volume"] for k, v in results.items()})


if __name__ == "__main__":
    run()
