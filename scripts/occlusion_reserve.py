"""Freeze visually reviewed DVIDS acquisitions and replay the released engine."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.grounding_backend import analyze
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.occlusion_detector import DATA, FRESH, PLAN, raster
from scripts.part_verifier_study import configure_cpu
from scripts.public_training_intake import raster_hashes

# Bounding boxes were drawn in the original preview coordinate system before inference.
SOURCES = (
    ("8071796", "2310", "Kelly Goonan", "Patrick SFB 2023-10-14 Kelly Goonan",
     "920th-aeromedical-staging-squadron-provides-vaccine-during-drill-weekends",
     (1000, 1107), [(56, 856, 502, 1045), (115, 861, 615, 1079)],
     "Two distinct capped syringes overlap at the rear. The vial is a distractor."),
    ("9584272", "2603", "Bailey Breving", "Smyrna 2026-03-21 Bailey Breving",
     "tennessee-national-guard-medics-conduct-trauma-validation-during-68w-recertification",
     (1000, 667), [(609, 345, 664, 416)],
     "One syringe assembly obscured by both hands. The container and hand are excluded."),
)


def freeze():
    if FRESH.exists():
        raise FileExistsError(FRESH)
    history, private, fingerprints = protection()
    rows = []
    fit = read_json(PLAN)["roles"]
    for identifier, month, author, group, slug, size, boxes, note in SOURCES:
        path = DATA / f"{identifier}.jpg"
        page = f"https://www.dvidshub.net/image/{identifier}/{slug}"
        checksum = digest(path)
        if (identifier in history["ids"] or any(author.split()[-1].casefold() in name for name in history["authors"])
                or page in history["urls"] or checksum in history["sha256"] | private
                or any(r["sha256"] == checksum for role in fit.values() for r in role)
                or fingerprint_collision(raster_hashes(path), fingerprints | history["dhash"])):
            raise ValueError(f"Existing exposure detected for {identifier}.")
        w, h = size
        row = dict(id=f"dvids-{identifier}", image=str(path), sha256=checksum, source_url=page,
                   original_url=f"https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/{month}/{identifier}/1000w_q95.jpg",
                   author=author, group=group, license="Public domain (DVIDS)", review=note,
                   objects=[dict(bbox=[l/w, t/h, r/w, b/h]) for l, t, r, b in boxes])
        if raster(row).size != size:
            raise ValueError(f"Unexpected preview size for {identifier}.")
        rows.append(row)
    write_json(FRESH, dict(rows=rows, implementation=digest(__file__), fitting_design=digest(PLAN),
                          exclusion="Previously indexed source, author, exact hash and transformed dHash screened.",
                          unscored={"6486790": "The tray was conservatively excluded from exact-count scoring "
                                               "because several overlapping instances cannot be fully traced."}))


def released():
    target = DATA / "reserve-released.json"
    locked = read_json(FRESH)
    if locked["implementation"] != digest(__file__) or locked["fitting_design"] != digest(PLAN):
        raise ValueError("The prospective reference or fitting design changed.")
    configure_cpu()
    root = Path.cwd()
    model = MedSyringeEngine(root)
    configure_device(model, root, "xpu")
    try:
        frames = {}
        for row in locked["rows"]:
            output = analyze(model, raster(row), False)
            checks = output["detection_checks"]
            if checks["status"] != "available" or checks["primary_backend"] != "xpu":
                raise ValueError("The released Intel detector is unavailable.")
            frames[row["id"]] = output["objects"]
        write_json(target, dict(reference_sha256=digest(FRESH), frames=frames))
    finally:
        model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "released"))
    {"freeze": freeze, "released": released}[parser.parse_args().stage]()
