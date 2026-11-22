"""Seal new public images and record the released detector for the assembly study."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.grounding_backend import analyze
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from scripts.detection_public_benchmark import lexical_key, url_key
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.native_candidate_bank import cohorts, collect
from scripts.native_candidate_reserve import references as older_prospective
from scripts.occlusion_detector import ART, path_for, raster
from scripts.public_training_intake import raster_hashes
from scripts.sam3_study import configure_cpu
from scripts.thin_syringe_tiles import WORK

FOLDER = WORK / "prospective"
REFERENCE = ART / "syringe_assembly_reserve.json"
RELEASE = FOLDER / "released.json"
POLICY = ART / "syringe_assembly_selection.json"

SOURCES = (
    ("infusion", "Shyamal", "Syringe infusion pump 2021-11-18",
     "https://commons.wikimedia.org/wiki/File:Syringe_infusion_pump.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/5/5f/Syringe_infusion_pump.jpg",
     "https://creativecommons.org/licenses/by-sa/4.0/", (3735, 1826), (1000, 489),
     [(210, 264, 881, 386)], "An installed syringe. External tubing, pump clamps and machine are excluded."),
    ("6900270", "Maxwell Bass", "Fort Bliss 2021-10-19",
     "https://www.dvidshub.net/image/6900270/medical-team-provides-care-fort-bliss-dona-ana-complex",
     "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2110/6900270/1000w_q95.jpg",
     "Public domain (DVIDS)", (1000, 667), (1000, 667),
     [(770, 277, 818, 402)], "One thin gloved-hand syringe. The vial above the green hub is separate."),
    ("7538446", "Crystal Housman", "Moffett 2022-12-03",
     "https://www.dvidshub.net/image/7538446/129th-medical-group-administers-flu-shots",
     "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2212/7538446/1000w_q95.jpg",
     "Public domain (DVIDS)", (1000, 667), (1000, 667),
     [(483, 286, 709, 381)], "One assembly between gloves. Reference covers the independently visible device extent."),
    ("8169009", "Samuel Mensah", "Fort Detrick 2023-12-13",
     "https://www.dvidshub.net/image/8169009/usamriid-blood-testing-protocol-will-improve-disease-surveillance",
     "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2312/8169009/1000w_q95.jpg",
     "Public domain (DVIDS)", (1000, 714), (1000, 714), [],
     "Laboratory pipetting scene. Review finds no visible syringe assembly."),
    ("9143423", "Isabel Tanner", "Hurlburt Field 2025-06-17",
     "https://www.dvidshub.net/image/9143423/role-laboratory-medicine-military-health",
     "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2507/9143423/1000w_q95.jpg",
     "Public domain (DVIDS)", (1000, 667), (1000, 667), [],
     "A clear dropper and sample tube held with gloves. Review finds no syringe assembly."),
    ("rbc", "Hariprasad Shetty10", "RBC pipette 2025-12-15",
     "https://commons.wikimedia.org/wiki/File:RBC_Pipette.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/7/7e/RBC_Pipette.jpg",
     "https://creativecommons.org/licenses/by/4.0/", (2340, 4160), (2340, 4160), [],
     "A graduated glass pipette with a bulb and tubing. Review finds no syringe assembly."),
)


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    history, private, protected = protection()
    archived = [*(r for role in ("fit", "development", "evaluation") for r in cohorts(role)), *older_prospective()]
    hashes = history["sha256"] | private | {r["sha256"] for r in archived}
    pixels = history["dhash"] | protected
    for row in archived:
        pixels.update(raster_hashes(path_for(row)))
    authors = history["authors"] | {lexical_key(r["author"]) for r in archived if r.get("author")}
    urls = history["urls"] | {url_key(r["source_url"]).rstrip("/") for r in archived if r.get("source_url")}
    records = []
    for key, author, group, source, original, license, size, units, positions, note in SOURCES:
        path = FOLDER / f"{key}.jpg"
        checksum = digest(path)
        visual = raster_hashes(path)
        alias = lexical_key(author)
        page = url_key(source).rstrip("/")
        if (key in history["ids"] or any(alias in known for known in authors) or page in urls
                or checksum in hashes or fingerprint_collision(visual, pixels)):
            raise ValueError(f"Known contributor, photo or visual overlap for {key}")
        width, height = units
        boxes = [[l/width, t/height, r/width, b/height] for l, t, r, b in positions]
        record = dict(id=f"assembly-{key}", image=str(path), sha256=checksum, source_url=source,
                      original_url=original, license=license, author=author, group=group,
                      objects=[dict(bbox=p) for p in boxes], review=note)
        if raster(record).size != size or any(not valid_box(box) for box in boxes):
            raise ValueError(f"Invalid raster or reviewed box for {key}")
        hashes.add(checksum)
        pixels.update(visual)
        authors.add(alias)
        urls.add(page)
        records.append(record)
    write_json(REFERENCE, dict(code=digest(__file__), rows=records,
                              exclusions={"hamilton": "Author and visual overlap in project history.",
                                          "wpi": "Author and visual overlap in project history.",
                                          "electronic": "Author and visual overlap in project history.",
                                          "glass-two": "Known photographer and ambiguous detached-barrel target.",
                                          "6543401": "Visible instance boundaries are ambiguous under the gloves.",
                                          "10105840": "Editorial artwork with overlaid text and graphics."},
                              protocol="Source and annotation checks before inference. Group, contributor, SHA-256 "
                                       "and rotated or mirrored dHash at Hamming radius four."))
    print(len(records), "acquisitions admitted.", flush=True)


def references():
    record = read_json(REFERENCE)
    if record["code"] != digest(__file__):
        raise ValueError("The frozen public reference changed.")
    return record["rows"]


def baseline():
    rows = references()
    binding = dict(reference=digest(REFERENCE), policy=digest(POLICY))
    state = read_json(RELEASE) if RELEASE.exists() else dict(binding=binding, outputs={})
    if state["binding"] != binding:
        raise ValueError("The issued detector cache belongs to another reserve.")
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    configure_device(engine, Path.cwd(), "xpu")
    try:
        for row in rows:
            key = row["id"]
            if key in state["outputs"]:
                continue
            result = analyze(engine, raster(row), False)
            check = result["detection_checks"]
            if check["status"] != "available" or check["primary_backend"] != "xpu":
                raise ValueError(f"The released detector is unavailable for {key}.")
            state["outputs"][key] = result["objects"]
            write_json(RELEASE, state)
            print(key, "done", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "baseline", "proposals"))
    {"freeze": freeze, "baseline": baseline,
     "proposals": lambda: collect(references(), "assembly-reserve")}[parser.parse_args().stage]()
