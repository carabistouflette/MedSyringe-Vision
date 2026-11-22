"""Freeze visually reviewed, acquisition-separated public references for recrop testing."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision, protection
from scripts.experiments.crowds.native_candidate_bank import cohorts
from scripts.experiments.crowds.native_candidate_reserve import references as prior_native
from scripts.experiments.occlusion.occlusion_detector import ART, path_for, raster
from scripts.experiments.public_data.public_training_intake import raster_hashes
from scripts.experiments.relocalization.recrop_proposal_bank import WORK
from scripts.experiments.identity.syringe_assembly_reserve import references as prior_assembly

REFERENCE = ART / "recrop_verifier_reserve.json"
RELEASE = WORK / "reserve-release.json"
POLICY = ART / "recrop_verifier_policy.json"
SCREEN = WORK / "source-screen"

SOURCES = (
    ("6576166", "2103", "preparing-covid-19-syringe", "Cynthia McIntyre", "Fort Hunter Liggett 2021-03-23",
     (1000, 743), ((514, 444, 747, 630),), "One diagonally held syringe, partly occluded by gloves."),
    ("6630055", "2105", "col-miller-visits-deployed-airmen-detroit-michigan", "Wesley Farnsworth",
     "Detroit Ford Field 2021-05-03", (1000, 1469), ((543, 668, 602, 1064),),
     "One narrow vertical syringe between vial and gloves. The vial is excluded."),
    ("3904803", "1710", "flu-vaccinaions-aboard-uss-bonhomme-richard-lhd-6", "Diana Quinlan",
     "USS Bonhomme Richard 2017-10-26", (1000, 1444), ((512, 421, 664, 857),),
     "One foreshortened syringe under a hand. The vial is excluded."),
    ("6481146", "2101", "leaders-across-okinawa-receive-covid-19-vaccine", "Zachary Larsen",
     "Camp Foster Okinawa 2021-01-12", (1000, 1500), ((427, 589, 565, 1092),),
     "One vertical syringe with a visible attached safety piece, excluding the vial."),
    ("9957502", "2609", "coyote-storm-262", "Samantha Melecio", "Davis-Monthan 2026-09-17",
     (1000, 667), ((512, 232, 556, 352),), "One small syringe in a wide scene, including its visible needle."),
    ("8071796", "2310", "920th-aeromedical-staging-squadron-provides-vaccine-during-drill-weekends",
     "Kelly Goonan", "Patrick Space Force Base 2023-10-14", (1000, 1107),
     ((21, 851, 500, 1028), (72, 859, 624, 1057)),
     "Two overlapping syringes with independent plungers, barrels and blue hubs. Their distal ends are soft."),
    ("9090709", "2506", "mdg-cares-offutt-patients", "Daniel Martinez", "Offutt 2025-06-02",
     (1000, 714), (), "Long pipette in use with sample vials. No visible syringe assembly."),
    ("10052835", "2610", "two-paths-one-mission-bjach-graduates-army-laboratory-specialists", "Jean Graves",
     "Bayne-Jones Fort Polk 2026-10-27", (1000, 1333), (), "A long held pipette and sample. No visible syringe."),
    ("9364294", "2510", "walters-way-behind-scenes-lab-professionals-walter-reed-national-military-medical-center",
     "Ricardo Reyes-Guevara", "Walter Reed 2025-10-17", (1000, 688), (),
     "Micropipette, plastic sample tubes and centrifuge. No visible syringe assembly."),
    ("9221790", "2507", "george-washington-conducts-fueling-sea-with-badlands-trader", "Bruce Morgan",
     "George Washington Indian Ocean 2025-07-27", (1000, 1333), (),
     "Glass pipette, separatory funnel and collection glassware. No visible syringe assembly."),
    ("9242074", "2508", "nreip-orise-interns-present-research-namru-san-antonio", "Burrell Parmer",
     "NAMRU San Antonio 2025-07-28", (1000, 714), (),
     "A held laboratory pipetting tool and sample containers. No visible syringe assembly."),
    ("9261559", "2508", "177th-civil-engineer-squadron-trains-with-nj-state-agencies-world-cup-security",
     "Alex Cadavid", "West Windsor 2025-07-31", (1000, 666), (),
     "A hand-held transfer pipette among glass and plastic containers. No visible syringe assembly."),
)
SCREEN_EXCLUSIONS = {
    "6486790": "Heavily overlapping rows. Complete instance boundaries could not be agreed from the preview.",
    "6476041": "Motion or focus blur obscures visible syringe endpoints.",
    "6892731": "Blur and superposition make the syringe count and boundaries uncertain.",
    "6630026": "Overfilled tray with indistinguishable assemblies and the same acquisition as 6630055.",
    "5850115": "Hands obscure most of the device, and visible endpoints are unclear.",
    "6897158": "Hands conceal most of the syringe.",
    "3053135": "The device partly merges with the gloved hand; a high-quality reference was unavailable.",
    "8355445": "The workbench and hands obscure the assembly.",
    "injectomat / polish-pump": "Public original retrieval returned HTTP 429; no images admitted.",
    "1144189 / 8819828 / 9905634 / 6563014": "Official preview retrieval returned HTTP 404.",
    "9468367 / 9918854": "Public pipette photos already present in the native-crop reserve.",
    "441102": "Specular reflections and tightly overlapped metal dental syringes make some boxes ambiguous.",
    "6573372 / 6573371": "The hands and tray obstruct the visibility of the syringe assemblies.",
}


def freeze():
    if REFERENCE.exists() or POLICY.exists():
        raise FileExistsError("The reference must be frozen once, before model selection.")
    history, private_sha, private_pixels = protection()
    older = [*(r for role in ("fit", "development", "evaluation") for r in cohorts(role)),
             *prior_native(), *prior_assembly()]
    hashes = history["sha256"] | private_sha | {r.get("derived_sha256", r["sha256"]) for r in older}
    pixels = history["dhash"] | private_pixels
    for row in older:
        pixels.update(raster_hashes(path_for(row)))
    authors = history["authors"] | {lexical_key(r["author"]) for r in older if r.get("author")}
    urls = history["urls"] | {url_key(r["source_url"]).rstrip("/") for r in older if r.get("source_url")}
    groups = {r["group"] for r in older}
    records, exclusions = [], dict(SCREEN_EXCLUSIONS)
    for key, month, slug, author, group, size, positions, note in SOURCES:
        path = SCREEN / f"{key}.jpg"
        source = f"https://www.dvidshub.net/image/{key}/{slug}"
        asset = f"https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/{month}/{key}/1000w_q95.jpg"
        checksum, fingerprint = digest(path), raster_hashes(path)
        contributor, page = lexical_key(author), url_key(source).rstrip("/")
        if (key in history["ids"] or page in urls or group in groups or checksum in hashes
                or any(contributor in other for other in authors) or fingerprint_collision(fingerprint, pixels)):
            exclusions[key] = "Known contributor, image, acquisition, or perceptual collision in the project history."
            continue
        boxes = [[a/size[0], b/size[1], c/size[0], d/size[1]] for a, b, c, d in positions]
        row = dict(id=f"recrop-{key}", image=str(path), sha256=checksum, source_url=source,
                   original_url=asset, author=author, group=group,
                   license="Public domain (DVIDS item page)", objects=[dict(bbox=b) for b in boxes], review=note)
        if raster(row).size != size or any(not valid_box(box) for box in boxes):
            raise ValueError(f"Reference dimensions or coordinates do not match for {key}.")
        hashes.add(checksum)
        pixels.update(fingerprint)
        groups.add(group)
        authors.add(contributor)
        urls.add(page)
        records.append(row)
    if len(records) < 6 or not 2 <= sum(not r["objects"] for r in records) <= len(records) - 3:
        raise ValueError(f"Insufficient independent references: {len(records)}. Exclusions: {exclusions}")
    write_json(REFERENCE, dict(code=digest(__file__), rows=records, exclusions=exclusions,
                              scope="Visually reviewed references from distinct public acquisitions. Exact and "
                                    "transformed perceptual hashes checked before inference. One reviewer."))
    print(len(records), "acquisitions", sum(len(r["objects"]) for r in records), "syringes", flush=True)


def references():
    record = read_json(REFERENCE)
    if record["code"] != digest(__file__):
        raise ValueError("The frozen reserve protocol changed.")
    return record["rows"]


def assert_frozen_policy():
    from scripts.experiments.relocalization.recrop_verifier import sealed

    return sealed()


def baseline():
    from medsyringe.runtime.grounding_backend import analyze
    from medsyringe.runtime.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.experiments.expanded.sam3_study import configure_cpu

    assert_frozen_policy()
    rows = references()
    binding = dict(reference=digest(REFERENCE), policy=digest(POLICY))
    state = read_json(RELEASE) if RELEASE.exists() else dict(binding=binding, outputs={})
    if state["binding"] != binding:
        raise ValueError("Released predictions belong to different references or policy.")
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    configure_device(engine, Path.cwd(), "xpu")
    try:
        for row in rows:
            if row["id"] in state["outputs"]:
                continue
            detection = analyze(engine, raster(row), False)
            check = detection["detection_checks"]
            if check["status"] != "available" or check["primary_backend"] != "xpu":
                raise ValueError(f"Released detector unavailable for {row['id']}.")
            state["outputs"][row["id"]] = detection["objects"]
            write_json(RELEASE, state)
            print(row["id"], len(detection["objects"]), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "baseline"))
    {"freeze": freeze, "baseline": baseline}[parser.parse_args().stage]()
