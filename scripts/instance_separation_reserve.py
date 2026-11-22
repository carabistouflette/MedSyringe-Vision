"""Freeze and score separate public acquisitions for instance separation."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.detection_public_benchmark import lexical_key, url_key
from scripts.instance_separation import ART, POLICY, WORK, sealed
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.native_candidate_bank import cohorts, views
from scripts.native_candidate_reserve import references as prior_native
from scripts.occlusion_detector import INITIAL, path_for, raster
from scripts.public_training_intake import raster_hashes
from scripts.recrop_reserve import references as prior_recrop
from scripts.relocalization_reserve import references as prior_selector
from scripts.syringe_assembly_reserve import references as prior_assembly

REFERENCE = ART / "instance_separation_reserve.json"
BASELINE = WORK / "released.json"
PROPOSALS = WORK / "proposals.json"
SOURCES = (
    ("33893209", "disposable-syringes-in-a-medical-container", "Jan van der Wolf", (1000,667),
     ((128,67,865,489), (52,146,850,544), (46,207,840,592), (106,258,862,586), (165,20,865,589)),
     "Five distinct seal and barrel paths. Four near-parallel assemblies and one crossing. "
     "Proximal fit and safety guard extents are approximate at overlaps. Previously screened visually, first scored here."),
    ("5973354", "close-up-shot-of-syringes", "Olia Danilevich", (433,650),
     ((87,125,251,313), (132,198,305,416), (167,291,356,551)),
     "Three distinct aligned syringes, with adjacent boxes overlapping."),
    ("15702696", "hands-holding-syringe", "Nicolas Rueda", (433,650),
     ((137,239,337,581),), "One partly occluded syringe and gloves."),
    ("9259935", "closeup-image-of-liquid-containers", "Jess Loiterton", (520,650), (),
     "A pipette in a clear laboratory bottle. No visible syringe assembly."),
    ("8325713", "close-up-view-of-colorful-liquids-in-test-tubes", "Kindel Media", (560,420), (),
     "A graduated transfer pipette above several test tubes. No syringe."),
)
EXCLUSIONS = {"6074939": "Possible extra transparent barrel between two visible colored barrels. Uncertain instance boundaries.",
              "7230842": "Gloves obscure the upper end. The other single-object control has a clearer outline."}


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    h, private, private_pixels = protection()
    historical = [*(r for role in ("fit", "development", "evaluation") for r in cohorts(role)),
                  *prior_native(), *prior_assembly(), *prior_recrop(), *prior_selector()]
    authors = h["authors"] | {lexical_key(r["author"]) for r in historical if r.get("author")}
    urls = h["urls"] | {url_key(r["source_url"]).rstrip("/") for r in historical if r.get("source_url")}
    hashes = h["sha256"] | private | {r["sha256"] for r in historical}
    pixels = h["dhash"] | private_pixels
    for r in historical:
        pixels.update(raster_hashes(path_for(r)))
    rows, rejected = [], dict(EXCLUSIONS)
    for key, slug, author, grid, boxes, note in SOURCES:
        p = WORK / f"{key}.jpg"
        page = f"https://www.pexels.com/photo/{slug}-{key}/"
        url = url_key(page).rstrip("/")
        checksum, dhash = digest(p), raster_hashes(p)
        who = lexical_key(author)
        if (key in h["ids"] or any(who in a for a in authors) or url in urls
                or checksum in hashes or fingerprint_collision(dhash, pixels)):
            rejected[key] = "Source, contributor, exact or near-image conflict with project history."
            continue
        normalized = [[x / dim for x, dim in zip(b, grid*2, strict=True)] for b in boxes]
        asset = f"https://images.pexels.com/photos/{key}/pexels-photo-{key}.jpeg?auto=compress&cs=tinysrgb&w=1000"
        row = dict(id=f"separation-{key}", image=str(p), sha256=checksum, group=f"Pexels {who} {key}",
                   author=author, source_url=page, original_url=asset, license="https://www.pexels.com/license/",
                   objects=[dict(bbox=b) for b in normalized], review=note)
        im = raster(row)
        if abs(im.width/im.height - grid[0]/grid[1]) > .004 or any(not valid_box(b) for b in normalized):
            raise ValueError(f"Invalid review coordinates for {key}.")
        rows.append(row)
        hashes.add(checksum)
        pixels.update(dhash)
        urls.add(url)
        authors.add(who)
    if len(rows) < 3 or not any(len(r["objects"]) > 1 for r in rows) or not any(not r["objects"] for r in rows):
        raise ValueError(f"Inadequate reserve after screening: {rejected}")
    write_json(REFERENCE, dict(code=digest(__file__), rows=rows, excluded=rejected,
                               scope="One visual reviewer, no predictions inspected before reference freeze."))
    print("admitted", [r["id"] for r in rows], "excluded", rejected, flush=True)


def references():
    record = read_json(REFERENCE)
    if record["code"] != digest(__file__):
        raise ValueError("The reference implementation changed.")
    return record["rows"]


def state(path):
    sealed()
    key = dict(reference=digest(REFERENCE), policy=digest(POLICY), checkpoint=read_json(INITIAL)["checkpoint_sha256"])
    record = read_json(path) if path.exists() else dict(binding=key, frames={})
    if record["binding"] != key:
        raise ValueError(f"Input mismatch for {path}.")
    return record


def baseline():
    from medsyringe.grounding_backend import analyze
    from medsyringe.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.sam3_study import configure_cpu

    record = state(BASELINE)
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), "xpu")
        for row in references():
            if row["id"] in record["frames"]:
                continue
            result = analyze(engine, raster(row), False)
            check = result["detection_checks"]
            if check["status"] != "available" or check["primary_backend"] != "xpu":
                raise ValueError("Released detector unavailable.")
            record["frames"][row["id"]] = result["objects"]
            write_json(BASELINE, record)
            print(row["id"], len(result["objects"]), flush=True)
    finally:
        engine.close()


def collect():
    import torch
    from rfdetr import RFDETRNano

    record = state(PROPOSALS)
    settings = read_json(INITIAL)
    if digest(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise ValueError("RF-DETR checkpoint changed.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = RFDETRNano.from_checkpoint(settings["checkpoint"], device="cpu")
    for row in references():
        if row["id"] not in record["frames"]:
            record["frames"][row["id"]] = views(model, raster(row))
            write_json(PROPOSALS, record)
            print(row["id"], "collected", flush=True)


def evidence():
    rows = references()
    base, raw = (state(p)["frames"] for p in (BASELINE, PROPOSALS))
    if set(base) != set(raw) or set(base) != {r["id"] for r in rows}:
        raise ValueError("Incomplete reserve evidence.")
    return rows, base, raw, {str(p): digest(p) for p in (REFERENCE, BASELINE, PROPOSALS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "baseline", "collect"))
    {"freeze": freeze, "baseline": baseline, "collect": collect}[parser.parse_args().stage]()
