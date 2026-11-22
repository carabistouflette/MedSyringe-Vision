"""Freeze and score separate public acquisitions for the completeness score comparison."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, url_key
from scripts.experiments.ranking.completeness_calibration import ART, POLICY, WORK, sealed
from scripts.experiments.ranking.complete_syringe_reserve import references as prior_complete
from scripts.experiments.ranking.instance_separation_reserve import references as prior_separation
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision, protection
from scripts.experiments.crowds.native_candidate_bank import cohorts, views
from scripts.experiments.crowds.native_candidate_reserve import references as prior_native
from scripts.experiments.occlusion.occlusion_detector import INITIAL, path_for, raster
from scripts.experiments.public_data.public_training_intake import raster_hashes
from scripts.experiments.relocalization.recrop_reserve import references as prior_recrop
from scripts.experiments.relocalization.relocalization_reserve import references as prior_selector
from scripts.experiments.identity.syringe_assembly_reserve import references as prior_assembly

REFERENCE = ART / "completeness_calibration_reserve.json"
BASELINE = WORK / "released.json"
PROPOSALS = WORK / "proposals.json"
SOURCES = (
    ("36675186", "black-and-white-close-up-of-syringes-in-gloved-hand", "Laura Beauty Designer", (433,650),
     ((63,272,154,401),(171,205,292,402)), "Two visible syringe assemblies partly hidden by the same glove."),
    ("5921984", "person-holding-syringe-and-vial", "Thirdman", (433,650),
     ((222,194,310,363),), "Translucent syringe between gloves. Vial excluded."),
    ("34775445", "healthcare-professional-holding-a-syringe", "Farhad Irani", (528,650),
     ((254,114,372,486),), "Single visible capped syringe held upright between fingers."),
    ("6823562", "a-person-holding-a-syringe", "Artem Podrez", (560,328),
     ((323,82,354,211),), "Single narrow syringe. Hands conceal the middle and lower end."),
    ("4299424", "anonymous-lab-worker-using-pipette-for-injecting-sample-into-flask", "Jorge Sepulveda", (560,374),
     (), "A pipette and sample vessel."),
    ("8770736", "person-holding-a-pipette-into-a-test-tube", "Gustavo Fring", (560,374),
     (), "Clear pipette and several lab vessels."),
    ("29226623", "scientist-conducting-lab-experiment-with-test-tubes", "Sergei Starostin", (560,374),
     (), "A transfer tool and sample tubes."),
)
EXCLUSIONS = {
    "3936421": "Decorative gold syringe-like objects are cropped and type is uncertain.",
    "5998508": "Alternative single syringe. The plunger endpoint is partly hidden.",
    "32418225": "Previously screened. Prefilled-device boundaries are ambiguous in the package.",
}


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    h, private, private_pixels = protection()
    historical = [*(r for role in ("fit", "development", "evaluation") for r in cohorts(role)),
                  *prior_native(), *prior_assembly(), *prior_recrop(), *prior_selector(), *prior_separation(), *prior_complete()]
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
        row = dict(id=f"calibration-{key}", image=str(p), sha256=checksum, group=f"Pexels {who} {key}",
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
    from medsyringe.runtime.grounding_backend import analyze
    from medsyringe.runtime.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.experiments.expanded.sam3_study import configure_cpu

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
