"""Freeze grouped public references and collect independent relocalization evidence."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision, protection
from scripts.experiments.crowds.native_candidate_bank import cohorts, views
from scripts.experiments.crowds.native_candidate_reserve import references as prior_native
from scripts.experiments.occlusion.occlusion_detector import ART, INITIAL, path_for, raster
from scripts.experiments.public_data.public_training_intake import raster_hashes
from scripts.experiments.relocalization.recrop_proposal_bank import crop_proposals
from scripts.experiments.relocalization.recrop_reserve import references as prior_recrop
from scripts.experiments.relocalization.relocalization_selector import POLICY, WORK, sealed
from scripts.experiments.identity.syringe_assembly_reserve import references as prior_assembly

REFERENCE = ART / "relocalization_selector_reserve.json"
RELEASE = WORK / "released.json"
PROPOSALS = WORK / "proposals.json"

# Review coordinates are in the listed preview grid. Inference uses the downloaded raster.
PEXELS = (
    ("19260361", "surgeon-holding-syringes", "Mehmet Turgut Kirkgoz", (433, 650),
     ((133, 83, 196, 450), (176, 64, 321, 454)), "Two devices whose needle tips meet near the glove."),
    ("4210557", "syringes-with-medical-drugs-on-yellow-background", "Kaboompics.com", (433, 650),
     ((92, 268, 384, 422), (60, 362, 368, 524)), "Two parallel syringes, each with an attached visible needle."),
    ("3786122", "syringes-and-petri-dish-on-white-background", "Anna Shvets", (433, 650),
     ((230, 218, 280, 464), (154, 326, 291, 605)), "Two syringes with overlapping boxes and separate visible bodies."),
    ("8670507", "close-up-shot-of-a-person-holding-syringes", "Mikhail Nilov", (433, 650),
     ((131, 189, 199, 310), (151, 173, 206, 306), (166, 160, 215, 301),
      (219, 149, 248, 302), (239, 160, 286, 303), (250, 170, 307, 305)),
     "Six independently visible green plungers. Some distal ends are occluded by adjacent syringes and the hand."),
    ("7722598", "photo-of-pieces-of-disposable-syringes", "Tara Winstead", (560, 374),
     ((151, 111, 198, 303), (213, 138, 251, 338), (247, 90, 293, 276),
      (301, 118, 341, 314), (344, 100, 384, 289)), "Five complete arranged syringes on a low-contrast background."),
    ("954583", "person-putting-a-drop-on-test-tube", "Martin Lopez", (560, 374), (),
     "A pipette and tubes. No visible syringe assembly."),
    ("37499105", "scientist-holding-pipette-and-test-tubes-close-up", "Jahra Tasfia Reza", (433, 650), (),
     "A glass transfer tool and beaker. No visible syringe assembly."),
)
DVIDS = (
    ("9905634", "csl-kismayo-medical-training", "Abigail Duell", "Kismayo training 2026-08-25",
     (1000, 665), ((440, 135, 535, 358),), "One partly occluded hand-held syringe with visible plunger and hub. Vial excluded."),
)
VISUAL_EXCLUSIONS = {
    "9340582": "Dense transparent tray with indistinct visible extents per assembly.",
    "6943307": "Background objects out of focus and several foreground safety parts overlap.",
    "33893209": "Overlapping safety guards and crop edges make individual extents uncertain.",
    "6470987": "Small tools with partly hidden endpoints.",
    "6672204": "Front assembly is heavily occluded by hands.",
    "4823468": "Most of the syringe is hidden by fingers.",
    "19677805": "Multiple assemblies viewed end-on and hidden behind hands.",
}


def sources():
    for key, slug, author, size, boxes, note in PEXELS:
        page = f"https://www.pexels.com/photo/{slug}-{key}/"
        asset = f"https://images.pexels.com/photos/{key}/pexels-photo-{key}.jpeg?auto=compress&cs=tinysrgb&w=1000"
        yield key, page, asset, author, f"Pexels {lexical_key(author)} {key}", size, boxes, note, "https://www.pexels.com/license/"
    for key, slug, author, group, size, boxes, note in DVIDS:
        page = f"https://www.dvidshub.net/image/{key}/{slug}"
        asset = f"https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2609/{key}/1000w_q95.jpg"
        yield key, page, asset, author, group, size, boxes, note, "Public domain (DVIDS item page)"


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    history, private_hashes, private_pixels = protection()
    previous = [*(row for role in ("fit", "development", "evaluation") for row in cohorts(role)),
                *prior_native(), *prior_assembly(), *prior_recrop()]
    hashes = history["sha256"] | private_hashes | {r["sha256"] for r in previous}
    urls = history["urls"] | {url_key(r["source_url"]).rstrip("/") for r in previous if r.get("source_url")}
    groups = {r["group"] for r in previous}
    authors = history["authors"] | {lexical_key(r["author"]) for r in previous if r.get("author")}
    pixels = history["dhash"] | private_pixels
    for row in previous:
        pixels.update(raster_hashes(path_for(row)))
    rows, rejects = [], dict(VISUAL_EXCLUSIONS)
    for key, page, asset, author, group, grid, boxes, note, license in sources():
        path = WORK / "screen" / f"{key}.jpg"
        content, fingerprints = digest(path), raster_hashes(path)
        first, url = lexical_key(author), url_key(page).rstrip("/")
        if (key in history["ids"] or group in groups or content in hashes or url in urls
                or any(first in other for other in authors) or fingerprint_collision(fingerprints, pixels)):
            rejects[key] = "Identity, author, source URL, capture group or near-image exposure in project history."
            continue
        normalized = [[value / span for value, span in zip(box, grid * 2, strict=True)] for box in boxes]
        row = dict(id=f"refine-{key}", source_url=page, original_url=asset, author=author, group=group,
                   image=str(path), sha256=content, objects=[dict(bbox=b) for b in normalized],
                   review=note, license=license)
        image = raster(row)
        if abs(image.width / image.height - grid[0] / grid[1]) > .004 or any(not valid_box(b) for b in normalized):
            raise ValueError(f"Review grid and source image do not match for {key}.")
        rows.append(row)
        hashes.add(content)
        pixels.update(fingerprints)
        urls.add(url)
        groups.add(group)
        authors.add(first)
    if len(rows) < 4 or not any(len(r["objects"]) >= 2 for r in rows):
        raise ValueError(f"Not enough eligible independent instances and groups: {rejects}.")
    write_json(REFERENCE, dict(code=digest(__file__), rows=rows, exclusions=rejects,
                               scope="One visual reviewer. Distinct public captures, no output inspected before freeze."))
    print("admitted", len(rows), "targets", sum(len(r["objects"]) for r in rows), "excluded", rejects, flush=True)


def references():
    record = read_json(REFERENCE)
    if record["code"] != digest(__file__):
        raise ValueError("The reference protocol changed.")
    return record["rows"]


def binding():
    sealed()
    return dict(policy=digest(POLICY), reference=digest(REFERENCE), rf_checkpoint=read_json(INITIAL)["checkpoint_sha256"])


def state(path):
    key = binding()
    saved = read_json(path) if path.exists() else dict(binding=key, frames={})
    if saved["binding"] != key:
        raise ValueError(f"Stale evidence at {path}.")
    return saved


def baseline():
    from medsyringe.runtime.grounding_backend import analyze
    from medsyringe.runtime.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.experiments.expanded.sam3_study import configure_cpu

    cache = state(RELEASE)
    pending = [r for r in references() if r["id"] not in cache["frames"]]
    if not pending:
        return
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), "xpu")
        for row in pending:
            result = analyze(engine, raster(row), False)
            check = result["detection_checks"]
            if check["status"] != "available" or check["primary_backend"] != "xpu":
                raise ValueError(f"Released detector unavailable on {row['id']}.")
            cache["frames"][row["id"]] = result["objects"]
            write_json(RELEASE, cache)
            print(row["id"], "released count", len(result["objects"]), flush=True)
    finally:
        engine.close()


def collect():
    import torch
    from rfdetr import RFDETRNano

    cache = state(PROPOSALS)
    settings = read_json(INITIAL)
    if digest(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise ValueError("Unexpected RF-DETR checkpoint.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = RFDETRNano.from_checkpoint(settings["checkpoint"], device="cpu")
    for row in references():
        key = row["id"]
        if key in cache["frames"]:
            continue
        photo = raster(row)
        cache["frames"][key] = crop_proposals(model, photo, views(model, photo))
        write_json(PROPOSALS, cache)
        print(key, "proposals cached", flush=True)


def evidence():
    rows = references()
    inputs = {str(p): digest(p) for p in (REFERENCE, RELEASE, PROPOSALS)}
    base, proposals = (state(p)["frames"] for p in (RELEASE, PROPOSALS))
    if set(base) != set(proposals) or set(base) != {r["id"] for r in rows}:
        raise ValueError("Reserve evidence is incomplete.")
    return rows, base, proposals, inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "baseline", "collect"))
    {"freeze": freeze, "baseline": baseline, "collect": collect}[parser.parse_args().stage]()
