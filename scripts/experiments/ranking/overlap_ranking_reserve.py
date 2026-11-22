"""Freeze public acquisition groups for the box-ranking and conflict comparison."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, records, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision, protection
from scripts.experiments.crowds.native_candidate_bank import cohorts, views
from scripts.experiments.occlusion.occlusion_detector import INITIAL, path_for, raster
from scripts.experiments.ranking.overlap_ranking import ART, LOCK, WORK, sealed
from scripts.experiments.public_data.public_training_intake import raster_hashes
from scripts.experiments.relocalization.recrop_proposal_bank import crop_proposals

REFERENCE = ART / "overlap_ranking_reserve.json"
PROPOSALS = WORK / "fresh-rf.json"
RELEASE = WORK / "fresh-release.json"
COMMONS = "https://commons.wikimedia.org/wiki/File:"
FILES = "https://upload.wikimedia.org/wikipedia/commons/"
CC = "https://creativecommons.org/licenses/by-sa/4.0/"
PD = "Public domain per DVIDS item record"
SOURCES = (
    ("6484814", "https://www.dvidshub.net/image/6484814/camp-arifjan-administers-covid-19-vaccinations",
     "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2101/6484814/1000w_q95.jpg",
     "Noel Gerig", PD, (1000, 667), ((592, 269, 646, 427), (739, 181, 794, 293)),
     "Two syringes held by separate technicians. The right syringe is softly focused. Vials excluded."),
    ("2940421", "https://www.dvidshub.net/image/2940421/vaccination-exercise-trains-medics",
     "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/1610/2940421/1000w_q95.jpg",
     "Jessica Montano (Jessica Weissman in original caption)", PD, (1000, 668),
     ((473, 71, 679, 215),), "One syringe partly hidden by fingers. The cap and needle are visible."),
    ("labpipette", COMMONS + "Pipette_in_lab.jpg", FILES + "2/24/Pipette_in_lab.jpg",
     "Akshaya govindasamy", CC, (495, 660), (), "A graduated glass pipette. No syringe."),
    ("nci", COMMONS + "Pipetting.jpg", FILES + "5/55/Pipetting.jpg",
     "Diane A. Reid", "Public domain per National Cancer Institute and Wikimedia Commons",
     (760, 660), (), "Hand-held mechanical pipette and flask. No syringe."),
    ("pipettes", COMMONS + "Pipettes.jpg", FILES + "5/58/Pipettes.jpg",
     "Steinsky and James Frost", "Public domain original and GFDL subsequent edit per Wikimedia Commons",
     (760, 407), (), "Numbered illustration photograph of multiple types of pipettes. No syringe."),
    ("bench", COMMONS + "Laboratory_pipettes.jpg", FILES + "5/5a/Laboratory_pipettes.jpg",
     "J.N. Eskra", CC, (760, 507), (), "Rack of adjustable pipettes. No syringe."),
)
EXCLUSIONS = {
    "annapolis": "Closely spaced, partly occluded barrels. Some distinct seals and visible box endpoints could not be assigned reliably.",
    "6484816": "Rear rows of injectors have blurred and hidden endpoints. Same Camp Arifjan acquisition as 6484814.",
    "6484815": "Same Camp Arifjan acquisition as 6484814.",
    "6484818": "Same Camp Arifjan acquisition as 6484814.",
    "farnsworth": "Contributor and related acquisition in earlier public experiments.",
    "baypines": "Earlier review rejected the same published image for unresolved shallow depth of field.",
}
ALIAS = {
    "labpipette": ("Akshaya govindasamy", "Akshaya Govindasamy"),
    "nci": ("Diane A. Reid", "Diane Reid"),
    "pipettes": ("Steinsky", "James Frost", "Tamudawson"),
    "2940421": ("Jessica Montano", "Jessica Weissman"),
}


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    h, private, private_pixels = protection()
    past = [r for role in ("fit", "development", "evaluation") for r in cohorts(role)]
    for source in sorted(ART.glob("*reserve.json")):
        past.extend(records(read_json(source)))
    hashes = h["sha256"] | private | {r["sha256"] for r in past if r.get("sha256")}
    pixels = h["dhash"] | private_pixels
    history = {path_for(r) for r in past if r.get("image")}
    for p in sorted(history):
        if p.exists():
            pixels.update(raster_hashes(p))
    authors = h["authors"] | {lexical_key(r["author"]) for r in past if r.get("author")}
    urls = h["urls"] | {url_key(r["source_url"]).rstrip("/") for r in past if r.get("source_url")}
    groups = {r["group"] for r in past if r.get("group")}
    rows, excluded = [], dict(EXCLUSIONS)
    for key, page, asset, author, license, preview, boxes, review in SOURCES:
        p = WORK / f"{key}.jpg"
        who = lexical_key(author)
        url, content, visual = url_key(page).rstrip("/"), digest(p), raster_hashes(p)
        group = f"overlap-ranking:{who}:{key}"
        known = {lexical_key(v) for v in ALIAS.get(key, (author,))}
        if (any(alias in a for alias in known for a in authors) or url in urls or content in hashes or group in groups
                or fingerprint_collision(visual, pixels)):
            excluded[key] = "Existing source, contributor, acquisition or image fingerprint."
            continue
        objects = [dict(bbox=[z/n for z, n in zip(b, preview*2, strict=True)]) for b in boxes]
        row = dict(id=f"overlap-{key}", image=str(p), source_url=page, original_url=asset,
                   author=author, group=group, license=license, objects=objects, review=review, sha256=content)
        im = raster(row)
        if abs(im.width/im.height - preview[0]/preview[1]) > .004 or any(not valid_box(o["bbox"]) for o in objects):
            raise ValueError(f"Preview coordinates do not match {key}.")
        rows.append(row)
        hashes.add(content)
        pixels.update(visual)
        urls.add(url)
        authors.add(who)
    if sum(bool(r["objects"]) for r in rows) < 2 or not any(len(r["objects"]) > 1 for r in rows) or (
        sum(not r["objects"] for r in rows) < 2
    ):
        raise ValueError(f"Insufficient positive and negative acquisition coverage: {excluded}")
    write_json(REFERENCE, dict(code=digest(__file__), rows=rows, excluded=excluded,
                               notes="Single visual reviewer, no predictions consulted before annotation."))
    print("Admitted", [(r["id"], len(r["objects"])) for r in rows], flush=True)


def references():
    ref = read_json(REFERENCE)
    if ref["code"] != digest(__file__):
        raise ValueError("The holdout protocol changed.")
    return ref["rows"]


def state(path):
    sealed()
    binding = dict(reference=digest(REFERENCE), policy=digest(LOCK),
                   rf=read_json(INITIAL)["checkpoint_sha256"])
    state = read_json(path) if path.exists() else dict(binding=binding, frames={})
    if state["binding"] != binding:
        raise ValueError(f"Wrong acquisition or policy evidence in {path}.")
    return state


def collect():
    import torch
    from rfdetr import RFDETRNano

    cache = state(PROPOSALS)
    cfg = read_json(INITIAL)
    if digest(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
        raise ValueError("Unexpected RF-DETR checkpoint.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = RFDETRNano.from_checkpoint(cfg["checkpoint"], device="cpu")
    for row in references():
        k = row["id"]
        if k not in cache["frames"]:
            image = raster(row)
            cache["frames"][k] = crop_proposals(model, image, views(model, image))
            write_json(PROPOSALS, cache)
            print(k, "collected", flush=True)


def baseline():
    from medsyringe.runtime.grounding_backend import analyze
    from medsyringe.runtime.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.experiments.expanded.sam3_study import configure_cpu

    cache = state(RELEASE)
    configure_cpu()
    with_engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(with_engine, Path.cwd(), "xpu")
        for row in references():
            k = row["id"]
            if k not in cache["frames"]:
                result = analyze(with_engine, raster(row), False)
                check = result["detection_checks"]
                if check["status"] != "available" or check["primary_backend"] != "xpu":
                    raise ValueError("Released detector unavailable.")
                cache["frames"][k] = result["objects"]
                write_json(RELEASE, cache)
                print(k, len(result["objects"]), flush=True)
    finally:
        with_engine.close()


def frozen_inputs():
    rows, frames = references(), state(PROPOSALS)["frames"]
    if set(frames) != {r["id"] for r in rows}:
        raise ValueError("Missing holdout RF-DETR proposals.")
    return rows, frames


def evidence():
    rows, frames = frozen_inputs()
    released = state(RELEASE)["frames"]
    if set(released) != set(frames):
        raise ValueError("Missing released holdout outputs.")
    return rows, released, frames, {str(p): digest(p) for p in (REFERENCE, RELEASE, PROPOSALS)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "collect", "baseline"))
    {"freeze": freeze, "collect": collect, "baseline": baseline}[cli.parse_args().stage]()
