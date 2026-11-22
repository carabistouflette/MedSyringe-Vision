"""Freeze new public evaluators and collect identical RF-DETR views for all study arms."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.detection_public_benchmark import lexical_key, records, url_key
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.native_candidate_bank import cohorts, views
from scripts.occlusion_detector import INITIAL, path_for, raster
from scripts.part_association import ART, LOCK, WORK, sealed
from scripts.public_training_intake import raster_hashes
from scripts.recrop_proposal_bank import crop_proposals

REFERENCE = ART / "part_association_reserve.json"
PROPOSALS = WORK / "fresh-rf.json"
RELEASE = WORK / "fresh-release.json"
COMMONS = "https://commons.wikimedia.org/wiki/File:"
FILES = "https://upload.wikimedia.org/wikipedia/commons/"
CC = "https://creativecommons.org/licenses/by-sa/4.0/"
PD = "Public domain per DVIDS item record"
PEXELS = "https://www.pexels.com/license/"
SOURCES = (
    ("aviano", "https://www.dvidshub.net/image/5841408/flu-shots-aviano",
     "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/1910/5841408/1000w_q95.jpg",
     "Ericka Woolever", PD, (850, 567),
     ((73, 330, 177, 539), (176, 311, 308, 452), (273, 306, 415, 430),
      (307, 368, 454, 567), (467, 341, 586, 512), (533, 334, 697, 468),
      (327, 210, 700, 379), (674, 332, 850, 455)),
     "Eight foreground assemblies traceable by their own barrel and distal or rear part. "
     "Far-background blur has insufficient identity evidence. Bounds are approximate."),
    ("anesth", COMMONS + "Anestezeye_schinrece_berbis_sirinke.jpg",
     FILES + "c/c4/Anestezeye_schinrece_berbis_sirinke.jpg", "Lucyin", CC, (431, 650),
     ((225, 260, 365, 312),), "One syringe beside wool. The distal portion is partly occluded."),
    ("give", COMMONS + "Verschenken_001_2006_10_26.jpg",
     FILES + "7/7b/Verschenken_001_2006_10_26.jpg", "Friedrich Haag", CC, (650, 440),
     ((99, 0, 221, 413),), "One oversized syringe beside a floral arrangement. The plunger is cropped."),
    ("bench", COMMONS + "Laboratory_pipettes.jpg", FILES + "5/5a/Laboratory_pipettes.jpg",
     "J.N. Eskra", CC, (650, 433), (), "Rack of adjustable pipettes. No syringe."),
    ("serological", COMMONS + "Pipetter_and_serological_pipette.jpg",
     FILES + "e/e4/Pipetter_and_serological_pipette.jpg", "MicrobiologyMarcus",
     "https://creativecommons.org/publicdomain/zero/1.0/", (650, 488), (),
     "Pipette and controller on a support. No syringe."),
    ("pipette", "https://www.pexels.com/photo/scientist-conducting-laboratory-experiment-with-pipette-38145554/",
     "https://images.pexels.com/photos/38145554/pexels-photo-38145554.jpeg?auto=compress&cs=tinysrgb&w=1000",
     "Maikol Herrera", PEXELS, (650, 434), (), "Held micropipette, beakers and other pipettes."),
    ("23655818", "https://www.pexels.com/photo/needles-and-syringe-in-hospital-23655818/",
     "https://images.pexels.com/photos/23655818/pexels-photo-23655818.jpeg?auto=compress&cs=tinysrgb&w=1800",
     "Marta Branco", PEXELS, (850, 567), (),
     "Visible objects are loose needles, connectors, caps and IV tubing. No syringe barrel and plunger assembly."),
)
EXCLUSIONS = {
    "32418225": "At least two foreground syringes. One packaged device cannot be labeled with confidence.",
    "19677805": "Several front-facing plungers cannot be associated unambiguously with visible bodies.",
    "guard-000": "The Illinois National Guard photo has strongly blurred devices whose extents cannot be reviewed.",
    "three": "Known contributor Bron766.",
    "set": "Known contributor Nadine90 and perceptual overlap.",
    "fish": "Perceptual overlap.",
    "glass": "Known contributor Kuebi and perceptual overlap.",
    "light": "Known Wikimedia Commons source and perceptual overlap.",
    "minot": "The documented public asset was inaccessible.",
    "beale": "The documented public asset was inaccessible.",
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
        group = f"part-association:{who}:{key}"
        if (any(who in a for a in authors) or url in urls or content in hashes or group in groups
                or fingerprint_collision(visual, pixels)):
            excluded[key] = "Existing source, contributor, acquisition or image fingerprint."
            continue
        objects = [dict(bbox=[z/n for z, n in zip(b, preview*2, strict=True)]) for b in boxes]
        row = dict(id=f"association-{key}", image=str(p), source_url=page, original_url=asset,
                   author=author, group=group, license=license, objects=objects, review=review, sha256=content)
        im = raster(row)
        if abs(im.width/im.height - preview[0]/preview[1]) > .004 or any(not valid_box(o["bbox"]) for o in objects):
            raise ValueError(f"Preview coordinates do not match {key}.")
        rows.append(row)
        hashes.add(content)
        pixels.update(visual)
        urls.add(url)
        authors.add(who)
    if len(rows) < 4 or not any(len(r["objects"]) > 1 for r in rows) or sum(not r["objects"] for r in rows) < 2:
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
    from medsyringe.grounding_backend import analyze
    from medsyringe.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.sam3_study import configure_cpu

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
