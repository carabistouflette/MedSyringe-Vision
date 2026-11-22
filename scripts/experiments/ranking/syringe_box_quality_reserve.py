"""Annotate and infer on public acquisitions reserved for box quality transfer."""
import argparse
from collections import Counter
from pathlib import Path

import cv2

from medsyringe.runtime.appearance_runtime import RELEASE as RUNTIME
from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision
from scripts.experiments.crowds.native_candidate_bank import views
from scripts.experiments.occlusion.occlusion_detector import INITIAL, path_for, raster
from scripts.experiments.ranking.overlap_candidate_features import annotate_embedding, candidate_pool, inputs, load
from scripts.experiments.public_data.public_training_intake import raster_hashes
from scripts.experiments.relocalization.recrop_proposal_bank import crop_proposals
from scripts.experiments.transfer.rfdetr_verifier import DINOV2_INFO, DinoEncoder
from scripts.experiments.expanded.sam3_study import configure_cpu
from scripts.experiments.ranking.syringe_box_quality import POLICY, REFERENCE, WORK, frozen
from scripts.experiments.identity.syringe_object_references import DVIDS, PD, historical, references
from scripts.experiments.identity.syringe_object_reserve import evidence as prior_evidence

RF = WORK / "reserve-rf.json"
RELEASED = WORK / "reserve-release.json"
SCREEN = WORK / "screen"

# Pixel coordinates in the reviewed preview. Include visible needle or cap and exclude vials.
SOURCES = (
    ("9970087", "2609", "soldiers-sea-jacks-all-trades", "Olivia Bithell",
     (800, 533), ((369, 286, 479, 323),), "Short visible barrel between black gloves."),
    ("6586276", "2104", "146th-medical-group-shines-rolls-out-dependent-vaccinations-its-first-week", "Nicholas Carzis",
     (800, 534), ((575, 129, 606, 290),), "One thin barrel with attached needle, partly held by fingers."),
    ("9642144", "2604", "combined-resolve-26-07", "Danil Pak",
     (800, 534), ((351, 154, 381, 230),), "Thin barrel between two gloves. Exclude hanging cables."),
    ("10108059", "2611", "uss-boxer-11th-meu-conduct-medical-training-and-maintenance", "Tyler Miles",
     (800, 640), ((514, 244, 541, 405),), "Upper and lower ends of the same assembly separated by hand occlusion."),
    ("9454977", "2512", "uss-carl-vinson-cvn-70-medical-operations", "Elyssia Rodriguez",
     (800, 533), ((327, 232, 370, 390),), "Graduated barrel with two partly occluded ends. Vial excluded."),
    ("8829491", "2501", "379th-emds-performs-tonsillectomy", "Robert Nichols",
     (800, 534), ((337, 110, 404, 534),), "One cropped syringe against hemostats and forceps."),
    ("8071173", "2310", "nmcb-133-deployed-rota-spain", "Andrew Waters",
     (453, 680), ((276, 227, 314, 294),), "A thin syringe with an orange hub. Vial and glove excluded."),
    ("10013917", "2610", "behind-diagnosis-inside-dgmcs-medical-laboratory", "Gary Edwards",
     (800, 600), (), "Laboratory dropper and racks. No visible syringe."),
    ("9974802", "2609", "expeditionary-blood-bank-trusted-care-heroes", "Alexandre Montes",
     (800, 534), (), "Three capped specimen tubes held in one hand. No visible syringe."),
    ("3596154", "1707", "wright-patt-hematology-laboratory-study-human-blood-health-and-disease", "Michelle Gigante",
     (800, 640), (), "Tube handling in a rack. No visible syringe."),
    ("8989546", "2504", "509th-hcos-medical-laboratory", "Joshua Hastings",
     (800, 534), (), "Rack with four red-cap specimens. No visible syringe."),
    ("8169009", "2312", "usamriid-blood-testing-protocol-will-improve-disease-surveillance", "Samuel Mensah",
     (800, 571), (), "Laboratory workbench and mechanical pipettes. No visible syringe."),
)
EXCLUSIONS = {
    "8811603": "The operator's fingers conceal most of the syringe.",
    "9899702": "The visible device extent is indeterminate in the dark field image.",
    "9724193": "The small device and the visible extent of surrounding injectors are uncertain.",
    "6672204": "Fingers conceal the foreground injector.",
    "9261559": "Known prior acquisition.",
    "9468367": "Known prior acquisition.",
    "8071796": "Known prior acquisition.",
    "9218426": "No clearly visible syringe-like object.",
}


def reference():
    data = read_json(REFERENCE)
    if data["protocol"] != digest(__file__):
        raise ValueError("The reserve annotation protocol changed.")
    return data["rows"]


def freeze_reference():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    known = historical()
    for row in [*references("fit"), *references("fresh")]:
        known["hashes"].add(row["sha256"])
        known["pixels"].update(raster_hashes(path_for(row)))
        known["authors"].add(lexical_key(row["author"]))
        known["urls"].add(url_key(row["source_url"]).rstrip("/"))
    rows, exclusions = [], dict(EXCLUSIONS)
    for key, month, slug, author, dimensions, boxes, note in SOURCES:
        path = SCREEN / f"{key}.jpg"
        checksum, visual, who = digest(path), raster_hashes(path), lexical_key(author)
        source = f"{DVIDS}{key}/{slug}"
        url = url_key(source).rstrip("/")
        if (checksum in known["hashes"] or url in known["urls"] or
                any(who in a for a in known["authors"]) or fingerprint_collision(visual, known["pixels"])):
            exclusions[key] = "Historical contributor, URL, exact file or close visual fingerprint."
            continue
        coords = [[px / size for px, size in zip(box, dimensions * 2, strict=True)] for box in boxes]
        row = dict(id=f"box-quality-{key}", author=author, source_url=source, image=str(path),
                   original_url=f"https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/{month}/{key}/1000w_q95.jpg",
                   license=PD, sha256=checksum, group=f"box-quality:{who}", review=note,
                   objects=[dict(bbox=box) for box in coords])
        im = raster(row)
        if abs(im.width / im.height - dimensions[0] / dimensions[1]) > .004 or not all(map(valid_box, coords)):
            raise ValueError(f"Wrong review geometry for {key}.")
        rows.append(row)
        known["hashes"].add(checksum)
        known["pixels"].update(visual)
        known["authors"].add(who)
        known["urls"].add(url)
    support = Counter(bool(r["objects"]) for r in rows)
    if support[True] < 4 or support[False] < 2:
        raise ValueError(f"Insufficient reserve coverage after deduplication: {exclusions}")
    write_json(REFERENCE, dict(protocol=digest(__file__), rows=rows, exclusions=exclusions,
                               note="One reviewer, annotated before either detector was run."))
    print("Reserve", [(r["id"], len(r["objects"])) for r in rows], flush=True)


def state(path):
    frozen()
    binding = dict(policy=digest(POLICY), reference=digest(REFERENCE),
                   runtime=digest(RUNTIME), rf=digest(INITIAL))
    item = read_json(path) if path.exists() else dict(binding=binding, frames={})
    if item["binding"] != binding:
        raise ValueError(f"Stale evidence in {path}.")
    return item


def collect_rf():
    from rfdetr import RFDETRNano

    cfg, archive = read_json(INITIAL), state(RF)
    if digest(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
        raise ValueError("Wrong RF-DETR checkpoint.")
    configure_cpu()
    model = RFDETRNano.from_checkpoint(cfg["checkpoint"], device="cpu")
    for row in reference():
        key = row["id"]
        if key not in archive["frames"]:
            photo = raster(row)
            archive["frames"][key] = crop_proposals(model, photo, views(model, photo))
            write_json(RF, archive)
            print(key, "proposals collected", flush=True)


def collect_release():
    from medsyringe.runtime.grounding_backend import analyze
    from medsyringe.runtime.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine

    archive = state(RELEASED)
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), "xpu")
        for row in reference():
            key = row["id"]
            if key in archive["frames"]:
                continue
            result = analyze(engine, raster(row), False)
            status = result["detection_checks"]
            if status["status"] != "available" or status["primary_backend"] != "xpu":
                raise ValueError(f"Release inference did not complete on XPU for {key}.")
            archive["frames"][key] = result["objects"]
            write_json(RELEASED, archive)
            print(key, len(result["objects"]), flush=True)
    finally:
        engine.close()


def sources(role):
    if role == "exposed":
        return prior_evidence()
    paths = (REFERENCE, RELEASED, RF)
    return reference(), state(RELEASED)["frames"], state(RF)["frames"], {str(p): digest(p) for p in paths}


def encoded(role, encode=None):
    rows, base, raw, source = sources(role)
    keys = {r["id"] for r in rows}
    if keys != set(base) or keys != set(raw):
        raise ValueError(f"Missing frame outputs for {role}.")
    binding = dict(source=source, protocol=digest(__file__), dino=digest(DINOV2_INFO))
    path = WORK / f"features-{role}.json"
    if path.exists():
        cached = read_json(path)
        if cached["binding"] != binding or keys != set(cached["frames"]):
            raise ValueError(f"Unexpected cache evidence in {path}.")
        return rows, base, cached["frames"]
    if encode is None:
        raise FileNotFoundError(path)
    bank = {}
    for row in rows:
        key = row["id"]
        incumbent = [dict(bbox=p["bbox"], score=p["score"], corroboration=0., view="released") for p in base[key]]
        detected = encode(raster(row), [*incumbent, *candidate_pool(raw[key])])
        bank[key] = dict(base=detected[:len(incumbent)], proposed=detected[len(incumbent):])
        print(role, key, len(detected), flush=True)
    write_json(path, dict(binding=binding, frames=bank))
    return rows, base, bank


def collect_features():
    configure_cpu()
    cv2.setNumThreads(1)
    encoder = DinoEncoder()
    for role in ("reserve", "exposed"):
        encoded(role, lambda photo, boxes: annotate_embedding(encoder, photo, boxes))


def packets():
    packets = {role: encoded(role) for role in ("reserve", "exposed")}
    for role in ("fresh", "replay"):
        rows, base, raw = inputs(role)
        packets[f"earlier-{role}"] = rows, base, load(role, rows, base, raw)
    return packets


if __name__ == "__main__":
    stages = dict(reference=freeze_reference, proposals=collect_rf, release=collect_release, features=collect_features)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=stages)
    stages[cli.parse_args().stage]()
