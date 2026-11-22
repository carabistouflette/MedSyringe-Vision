"""Lock new public references and collect fixed detector evidence for replacement."""
import argparse
from collections import Counter
from pathlib import Path

import cv2

from medsyringe.runtime.appearance_runtime import RELEASE as RUNTIME
from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, records, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision
from scripts.experiments.crowds.native_candidate_bank import views
from scripts.experiments.occlusion.occlusion_detector import ART, INITIAL, path_for, raster
from scripts.experiments.ranking.overlap_candidate_features import annotate_embedding, candidate_pool, inputs, load
from scripts.experiments.relocalization.pair_relocalization import POLICY, REFERENCE, WORK, frozen
from scripts.experiments.public_data.public_training_intake import raster_hashes
from scripts.experiments.relocalization.recrop_proposal_bank import crop_proposals
from scripts.experiments.transfer.rfdetr_verifier import DINOV2_INFO, DinoEncoder
from scripts.experiments.expanded.sam3_study import configure_cpu
from scripts.experiments.ranking.syringe_box_quality_reserve import encoded as prior_encoded, sources as prior_sources
from scripts.experiments.identity.syringe_object_references import DVIDS, PD, PEXELS, historical

SCREEN = WORK / "screen"
RF = WORK / "reserve-rf.json"
RELEASE = WORK / "reserve-release.json"
FEATURES = WORK / "features-reserve.json"
CACHE = "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos"
PEXELS_PAGE = "https://www.pexels.com/photo/"
PEXELS_IMAGE = "https://images.pexels.com/photos"

DVIDS_SOURCES = (
    ("4823468", "1810", "flu-vaccinations-uss-bonhomme-richard", "Daniel Charest", (544, 680),
     ((286, 214, 305, 262),), "One visible barrel partly hidden by fingers in a dim room."),
    ("8485748", "2406", "us-chad-continue-medical-partnership-garnison-military-hospital", "Anthony Hopper", (453, 680),
     ((89, 234, 117, 307),), "A narrow syringe with attached needle. Fingers conceal the plunger."),
    ("9873364", "2608", "army-reserve-medical-personnel-build-readiness-through-active-duty-joint-integration",
     "Jerry Zuetrong", (800, 534), (), "A pneumatic dental handpiece with tubing. No plunger syringe."),
    ("6368419", "2009", "saving-warfighter-bloodline", "Azaria Loyd", (800, 534),
     (), "Blood sample tubes in metal racks. No plunger syringe."),
    ("3356232", "1705", "28th-mdss-lab-asks-marrow-donors", "Nicolas Erwin", (800, 534),
     (), "Capped sample tubes and laboratory racks. No plunger syringe."),
)
PEXELS_SOURCES = (
    ("6074934", "set-of-medical-syringes-filled-with-medication-on-pink-desk", "Alena Shekhovtcova", (800, 534),
     ((47, 27, 215, 280), (267, 0, 375, 173), (197, 158, 460, 534), (491, 0, 745, 373), (541, 292, 700, 534)),
     "Five neighboring syringes. Two diagonal assemblies are cropped by the frame."),
    ("5973349", "close-up-shot-of-syringes-on-a-red-surface", "olia danilevich", (800, 534),
     ((125, 126, 386, 400), (274, 124, 523, 401), (431, 132, 679, 419)),
     "Three adjacent syringes with protective caps."),
)
EXCLUSIONS = {
    "8577983": "Multiple intersecting devices behind a container have uncertain visible extents.",
    "5863310": "The identity of the cropped assembly at the upper-right edge is uncertain.",
    "6847585": "Hand occlusion and distant devices prevent a consistent scene count.",
    "6627728": "Closed packages and drawer dividers prevent reliable instance attribution.",
    "9190822": "The same contributor appears in a previous reserve.",
    "6576166": "The same contributor appears in a previous reserve.",
    "8920974": "The same contributor appears in a previous reserve.",
    "32418225": "The same contributor appears in a previous reserve.",
    "7723202": "The same contributor appears in project history.",
    "9858478": "No clear syringe or useful syringe lookalike.",
    "wikimedia": "Candidate raster download returned HTTP 429.",
}


def source_rows():
    for key, month, slug, author, size, boxes, note in DVIDS_SOURCES:
        yield key, f"{DVIDS}{key}/{slug}", f"{CACHE}/{month}/{key}/1000w_q95.jpg", author, PD, size, boxes, note
    for key, slug, author, size, boxes, note in PEXELS_SOURCES:
        url = f"{PEXELS_IMAGE}/{key}/pexels-photo-{key}.jpeg?auto=compress&cs=tinysrgb&w=1000"
        yield key, f"{PEXELS_PAGE}{slug}-{key}/", url, author, PEXELS, size, boxes, note


def reference():
    packet = read_json(REFERENCE)
    if packet["protocol"] != digest(__file__):
        raise ValueError("The reserve protocol changed.")
    return packet["rows"]


def freeze_reference():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    known = historical()
    for file in sorted(ART.glob("*reference*.json")):
        for row in records(read_json(file)):
            if isinstance(row.get("author"), str):
                known["authors"].add(lexical_key(row["author"]))
            if isinstance(row.get("source_url"), str):
                known["urls"].add(url_key(row["source_url"]).rstrip("/"))
            if row.get("sha256"):
                known["hashes"].add(row["sha256"])
            if isinstance(row.get("image"), str) and path_for(row).is_file():
                known["pixels"].update(raster_hashes(path_for(row)))
    rows, excluded = [], dict(EXCLUSIONS)
    for key, page, asset, author, license, size, boxes, note in source_rows():
        file = SCREEN / f"{key}.jpg"
        who, url = lexical_key(author), url_key(page).rstrip("/")
        checksum, fingerprint = digest(file), raster_hashes(file)
        if (url in known["urls"] or checksum in known["hashes"] or
                any(who in name for name in known["authors"]) or fingerprint_collision(fingerprint, known["pixels"])):
            excluded[key] = "Historical URL, contributor, exact byte hash or visual match."
            continue
        coords = [[v / n for v,n in zip(b, size * 2, strict=True)] for b in boxes]
        row = dict(id=f"pair-reloc-{key}", source_url=page, original_url=asset, author=author, license=license,
                   group=f"pair-reloc:{who}", image=str(file), sha256=checksum, review=note,
                   objects=[dict(bbox=b) for b in coords])
        im = raster(row)
        if abs(im.width / im.height - size[0] / size[1]) > .004 or not all(map(valid_box, coords)):
            raise ValueError(f"The review coordinates do not match the raster for {key}.")
        rows.append(row)
        known["urls"].add(url)
        known["hashes"].add(checksum)
        known["authors"].add(who)
        known["pixels"].update(fingerprint)
    positive = Counter(bool(r["objects"]) for r in rows)
    if positive[True] < 3 or positive[False] < 2 or not any(len(r["objects"]) > 1 for r in rows):
        raise ValueError(f"Insufficient independent acquisitions after screening: {excluded}")
    write_json(REFERENCE, dict(protocol=digest(__file__), rows=rows, excluded=excluded,
                               note="One reviewer. Labels and inclusion decisions preceded detector inference."))
    print("Reserve", [(r["id"], len(r["objects"])) for r in rows], excluded, flush=True)


def state(path):
    frozen()
    binding = dict(policy=digest(POLICY), reference=digest(REFERENCE), runtime=digest(RUNTIME), rf=digest(INITIAL))
    packet = read_json(path) if path.exists() else dict(binding=binding, frames={})
    if packet["binding"] != binding:
        raise ValueError(f"Stale detector outputs {path}.")
    return packet


def collect_proposals():
    from rfdetr import RFDETRNano

    cfg, packet = read_json(INITIAL), state(RF)
    if digest(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
        raise ValueError("Wrong RF-DETR weights.")
    configure_cpu()
    model = RFDETRNano.from_checkpoint(cfg["checkpoint"], device="cpu")
    for row in reference():
        key = row["id"]
        if key not in packet["frames"]:
            im = raster(row)
            packet["frames"][key] = crop_proposals(model, im, views(model, im))
            write_json(RF, packet)
            print(key, "proposals collected", flush=True)


def collect_release():
    from medsyringe.runtime.grounding_backend import analyze
    from medsyringe.runtime.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine

    packet = state(RELEASE)
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), "xpu")
        for row in reference():
            key = row["id"]
            if key in packet["frames"]:
                continue
            result = analyze(engine, raster(row), False)
            checks = result["detection_checks"]
            if checks["status"] != "available" or checks["primary_backend"] != "xpu":
                raise ValueError(f"Released detector unavailable on XPU for {key}.")
            packet["frames"][key] = result["objects"]
            write_json(RELEASE, packet)
            print(key, len(result["objects"]), flush=True)
    finally:
        engine.close()


def reserve(encode=None):
    rows, base, raw = reference(), state(RELEASE)["frames"], state(RF)["frames"]
    keys = {r["id"] for r in rows}
    if keys != set(base) or keys != set(raw):
        raise ValueError("Missing released or RF-DETR frames.")
    binding = {str(p): digest(p) for p in (REFERENCE, POLICY, RELEASE, RF, DINOV2_INFO)}
    if FEATURES.is_file():
        saved = read_json(FEATURES)
        if saved["binding"] != binding or set(saved["frames"]) != keys:
            raise ValueError("Reserve embeddings changed.")
        return rows, base, raw, saved["frames"]
    if encode is None:
        raise FileNotFoundError(FEATURES)
    bank = {}
    for row in rows:
        key = row["id"]
        incumbent = [dict(bbox=p["bbox"], score=p["score"], corroboration=0., view="released") for p in base[key]]
        candidates = [*incumbent, *candidate_pool(raw[key])]
        observed = encode(raster(row), candidates)
        bank[key] = dict(base=observed[:len(incumbent)], proposed=observed[len(incumbent):])
        print(key, len(observed), "embedded", flush=True)
    write_json(FEATURES, dict(binding=binding, frames=bank))
    return rows, base, raw, bank


def collect_features():
    configure_cpu()
    cv2.setNumThreads(1)
    encoder = DinoEncoder()
    reserve(lambda image, boxes: annotate_embedding(encoder, image, boxes))


def packets():
    data = {"reserve": reserve()}
    for role in ("reserve", "exposed"):
        rows, base, bank = prior_encoded(role)
        raw = prior_sources(role)[2]
        data[f"box-quality-{role}"] = rows, base, raw, bank
    for role in ("fresh", "replay"):
        rows, base, raw = inputs(role)
        data[f"earlier-{role}"] = rows, base, raw, load(role, rows, base, raw)
    return data


if __name__ == "__main__":
    stages = dict(reference=freeze_reference, proposals=collect_proposals, release=collect_release, features=collect_features)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=stages)
    stages[cli.parse_args().stage]()
