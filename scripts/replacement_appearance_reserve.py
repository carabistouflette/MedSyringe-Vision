"""Review independent syringe and lookalike photos before collecting detector evidence."""
import argparse
from collections import Counter
from pathlib import Path

import cv2

from medsyringe.appearance_runtime import RELEASE as RUNTIME
from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.detection_public_benchmark import lexical_key, records, url_key
from scripts.instance_source_audit import fingerprint_collision
from scripts.native_candidate_bank import views
from scripts.occlusion_detector import ART, INITIAL, path_for, raster
from scripts.overlap_candidate_features import WORK as OVERLAP, annotate_embedding, candidate_pool, inputs, load
from scripts.pair_relocalization import WORK as RELOCALIZATION
from scripts.pair_relocalization_reserve import packets as older_packets
from scripts.public_training_intake import raster_hashes
from scripts.recrop_proposal_bank import crop_proposals
from scripts.replacement_appearance import POLICY, REFERENCE, VERIFIER, WORK, frozen
from scripts.rfdetr_verifier import DINOV2_INFO, DinoEncoder
from scripts.sam3_study import configure_cpu
from scripts.syringe_box_quality import WORK as QUALITY
from scripts.syringe_object_references import DVIDS, PD, PEXELS, historical

SCREEN = WORK / "screen"
RF = WORK / "reserve-rf.json"
RELEASE = WORK / "reserve-release.json"
FEATURES = WORK / "reserve-dino.json"
PEXELS_PAGE = "https://www.pexels.com/photo/"
PEXELS_IMAGE = "https://images.pexels.com/photos"
MEDIA = "https://d1ldvf68ux039x.cloudfront.net/thumbs/photos"
DVIDS_SOURCES = (
    ("6168264", "2004", "usns-mercy-sailor-prepares-syringe", "Ryan Breeden", (800, 571),
     ((278, 286, 423, 379),), ()),
)
PEXELS_SOURCES = (
    ("37807023", "close-up-of-black-gloved-hand-holding-syringe", "Laura Beauty Designer | Brasil", (453, 680),
     ((209, 198, 270, 424),), ()),
    ("5857419", "person-holding-a-syringe", "Anna Tarazevich", (453, 680), ((296, 32, 380, 500),), ()),
    ("5863376", "person-injecting-himself-with-covid-19-vaccine", "Nataliya Vaitkevich", (800, 534),
     ((350, 266, 631, 309),), ()),
    ("4299424", "anonymous-lab-worker-using-pipette-for-injecting-sample-into-flask", "Jorge Sepúlveda", (800, 534),
     (), (("pipette", (0, 294, 417, 409)), ("tube", (331, 210, 468, 354)))),
    ("4031442", "person-using-a-pipette", "Edward Jenner", (800, 534),
     (), (("pipette", (373, 232, 800, 517)),)),
    ("10188009", "woman-hand-making-chemistry-experiment", "Ron Lach", (800, 534),
     (), (("pipette", (405, 123, 536, 298)), ("pipette", (648, 402, 734, 534)))),
    ("8940521", "a-scientist-holding-a-dropper", "Thirdman", (453, 680),
     (), (("pipette", (176, 27, 252, 358)),)),
    ("3912481", "scientist-in-laboratory", "ThisIsEngineering", (800, 534),
     (), (("pipette", (292, 54, 333, 306)), ("tube", (294, 324, 329, 382)))),
    ("17515220", "woman-in-a-lab-coat-holding-a-test-tube", "Diego Romero", (453, 680),
     (), (("pipette", (112, 166, 259, 343)),)),
    ("8325942", "woman-in-white-coat-smiling-while-doing-an-experiment", "Kindel Media", (510, 680),
     (), (("pipette", (0, 211, 226, 329)),)),
)
EXCLUSIONS = {
    "36675185": "Same credited photographer as 37807023.",
    "9285771": "Fingers and soft focus obscure the visible syringe extent.",
    "9256097": "The identity and extent of syringes on the rear tray are ambiguous.",
    "525496": "Gloves conceal the limits of the syringe.",
    "8711851": "Uncertain syringe extents on the rear table.",
    "8770735": "Motion blur obscures the pipette tip.",
    "37499105": "Earlier project source.",
}


def source_rows():
    for key, date, slug, author, size, positive, negative in DVIDS_SOURCES:
        yield key, f"{DVIDS}{key}/{slug}", f"{MEDIA}/{date}/{key}/1000w_q95.jpg", author, PD, size, positive, negative
    for key, slug, author, size, positive, negative in PEXELS_SOURCES:
        asset = f"{PEXELS_IMAGE}/{key}/pexels-photo-{key}.jpeg?auto=compress&cs=tinysrgb&w=1000"
        yield key, f"{PEXELS_PAGE}{slug}-{key}/", asset, author, PEXELS, size, positive, negative


def references():
    p = read_json(REFERENCE)
    if p["protocol"] != digest(__file__):
        raise ValueError("The reference protocol changed.")
    return p["rows"]


def freeze_reference():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    known = historical()
    for file in sorted(ART.glob("*reference*.json")):
        for r in records(read_json(file)):
            if isinstance(r.get("author"),str):
                known["authors"].add(lexical_key(r["author"]))
            if isinstance(r.get("source_url"),str):
                known["urls"].add(url_key(r["source_url"]).rstrip("/"))
            if r.get("sha256"):
                known["hashes"].add(r["sha256"])
            if isinstance(r.get("image"),str) and path_for(r).is_file():
                known["pixels"].update(raster_hashes(path_for(r)))
    rows, excluded = [], dict(EXCLUSIONS)
    for key, page, asset, author, license, shape, positive, negative in source_rows():
        file = SCREEN / f"{key}.jpg"
        who, url = lexical_key(author), url_key(page).rstrip("/")
        checksum, pixels = digest(file), raster_hashes(file)
        if (checksum in known["hashes"] or url in known["urls"] or
                any(who in a for a in known["authors"]) or fingerprint_collision(pixels, known["pixels"])):
            excluded[key] = "Known contributor, URL, exact bytes or close image fingerprint."
            continue

        def convert(coords):
            b = [v/n for v,n in zip(coords, shape * 2, strict=True)]
            if not valid_box(b):
                raise ValueError(f"Bad reference geometry {key}.")
            return b

        row = dict(id=f"presence-{key}", author=author, source_url=page, original_url=asset, license=license,
                   group=f"presence:{who}", image=str(file), sha256=checksum,
                   objects=[dict(bbox=convert(b)) for b in positive],
                   lookalikes=[dict(kind=t,bbox=convert(b)) for t,b in negative])
        im = raster(row)
        if abs(im.width / im.height - shape[0]/shape[1]) > .004:
            raise ValueError(f"Scale mismatch for {key}.")
        rows.append(row)
        known["hashes"].add(checksum)
        known["pixels"].update(pixels)
        known["authors"].add(who)
        known["urls"].add(url)
    support = Counter(bool(r["objects"]) for r in rows)
    if support[True] < 2 or support[False] < 2:
        raise ValueError(f"Insufficient acquisition coverage: {excluded}.")
    write_json(REFERENCE, dict(rows=rows, protocol=digest(__file__), exclusions=excluded,
                               note="One reviewer. Syringe and lookalike boxes were fixed before detector inference."))
    print("Reserve",[(r["id"],len(r["objects"]),len(r["lookalikes"])) for r in rows],excluded,flush=True)


def state(path):
    frozen()
    files = (POLICY, REFERENCE, RUNTIME, INITIAL)
    binding = {str(p):digest(p) for p in files}
    data = read_json(path) if path.is_file() else dict(binding=binding, frames={})
    if data["binding"] != binding:
        raise ValueError(f"Stale detector evidence in {path}.")
    return data


def collect_rf():
    from rfdetr import RFDETRNano

    data, config = state(RF), read_json(INITIAL)
    if digest(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Unexpected RF-DETR weights.")
    configure_cpu()
    model = RFDETRNano.from_checkpoint(config["checkpoint"], device="cpu")
    for row in references():
        key = row["id"]
        if key not in data["frames"]:
            image = raster(row)
            data["frames"][key] = crop_proposals(model, image, views(model, image))
            write_json(RF, data)
            print(key, "RF-DETR", flush=True)


def collect_release():
    from medsyringe.grounding_backend import analyze
    from medsyringe.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine

    data = state(RELEASE)
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), "xpu")
        for row in references():
            key = row["id"]
            if key not in data["frames"]:
                result = analyze(engine, raster(row), False)
                check = result["detection_checks"]
                if check["status"] != "available" or check["primary_backend"] != "xpu":
                    raise ValueError(f"Missing released XPU detection for {key}.")
                data["frames"][key] = result["objects"]
                write_json(RELEASE, data)
                print(key,len(result["objects"]),flush=True)
    finally:
        engine.close()


def capture():
    rows, base, rf = references(), state(RELEASE)["frames"], state(RF)["frames"]
    keys = {r["id"] for r in rows}
    if set(base) != keys or set(rf) != keys:
        raise ValueError("The reserve detector evidence is incomplete.")
    return rows, base, rf


def collect_dino():
    rows, base, rf = capture()
    configure_cpu()
    cv2.setNumThreads(1)
    encoder = DinoEncoder()
    bank = {}
    for r in rows:
        key = r["id"]
        previous = [dict(bbox=p["bbox"], score=p["score"], view="released", corroboration=0.) for p in base[key]]
        proposed = candidate_pool(rf[key])
        features = annotate_embedding(encoder, raster(r), [*previous,*proposed])
        bank[key] = dict(base=features[:len(previous)], proposed=features[len(previous):])
        print(key, len(features), "embedded", flush=True)
    write_json(FEATURES, dict(binding={str(p):digest(p) for p in (RF, RELEASE, REFERENCE, DINOV2_INFO)},
                              frames=bank))


def batches(roles):
    for role in roles:
        if role == "development":
            rows, base, raw = inputs(role)
            yield role, (rows,base,load(role,rows,base,raw),(OVERLAP/"features-development.json",))
        elif role == "reserve":
            rows, base, _ = capture()
            features = read_json(FEATURES)
            files = (RF,RELEASE,REFERENCE,DINOV2_INFO)
            if features["binding"] != {str(p):digest(p) for p in files}:
                raise ValueError("Mismatched reserve embeddings.")
            yield role, (rows,base,features["frames"],(*files,FEATURES))
        else:
            older = older_packets()
            labels = ("reserve", "box-quality-reserve", "box-quality-exposed", "earlier-replay")
            rows, base, bank = [], {}, {}
            for label in labels:
                refs, old, _, observed = older[label]
                if set(old)&set(base):
                    raise ValueError("Exposed cohorts overlap by identifier.")
                rows.extend(refs)
                base.update(old)
                bank.update(observed)
            files = (RELOCALIZATION/"features-reserve.json",QUALITY/"features-reserve.json",
                     QUALITY/"features-exposed.json",OVERLAP/"features-replay.json")
            yield role, (rows,base,bank,files)


def signature(files):
    return {str(p):digest(p) for p in (*files, VERIFIER, RUNTIME, Path("scripts/replacement_appearance.py"))}


def store(role, packet, files):
    path = WORK / f"semantic-{role}.json"
    if path.is_file():
        raise FileExistsError(path)
    write_json(path, dict(binding=signature(files), packet=packet))


def cohorts():
    for role, (rows, _, _, files) in batches(("reserve","exposed")):
        cached = read_json(WORK/f"semantic-{role}.json")
        if cached["binding"] != signature(files) or set(cached["packet"]["base"]) != {r["id"] for r in rows}:
            raise ValueError(f"Unexpected semantic cache for {role}.")
        yield role, (rows,cached["packet"])


if __name__ == "__main__":
    stages = dict(reference=freeze_reference, proposals=collect_rf, release=collect_release, features=collect_dino)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=stages)
    stages[cli.parse_args().stage]()
