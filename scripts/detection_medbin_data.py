"""Select and manually review a test-only native-resolution MedBin cohort."""
import argparse
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from medsyringe.data import dhash, digest, read_image, read_json, valid_box, write_json
from scripts.detection_empty_scene_data import MANIFEST as SCENES, exposure
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_precision_negative_audit import protect
from scripts.detection_public_benchmark import near_any, order, request, sheet

PROJECT = "https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2"
POOL = DATA/"medbin-source-pool.json"
CANDIDATES = DATA/"medbin_candidates.json"
REVIEW = ART/"medbin_reference_review.json"
MANIFEST = ART/"medbin_held_out_manifest.json"
ASSETS = DATA/"medbin-original"


def history():
    known = exposure()
    for row in read_json(SCENES)["rows"]:
        protect(known,row)
    return known


def classify(source):
    count = {k.strip().casefold():v for k,v in source["annotations"]["classes"].items()}
    if "syringe" in count:
        return "syringe with other labeled objects" if sum(count.values())>count["syringe"] else "syringe only"
    if any("pipette" in k or "pettor" in k for k in count):
        return "pipettes"
    if any("needle" in k for k in count):
        return "needles"
    if any("tube" in k for k in count):
        return "tubes"
    return None


def retrieve():
    if MANIFEST.exists() or CANDIDATES.exists():
        raise ValueError("The native candidate pool is already captured.")
    ASSETS.mkdir(parents=True,exist_ok=True)
    known = history()
    original = {r["id"]:r for rows in read_json(POOL).values() for r in rows
                if r["mediaType"]=="image"}
    groups = defaultdict(list)
    for row in original.values():
        if bucket:=classify(row):
            groups[bucket].append(row)
    selected=[]
    for bucket,quota in (("syringe with other labeled objects",12),("syringe only",10),
                         ("needles",7),("pipettes",7),("tubes",6)):
        count = 0
        for row in sorted(groups[bucket],key=lambda p:order(p["id"])):
            key,url = row["id"],row["url"]
            if not key.isalnum() or urlsplit(url).hostname!="source.roboflow.com":
                continue
            destination = ASSETS/f"{key}.jpg"
            try:
                if not destination.exists():
                    content = request(url,18*1024*1024)
                    with Image.open(BytesIO(content)) as candidate:
                        if candidate.format!="JPEG" or candidate.width*candidate.height>30_000_000:
                            continue
                        candidate.verify()
                    destination.write_bytes(content)
                picture = read_image(destination)
            except (OSError,ValueError,Image.DecompressionBombError):
                continue
            tag = digest(destination)
            if tag in known["hashes"] or near_any(picture,known["difference"]):
                continue
            selected.append(dict(id=key,image=str(destination.relative_to("data")),sha256=tag,
                                 original=True,dhash=dhash(picture),dimensions=list(picture.size),
                                 catalog_label_counts=row["annotations"]["classes"],
                                 catalog_split=row["split"],intent=bucket,
                                 group="medbin:unresolved-acquisition",title=bucket,
                                 author="medical waste dataset contributors",source_url=PROJECT,
                                 original_url=url,image_url=url, license="https://creativecommons.org/licenses/by/4.0/"))
            known["hashes"].add(tag)
            known["difference"].add(int(dhash(picture),16))
            count += 1
            if count==quota:
                break
        print(bucket,count,flush=True)
    write_json(CANDIDATES,dict(rows=selected,metadata_sha256=digest(POOL)))
    sheet(selected,"medbin-manual-review",4,3)


def freeze():
    if MANIFEST.exists():
        raise ValueError("The native test data are already frozen.")
    qa=read_json(REVIEW)
    raw=read_json(CANDIDATES)
    if qa["candidates_sha256"]!=digest(CANDIDATES):
        raise ValueError("The reviewed files differ from the downloaded cohort.")
    rows=[]
    for row in raw["rows"]:
        kept=qa["confirmed"].get(row["id"])
        path=Path("data")/row["image"]
        if kept is None:
            continue
        if digest(path)!=row["sha256"] or any(not valid_box(o["bbox"]) for o in kept["objects"]):
            raise ValueError("An accepted reference has changed or supplies invalid boxes.")
        rows.append({**row,**kept,"split":"medbin-reserve"})
    positive=sum(bool(r["objects"]) for r in rows)
    if positive<10 or len(rows)-positive<8 or len(rows)!=len(qa["confirmed"]):
        raise ValueError("The test needs at least ten positive and eight negative scenes.")
    write_json(MANIFEST,dict(
        rows=rows,reference_sha256=digest(REVIEW),catalog_url=PROJECT,
        scope="Native source JPEGs downloaded from the public CC BY 4.0 Universe listing and visually reviewed. "
              "No publisher inference, training or train/validation split is used. Grouped as one unresolved "
              "acquisition source because independent specimen and session identifiers are unavailable. "
              "Results apply to this test-only subset and do not establish the paper's complete dataset lineage.",
    ))
    print("Frozen test-only MedBin photos",len(rows),"positive",positive,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("retrieve","freeze"))
    (retrieve if parser.parse_args().action=="retrieve" else freeze)()
