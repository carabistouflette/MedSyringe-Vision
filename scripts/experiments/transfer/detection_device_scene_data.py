"""Stage native public device scenes with an emphasis on simultaneous instances."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from medsyringe.data import dhash, digest, read_image, read_json, valid_box, write_json
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.detector_audits.detection_medbin_data import MANIFEST as MEDBIN, history
from scripts.experiments.public_data.detection_public_benchmark import near_any, order, request, sheet

POOL=DATA/"device-catalog-pool.json"
CANDIDATES=DATA/"device-native-candidates.json"
REVIEW=ART/"device_scene_reference_review.json"
MANIFEST=ART/"device_scene_manifest.json"
ASSETS=DATA/"device-native"
PROJECT="https://universe.roboflow.com/test-uugy6/syringe-test"


def raw_count(row):
    labels=row["annotations"]["classes"]
    return labels.get("test-syringe",0)+labels.get("0",0)


def download(row):
    key,url=row["id"],row["url"]
    if not key.isalnum() or urlsplit(url).hostname!="source.roboflow.com":
        return None
    target=ASSETS/f"{key}.jpg"
    try:
        if not target.exists():
            content=request(url,14*1024*1024)
            with Image.open(BytesIO(content)) as probe:
                if probe.format!="JPEG" or probe.width*probe.height>30_000_000:
                    return None
                probe.verify()
            target.write_bytes(content)
        img=read_image(target)
    except (OSError,ValueError,Image.DecompressionBombError):
        return None
    checksum=digest(target)
    return (dict(id=key,image=str(target.relative_to("data")),sha256=checksum,dhash=dhash(img),
                 original_url=url,image_url=url,source_url=PROJECT,group="device:one-visual-acquisition-source",
                 author="test dataset contributors",license="https://creativecommons.org/licenses/by/4.0/",
                 dimensions=list(img.size),catalog_labels=raw_count(row),
                 catalog_split=row["split"],title="Native device-scene benchmark"), img)


def prepare():
    if CANDIDATES.exists() or MANIFEST.exists():
        raise ValueError("The candidate set is already staged.")
    raw=read_json(POOL)["rows"]
    ASSETS.mkdir(exist_ok=True)
    known=history()
    for row in read_json(MEDBIN)["rows"]:
        known["hashes"].add(row["sha256"])
        known["difference"].add(int(row["dhash"],16))
    multi=sorted([row for row in raw if raw_count(row)>1],key=lambda r:order(r["id"]))
    single=sorted([row for row in raw if raw_count(row)==1],key=lambda r:order(r["id"]))[:18]
    reference=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for retrieved in pool.map(download,(*multi,*single)):
            if retrieved is None:
                continue
            r,img=retrieved
            if r["sha256"] in known["hashes"] or near_any(img,known["difference"]):
                continue
            reference.append(r)
            known["hashes"].add(r["sha256"])
            known["difference"].add(int(r["dhash"],16))
    write_json(CANDIDATES,dict(rows=reference,source_pool_sha256=digest(POOL)))
    sheet(reference,"device-scene",4,3)
    print("Native candidates",len(reference),sum(r["catalog_labels"]>1 for r in reference),flush=True)


def freeze():
    if MANIFEST.exists():
        raise ValueError("The device benchmark is already sealed.")
    old=read_json(REVIEW)
    if digest(CANDIDATES)!=old["candidates_sha256"]:
        raise ValueError("The reviewed sources and current raster list differ.")
    reviewed=[]
    for r in read_json(CANDIDATES)["rows"]:
        reference=old["references"].get(r["id"])
        if reference is None:
            continue
        if digest(Path("data")/r["image"])!=r["sha256"] or not all(valid_box(a["bbox"]) for a in reference["objects"]):
            raise ValueError("The reference contains invalid coordinates or changed rasters.")
        reviewed.append({**r,**reference,"split":"device-count-reserve"})
    multi=sum(len(r["objects"])>1 for r in reviewed)
    if (len(reviewed)!=len(old["references"]) or len(reviewed)<12 or multi<5):
        raise ValueError("The visible references must include at least five crowded scenes.")
    write_json(MANIFEST,dict(rows=reviewed,review_sha256=digest(REVIEW),source_url=PROJECT,
                             scope="Independent of the provided project images and all training. Photos from this "
                                   "external project are one source group with unknown specimen and session IDs. "
                                   "Primary outcome is simultaneous object counting in the source's conditions."))
    print("Device-count reference sealed",len(reviewed),multi,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","freeze"))
    (prepare if parser.parse_args().action=="prepare" else freeze)()
