"""Stage the official Open Images train boxes for a separate, later study."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image,ImageDraw,ImageOps

from medsyringe.data import digest, dhash, leakage, read_image, read_json, valid_box, write_json
from scripts.experiments.detector_training.detection_hillclimb_corpus import ART, DATA
from scripts.experiments.public_data.detection_public_benchmark import lexical_key,near_any,prior_exposure,request,url_key

PUBLIC = "https://storage.googleapis.com/openimages/web/download_v7.html"
FOLDER = DATA/"openimages"
BOXES = FOLDER/"boxes-syringe.csv"
META = FOLDER/"syringe-metadata.csv"
STAGED = FOLDER/"source-screen.json"
REVIEW = ART/"openimages_additional_reference_review.json"
MANIFEST = ART/"openimages_additional_manifest.json"
FILTER_FLAGS = ("IsGroupOf","IsDepiction","IsInside")
SEED = "open-images-additional-2026-11-21"


def rank(value):
    return sha256(f"{SEED}:{value}".encode()).digest()


def sources():
    data = defaultdict(list)
    with BOXES.open(newline="",encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["LabelName"]!="/m/012ysf":
                raise ValueError("Filtered CSV contains a different class.")
            data[row["ImageID"]].append(row)
    with META.open(newline="",encoding="utf-8") as file:
        metadata = {row["ImageID"]:row for row in csv.DictReader(file)}
    blocked = prior_exposure()
    for file in ("public_manifest.json","public_additional_fit.json","dental_manifest.json","phlebotomy.json"):
        for r in read_json(ART/file)["rows"]:
            blocked["ids"].add(r["id"])
            blocked["hashes"].add(r["sha256"])
            blocked["difference"].add(int(r["dhash"],16))
            blocked["authors"].update(r.get("profile_keys",[]))
            if r.get("author"):
                blocked["authors"].add(lexical_key(r["author"]))
            blocked["urls"].update(url_key(r[k]) for k in ("source_url","original_url","image_url") if r.get(k))
    return data,metadata,blocked


def staged(identifier,records,metadata):
    if not metadata:
        return None,"missing publisher metadata"
    if any(label[k]!="0" for label in records for k in FILTER_FLAGS):
        return None,"depiction, group or uncertain category scope"
    if metadata["Rotation"] not in {"0","0.0"}:
        return None,"nonzero or unknown publisher rotation"
    license_key = url_key(metadata["License"]).rstrip("/")
    author = metadata["Author"].strip()
    profile = metadata["AuthorProfileURL"].strip()
    landing = metadata["OriginalLandingURL"].strip()
    if license_key!="creativecommons.org/licenses/by/2.0" or not (author and profile and landing):
        return None,"attribution or image license unavailable"
    relative = FOLDER/"images"/(identifier+".jpg")
    url = f"https://open-images-dataset.s3.amazonaws.com/train/{identifier}.jpg"
    try:
        if not relative.exists():
            bytes_read = request(url,9*1024*1024)
            with Image.open(BytesIO(bytes_read)) as im:
                im.verify()
            relative.write_bytes(bytes_read)
        raster = read_image(relative)
    except (OSError,ValueError):
        return None,"image unavailable or unreadable"
    if min(raster.size)<200:
        return None,"image too small"
    objects = [dict(bbox=[float(label[k]) for k in ("XMin","YMin","XMax","YMax")],
                    occluded=label["IsOccluded"]=="1",truncated=label["IsTruncated"]=="1",provider=label["Source"])
               for label in records]
    if any(not valid_box(o["bbox"]) for o in objects):
        return None,"invalid provider coordinates"
    return (dict(id=identifier,title=metadata["Title"],author=author,
                 group="openimages-profile:"+url_key(profile),author_profile=profile,source_url=landing,
                 original_url=metadata["OriginalURL"],image_url=url,license="https://"+license_key+"/",
                 image=relative.relative_to(Path("data")).as_posix(),width=raster.width,height=raster.height,
                 sha256=digest(relative),dhash=dhash(raster),objects=objects),raster),None


def contact(rows):
    destination = FOLDER/"review"
    destination.mkdir(exist_ok=True)
    for start in range(0,len(rows),12):
        page = rows[start:start+12]
        img = Image.new("RGB",(1200,810),"#f4f7f7")
        drawer = ImageDraw.Draw(img)
        for no,row in enumerate(page):
            column,line = no%4,no//4
            left,top = column*300+8,line*270+6
            scene = ImageOps.contain(read_image(Path("data")/row["image"]),(281,233))
            img.paste(scene,(left,top))
            for obj in row["objects"]:
                l,t,r,b = obj["bbox"]
                drawer.rectangle((left+l*scene.width,top+t*scene.height,left+r*scene.width,top+b*scene.height),
                                 outline="#f78c2f",width=2)
            drawer.text((left,top+239),row["id"]+" · "+row["title"][:22],fill="#1a2e35")
        img.save(destination/f"official-{start//12:02}.jpg",quality=92)


def prepare():
    if MANIFEST.exists():
        raise ValueError("The reviewed partition assignments have already been frozen.")
    rows,meta,blocked = sources()
    (FOLDER/"images").mkdir(exist_ok=True)
    exclusions = Counter()
    kept=[]
    ids = sorted(rows,key=rank)
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared = pool.map(lambda i:staged(i,rows[i],meta.get(i)),ids)
        for (response,message) in prepared:
            if message:
                exclusions[message] += 1
                continue
            item,scene = response
            if (item["id"] in blocked["ids"] or item["sha256"] in blocked["hashes"]
                or lexical_key(item["author"]) in blocked["authors"] or
                any(url_key(item[key]) in blocked["urls"] for key in ("original_url","source_url","image_url")) or
                near_any(scene,blocked["difference"])):
                exclusions["known author, file, source URL, or close transformed fingerprint"] += 1
                continue
            kept.append(item)
            blocked["hashes"].add(item["sha256"])
            blocked["difference"].add(int(item["dhash"],16))
    write_json(STAGED,dict(rows=kept,native_boxes=sum(map(len,rows.values())),native_images=len(ids),
                           excluded=dict(exclusions),metadata_sha256=digest(META),boxes_sha256=digest(BOXES),
                           instructions="Review anatomy, every visible instance and the linked provider boxes before use."))
    contact(kept)
    print(len(kept),sum(len(r["objects"]) for r in kept),dict(exclusions),flush=True)


def seal():
    if MANIFEST.exists():
        raise ValueError("The study assignments have already been frozen.")
    reviewed = read_json(REVIEW)
    draft = read_json(STAGED)
    if reviewed["source_screen_sha256"]!=digest(STAGED):
        raise ValueError("Blind references differ from their staged photos.")
    all_rows = {r["id"]:r for r in draft["rows"]}
    if set(reviewed["reviews"])!=set(all_rows):
        raise ValueError("The visual review must explicitly settle every staged photo.")
    confirmed = []
    for identifier,review in reviewed["reviews"].items():
        if review["status"] in {"verified","confirmed lookalike"}:
            row = all_rows[identifier]
            objects = review.get("objects",row["objects"]) if review["status"]=="verified" else []
            lookalikes = review.get("lookalikes",[])
            if not (objects or lookalikes) or any(not valid_box(o["bbox"]) for o in (*objects,*lookalikes)):
                raise ValueError(f"The accepted image has unresolved target identity or geometry: {identifier}")
            confirmed.append({**row,"objects":objects,"lookalikes":lookalikes,
                              "reference_review":"Visual audit without model proposals"})
    groups = defaultdict(list)
    for row in confirmed:
        groups[row["group"]].append(row)
    strata = defaultdict(list)
    for group,rows in groups.items():
        multiple = any(len(r["objects"])>1 for r in rows)
        partial = any(o.get("occluded") or o.get("truncated") for r in rows for o in r["objects"])
        kinds = tuple(sorted({bool(r["objects"]) for r in rows}))
        strata[kinds,multiple,partial].append(group)
    for stratum in strata.values():
        groups_in_order = sorted(stratum,key=rank)
        for index,group in enumerate(groups_in_order):
            reserved = index%5==2 or (len(groups_in_order)==2 and index==1)
            for row in groups[group]:
                row["split"] = "openimages-audit" if reserved else "openimages-fit"
    if leakage(confirmed):
        raise ValueError("The locked assignments share an author, source or apparent near duplicate.")
    counts = {role:dict(images=sum(r["split"]==role for r in confirmed),
                        objects=sum(len(r["objects"]) for r in confirmed if r["split"]==role),
                        negatives=sum(r["split"]==role and not r["objects"] for r in confirmed),
                        groups=len({r["group"] for r in confirmed if r["split"]==role}))
              for role in ("openimages-fit","openimages-audit")}
    write_json(MANIFEST,dict(rows=confirmed,counts=counts,source=PUBLIC,seed=SEED,
                             reviewed_sha256=digest(REVIEW), source_sha256=digest(STAGED),
                             scope="New fit and audit partitions remixed from the publisher train boxes only. "
                                   "Syringe-positive enrichment and visually confirmed lookalikes. "
                                   "Foundation-model pretraining exposure is unknown."))
    print(counts,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","seal"))
    (prepare if parser.parse_args().action=="prepare" else seal)()
