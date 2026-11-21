"""Prepare fixed, background-aware inputs for local detector comparisons."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps

from medsyringe.data import dhash, digest, leakage, read_json, write_json
from scripts.public_transfer_study import ARCHIVE, MANIFEST
from scripts.rfdetr_transfer import PUBLIC_VALID, export, polygons

ART = Path("artifacts/detection_hillclimb")
DATA = Path("data/detection_hillclimb")
LOCK = ART / "phlebotomy.json"
COCO = ART / "training_inputs.json"
DOI = "https://doi.org/10.5281/zenodo.16924786"
SEED = 1107
PATTERN = re.compile(r"frame_(\d+)_(\d+)_jpg\.rf\..+\.jpg")
RESERVES = ("artifacts/rfdetr_audit_manifest.json", "artifacts/rfdetr_verifier_reserve.json")


def numeric_id(member):
    match = PATTERN.fullmatch(PurePosixPath(member).name)
    return (match[1], int(match[2])) if match else None


def old_groups():
    training = read_json(MANIFEST)["rows"]
    audits = [r for path in RESERVES for r in read_json(path)["rows"]]
    groups = lambda rows: {key[0] for r in rows if (key := numeric_id(r["member"]))}
    blocked = groups(audits)
    calibration = set(PUBLIC_VALID) - blocked
    fit = groups(training) - blocked - calibration
    return fit, calibration, blocked, training, audits


def design():
    if LOCK.exists():
        raise ValueError("The archive design has already been frozen.")
    manifest = read_json(MANIFEST)
    if digest(ARCHIVE) != manifest["archive_sha256"]:
        raise ValueError("The locally stored source archive differs from its manifest.")
    fit, calibration, blocked, old, audits = old_groups()
    previous = [*old, *audits]
    seen_members = {r["member"] for r in previous}
    reference = {role: [] for role in ("fit", "development", "unseen_negatives")}
    folder = DATA / "phlebotomy"
    folder.mkdir(parents=True, exist_ok=True)
    rejected = Counter()
    timings = defaultdict(list)
    with ZipFile(ARCHIVE) as zip_file:
        identifiers = set(zip_file.namelist())
        by_role = defaultdict(list)
        defined_groups = fit | calibration | blocked
        for file in sorted(identifiers):
            key = numeric_id(file)
            if not key:
                continue
            key_group, frame = key
            parent = PurePosixPath(file)
            if parent.parent.name != "images":
                continue
            if key_group in blocked:
                continue
            split = "fit" if key_group in fit else "development" if key_group in calibration else "unseen_negatives"
            if split == "fit" and parent.parts[0] != "train":
                continue
            if split == "unseen_negatives" and key_group in defined_groups:
                continue
            label = str(parent.parent.parent / "labels" / (parent.stem+".txt"))
            if label not in identifiers:
                continue
            try:
                objects = list(polygons(zip_file.read(label).decode(), manifest["class_id"]))
            except (UnicodeError, ValueError):
                rejected["invalid_original_polygon"] += 1
                continue
            if split == "unseen_negatives" and objects:
                raise ValueError("An unassigned clip contains positive examples. Revise the benchmark design.")
            by_role[split].append((file, key_group, frame, objects, label))
        # Freeze development and audit examples before taking any fit sample.
        order = ("unseen_negatives", "development", "fit")
        old_fit = [r for r in old if (key := numeric_id(r["member"])) and key[0] in fit]
        old_dev = [r for r in old if (key := numeric_id(r["member"])) and key[0] in calibration]
        for role in order:
            opposite = old_fit+old_dev+audits if role == "unseen_negatives" else (
                old_fit+audits+reference["unseen_negatives"] if role == "development" else
                old_dev+audits+reference["development"]+reference["unseen_negatives"])
            blocked_hashes = {r["sha256"] for r in opposite}
            visual_blocks = [int(r["dhash"],16) for r in opposite if r.get("dhash")]
            totals = Counter()
            selected_frames = set()
            ordered = sorted(by_role[role], key=lambda row: sha256(f"{SEED}:{row[0]}".encode()).digest())
            for file, group, frame, objects, label in ordered:
                positive = bool(objects)
                maximum = {"fit": (2,4), "development": (6,12), "unseen_negatives": (8,0)}[role][positive]
                if totals[group,positive] >= maximum or (group,frame) in selected_frames:
                    continue
                if file in seen_members or any(abs(frame-older) < 5 for older in timings[role,group]):
                    continue
                contents = zip_file.read(file)
                checksum = sha256(contents).hexdigest()
                with Image.open(BytesIO(contents)) as image:
                    width, height = image.size
                    visual = dhash(image)
                if checksum in blocked_hashes or any((int(visual,16)^h).bit_count() <= 4 for h in visual_blocks):
                    rejected[f"{role}_hash_collision"] += 1
                    continue
                identifier = f"phl-{checksum[:16]}"
                path = folder / f"{identifier}.jpg"
                path.write_bytes(contents)
                reference[role].append(dict(id=identifier, image=path.relative_to("data").as_posix(),
                      split=role, group=f"phlebotomy-{group}", member=file, source_url=DOI+"#"+file,
                      license="https://creativecommons.org/licenses/by/4.0/", sha256=checksum, dhash=visual,
                      width=width, height=height, objects=objects, label_member=label))
                blocked_hashes.add(checksum)
                visual_blocks.append(int(visual,16))
                timings[role,group].append(frame)
                selected_frames.add((group,frame))
                totals[group,positive] += 1
    combined = [r for candidates in reference.values() for r in candidates]
    collisions = leakage(combined)
    if collisions:
        raise ValueError(f"The generated archive inputs overlap: {collisions[:3]}")
    counts = {role: dict(images=len(rows), positives=sum(bool(r["objects"]) for r in rows),
                         boxes=sum(len(r["objects"]) for r in rows), groups=len({r["group"] for r in rows}))
              for role,rows in reference.items()}
    if min(counts["development"]["positives"], counts["fit"]["positives"]) < 8:
        raise ValueError(f"Too few development references: {counts}")
    write_json(LOCK, dict(rows=combined, counts=counts, seed=SEED, archive_sha256=digest(ARCHIVE),
                         provenance=DOI, excluded=dict(rejected), prior_reserve_clips=sorted(blocked),
                         source_manifests={str(path): digest(path) for path in (MANIFEST, *RESERVES)},
                         scope="Within one phlebotomy simulator domain. Filename groups are capture proxies."))
    print(counts, dict(rejected), flush=True)


def prepare():
    manifest = read_json(LOCK)
    coat = {}
    for group, role in (("train", "fit"), ("valid", "development")):
        subset = [r for r in manifest["rows"] if r["split"] == role]
        coat[group] = export(subset, DATA / "whole-frame" / group)
    write_json(COCO, dict(corpus_sha256=digest(LOCK), **coat))
    print({r: {k: rows[k] for k in ("images", "objects", "negatives")} for r,rows in coat.items()}, flush=True)


def review():
    directory = DATA / "review"
    directory.mkdir(parents=True, exist_ok=True)
    data = read_json(LOCK)["rows"]
    for role in ("fit","development","unseen_negatives"):
        rows = [r for r in data if r["split"] == role]
        rows = sorted(rows, key=lambda r: sha256(r["id"].encode()).digest())[:24]
        if not rows:
            continue
        montage = Image.new("RGB", (1280, 205*((len(rows)+3)//4)), "#202027")
        draw = ImageDraw.Draw(montage)
        for i,row in enumerate(rows):
            with Image.open(Path("data")/row["image"]) as image:
                small = ImageOps.contain(image, (308,175))
                left, top = (i%4)*320+6, (i//4)*205+25
                montage.paste(small,(left,top))
                draw.text((left,top-19),row["id"],fill="white")
                for obj in row["objects"]:
                    rect = [left+obj["bbox"][0]*small.width, top+obj["bbox"][1]*small.height,
                            left+obj["bbox"][2]*small.width, top+obj["bbox"][3]*small.height]
                    draw.rectangle(rect,outline="#f3f564",width=2)
        montage.save(directory/f"{role}.jpg",quality=91)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "prepare", "review"))
    args = parser.parse_args()
    {"design": design, "prepare": prepare, "review": review}[args.stage]()
