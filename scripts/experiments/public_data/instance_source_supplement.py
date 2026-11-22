"""Prepare and verify newly drawn syringe boxes on licensed public photographs."""
import argparse
from collections import Counter
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageOps

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, transformed_hashes, url_key
from scripts.experiments.public_data.instance_source_audit import ART, ROOT, fingerprint_collision, protection

DATA = Path("data")
ANNOTATED = DATA / "public-annotated-collection"
LOCAL = (ANNOTATED / "manifest.json", DATA / "public-bulk-syringe-collection/manifest.json",
         DATA / "public-web-collection/manifest.json")
QUEUE = ROOT / "manual-box-queue.json"
REVIEW = ART / "instance_source_supplement_review.json"
MANIFEST = ROOT / "manual-boxes.json"
REPORT = ART / "instance_source_supplement.json"
CC = re.compile(r"^https?://creativecommons\.org/(?:licenses/(?:by|by-sa)/[1-4]\.\d|publicdomain/zero/1\.0)(?:/|$)")
SYRINGE_TITLE = re.compile(r"sering|syring|spritz|inject|siring|jering|vacci|hypoderm", re.I)
CROWD_TAGS = {"dense", "overlap"}


def catalog():
    visual = read_json(ANNOTATED / "visual_review.json")["rows"]
    for manifest in LOCAL:
        for row in read_json(manifest)["rows"]:
            if not row["id"].startswith(("commons-", "cdc-phil-")) or not row.get("author"):
                continue
            source, license_url = row["source_url"], row.get("license_url") or row["license"]
            cdc = (source.startswith("https://wwwn.cdc.gov/phil/Details.aspx?pid=")
                   and license_url == "Public Domain (CDC PHIL)")
            commons = source.startswith("https://commons.wikimedia.org/wiki/File:")
            federal = (commons and license_url == "Public domain" and
                       bool(re.search(r"\bU\.S\. (?:Air Force|Army|Navy|Food and Drug)|"
                                      r"United States (?:Navy|Air Force|Army)|National Cancer Institute|"
                                      r"obamawhitehouse\.archives\.gov",
                                      f"{row['author']} {row.get('credit', '')}", re.I)))
            if not (cdc or federal or (commons and CC.search(license_url))):
                continue
            note = visual.get(row["id"], {})
            tags = set(note.get("tags", []))
            if row.get("stratum") == "crowded" or CROWD_TAGS & tags or row.get("intent") == "lookalike":
                continue
            if note.get("scene_role") in {"lookalike", "reject", "detail-only"}:
                continue
            title = row.get("title") or row.get("caption") or ""
            if not (cdc or note.get("scene_role") == "syringe" or SYRINGE_TITLE.search(title)):
                continue
            if federal or license_url.startswith("http:"):
                score = 2
            else:
                score = 0 if cdc or note.get("scene_role") == "syringe" else 1
            yield dict(id=row["id"], title=title, source_url=source, original_url=row["original_url"],
                       author=row["author"], group=row["group"], license=license_url, priority=score,
                       credit=row.get("credit"),
                       prior_tags=sorted(tags), sha256=row["sha256"], image=str(DATA / row["image"]),
                       source_manifest=manifest.as_posix())


def prepare():
    known, private_sha, private_dhash = protection()
    selected, local_ids, local_hashes, local_dhashes = [], set(), set(), set()
    rejected = Counter()
    for row in sorted(catalog(), key=lambda row: (row["priority"], row["id"])):
        identifier = row["id"]
        native_id = identifier.removeprefix("commons-")
        if identifier in local_ids:
            continue
        local_ids.add(identifier)
        if (identifier in known["ids"] or native_id in known["ids"]
                or lexical_key(row["author"]) in known["authors"]
                or any(url_key(row[field]).rstrip("/") in known["urls"] for field in ("source_url", "original_url"))):
            rejected["known public file or author"] += 1
            continue
        path = Path(row["image"])
        if digest(path) != row["sha256"]:
            raise ValueError(f"Manifest checksum differs from {path}")
        raster = read_image(path)
        if min(raster.size) < 180:
            rejected["small raster"] += 1
            continue
        fingerprint = transformed_hashes(raster)
        if (row["sha256"] in private_sha | known["sha256"] or
                fingerprint_collision(fingerprint, private_dhash | known["dhash"])):
            rejected["historical byte or perceptual collision"] += 1
            continue
        if row["sha256"] in local_hashes or fingerprint_collision(fingerprint, local_dhashes):
            rejected["exact or close peer image"] += 1
            continue
        row["dimensions"] = list(raster.size)
        selected.append(row)
        local_hashes.add(row["sha256"])
        local_dhashes.update(fingerprint)
    write_json(QUEUE, dict(rows=selected, rejected=rejected))
    grids(selected)
    print(len(selected), rejected, flush=True)


def grids(rows):
    directory = ROOT / "sheets"
    directory.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=14)
    for start in range(0, len(rows), 12):
        grid = Image.new("RGB", (1440, 960), "#f2f5f5")
        label = ImageDraw.Draw(grid)
        for i, row in enumerate(rows[start:start + 12]):
            left, top = 360 * (i % 4) + 20, 320 * (i // 4) + 20
            photo = ImageOps.contain(read_image(row["image"]), (320, 255))
            width, height = photo.size
            overlay = Image.new("RGBA", (width, height))
            draw = ImageDraw.Draw(overlay)
            for index in range(1, 10):
                x, y = round(index * width / 10), round(index * height / 10)
                color = (39, 220, 232, 125 if index == 5 else 65)
                draw.line((x, 0, x, height - 1), fill=color)
                draw.line((0, y, width - 1, y), fill=color)
            annotated = Image.alpha_composite(photo.convert("RGBA"), overlay).convert("RGB")
            grid.paste(annotated, (left, top))
            label.rectangle((left - 1, top - 1, left + width, top + height), outline="#04090a")
            label.text((left, top + 256), f"{row['id']}  {width}x{height}", fill="#143033", font=font)
            label.text((left, top + 277), row["title"].removeprefix("File:")[:42], fill="#143033", font=font)
        grid.save(directory / f"supplement-{start // 12:02}.jpg", quality=94)


def publish():
    queue = {row["id"]: row for row in read_json(QUEUE)["rows"]}
    reviews = read_json(REVIEW)
    if reviews["queue_sha256"] != digest(QUEUE):
        raise ValueError("Reviewed boxes are attached to a different candidate index")
    final, negative = [], []
    counts = Counter()
    for identifier, decision in sorted(reviews["rows"].items()):
        row = queue[identifier]
        if digest(row["image"]) != row["sha256"] or decision["sha256"] != row["sha256"]:
            raise ValueError(f"Review refers to a different raster {identifier}")
        boxes = decision.get("boxes", [])
        status = decision["status"]
        if status not in {"positive", "negative", "ambiguous", "excluded", "duplicate"}:
            raise ValueError(f"Unknown visual review decision on {identifier}")
        if bool(boxes) != (status == "positive") or not all(valid_box(b) for b in boxes):
            raise ValueError(f"Invalid boxes or inventory decision on {identifier}")
        counts[status] += 1
        row["group"] = decision.get("capture_group", row["group"])
        if status == "positive":
            final.append(dict(row, objects=[dict(label="syringe", bbox=b) for b in boxes], visual_review=decision))
        elif status == "negative":
            negative.append(dict(row, objects=[], visual_review=decision))
    write_json(MANIFEST, dict(rows=final, negative_rows=negative, independent_human_review=False))
    write_json(REPORT, dict(
        target=500, candidate_photos=len(queue), reviewed_photos=len(reviews["rows"]),
        decisions=counts, accepted_positive_photos=len(final), accepted_negative_photos=len(negative),
        visible_syringe_instances=sum(len(r["objects"]) for r in final),
        acquisition_groups=len({r["group"] for r in final}), manifest=str(MANIFEST),
        manifest_sha256=digest(MANIFEST), review_sha256=digest(REVIEW),
        review_status="One agent reviewed every listed scene and drew approximate visible-extent boxes without inference. "
                      "Independent human review remains pending."))
    print(read_json(REPORT), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    (prepare if parser.parse_args().action == "prepare" else publish)()
