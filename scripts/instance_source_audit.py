"""Audit public syringe instance labels and bind visual decisions to local photos."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps

from medsyringe.data import dhash, digest, read_image, read_json, valid_box, write_json
from scripts.detection_public_benchmark import lexical_key, records, request, transformed_hashes, url_key
from scripts.public_annotated_collection import open_images
from scripts.public_catalog_exposure_audit import ART, protected_projects

ROOT = Path("data/instance-source-review")
INPUTS = ROOT / "inputs"
STAGED = ROOT / "open-images-publisher-boxes.json"
REVIEW = ART / "instance_source_visual_review.json"
REPORT = ART / "instance_source_audit.json"
LABPICS = INPUTS / "LabPicsMedical.zip"
LABPICS_MD5 = "a737eaa500169916d7269b3d9399b525"
HISTORICAL_OI = Path("data/detection_hillclimb/openimages/images")
SOURCE_FAMILIES = {
    "10.17632/mh6zpc7y9g.1": "Mendeley Dental held out as a source",
    "10.5281/zenodo.16924786": "Simulated phlebotomy used in prior fit/development",
    "michaels-workspace-ldsqh/syringe-project": "Capture family of protected syringe-detection-fnu4f desk setup",
}
URL_FIELDS = ("source_url", "original_url", "image_url", "download_url", "author_profile")
PRIVATE_INVENTORIES = (Path("artifacts/source_inventory.json"), Path("artifacts/expanded/inventory.json"))


def fingerprint_collision(hashes, index):
    return any((h ^ prior).bit_count() <= 4 for h in hashes for prior in index)


def protection():
    manifests = [*Path("artifacts").glob("*manifest*.json"),
                 *Path("artifacts/expanded").glob("*manifest*.json"),
                 *ART.glob("*manifest*.json"),
                 ART / "public_additional_fit.json", ART / "commons_crowds_reserve.json",
                 ART / "phlebotomy.json", ART / "dfine_photo_ablation_manifest.json"]
    history = {key: set() for key in ("ids", "authors", "urls", "sha256", "dhash")}
    for path in sorted(set(manifests)):
        for row in records(read_json(path)):
            history["ids"].add(str(row["id"]).removeprefix("oi-"))
            if row.get("author"):
                history["authors"].add(lexical_key(row["author"]))
            history["urls"].update(url_key(row[k]).rstrip("/") for k in URL_FIELDS if row.get(k))
            history["sha256"].update(row[k] for k in ("sha256", "derived_sha256") if row.get(k))
            if re.fullmatch(r"[\da-fA-F]{16}", str(row.get("dhash", ""))):
                history["dhash"].add(int(row["dhash"], 16))
    private_sha, private_dhash = set(), set()
    for path in PRIVATE_INVENTORIES:
        inventory = read_json(path)
        for row in inventory if isinstance(inventory, list) else inventory["rows"]:
            private_sha.update(row[key] for key in ("sha256", "derived_sha256") if row.get(key))
            if row.get("dhash"):
                private_dhash.add(int(row["dhash"], 16))
    return history, private_sha, private_dhash


def asset(row):
    key = row["source_id"]
    target = ROOT / "open-images" / f"{key}.jpg"
    if not target.exists():
        old = HISTORICAL_OI / f"{key}.jpg"
        try:
            payload = old.read_bytes() if old.exists() else request(row["download_urls"][0], 18 * 1024**2)
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            target.write_bytes(payload)
        except (OSError, ValueError) as error:
            return row, None, f"Raster unavailable ({type(error).__name__})"
    return row, target, None


def open_images_audit(history, private_sha, private_dhash):
    (ROOT / "open-images").mkdir(parents=True, exist_ok=True)
    metadata_rejections, candidates = [], []
    for row in open_images():
        if row["publisher_partition"] != "train":
            continue
        rid = row["source_id"]
        aliases = {url_key(row[key]).rstrip("/") for key in URL_FIELDS if row.get(key)}
        if rid in history["ids"] or aliases & history["urls"] or lexical_key(row["author"]) in history["authors"]:
            metadata_rejections.append(dict(id=row["id"], reason="Known public source photo or contributor"))
        else:
            candidates.append(row)
    output, exclusions = [], []
    unique_sha, unique_dhash = set(), set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row, path, failure in pool.map(asset, sorted(candidates, key=lambda row: row["id"])):
            if failure:
                exclusions.append(dict(id=row["id"], reason=failure))
                continue
            raster = read_image(path)
            checksum = digest(path)
            fingerprints = transformed_hashes(raster)
            if checksum in private_sha or fingerprint_collision(fingerprints, private_dhash):
                failure = "Fil Rouge hash or visual fingerprint"
            elif checksum in history["sha256"] or fingerprint_collision(fingerprints, history["dhash"]):
                failure = "Previously referenced public hash or visual fingerprint"
            elif checksum in unique_sha or fingerprint_collision(fingerprints, unique_dhash):
                failure = "Exact or close peer photo"
            if failure:
                exclusions.append(dict(id=row["id"], reason=failure))
                continue
            unique_sha.add(checksum)
            unique_dhash.update(fingerprints)
            annotations = row["annotations"]
            with Image.open(path) as source:
                orientation = source.getexif().get(274, 1)
            geometry_issues = []
            if annotations["coordinates_require_review"] or orientation != 1:
                geometry_issues.append("rotation")
            for flag in ("group_of", "depiction", "inside"):
                if any(box[flag] for box in annotations["boxes"]):
                    geometry_issues.append(flag)
            if min(raster.size) < 200:
                geometry_issues.append("small raster")
            output.append(dict(row, image=path.as_posix(), dimensions=list(raster.size),
                               sha256=checksum, dhash=dhash(raster), geometry_issues=geometry_issues,
                               license_basis="Open Images metadata record with Flickr photo and photographer",
                               license_checked_at_origin=False))
    write_json(STAGED, dict(rows=output, metadata_rejections=metadata_rejections, raster_rejections=exclusions))
    contact(output, "open-images")
    return output


def labpics_audit():
    with LABPICS.open("rb") as file:
        if file_digest(file, "md5").hexdigest() != LABPICS_MD5:
            raise ValueError("LabPics archive differs from the depositor checksum")
    result = []
    thumbnails = ROOT / "labpics-quarantine"
    thumbnails.mkdir(exist_ok=True)
    with ZipFile(LABPICS) as archive:
        annotation_names = sorted(name for name in archive.namelist()
                                  if name.startswith(("LabPics Medical/Train/", "LabPics Medical/Test/"))
                                  and name.endswith("/Data.json"))
        for name in annotation_names:
            base = name.removesuffix("Data.json")
            annotations = json.loads(archive.read(name))
            vessels = [v for v in annotations["Vessels"].values() if "Syringe" in v["VesselType_ClassNames"]]
            if not vessels:
                continue
            payload = archive.read(base + "Image.jpg")
            with Image.open(BytesIO(payload)) as photo:
                width, height = photo.size
            objects = []
            for vessel in vessels:
                member = base + vessel["MaskFilePath"].lstrip("/")
                with Image.open(BytesIO(archive.read(member))) as mask:
                    if mask.size != (width, height):
                        raise ValueError(f"Mask dimensions differ from the photo in {member}")
                    l, t, r, b = mask.convert("L").getbbox()
                bbox = [l / width, t / height, r / width, b / height]
                if not valid_box(bbox):
                    raise ValueError(f"Invalid converted bbox for {member}")
                objects.append(dict(bbox=bbox, member=member, instance_id=vessel["Indx"]))
            details = archive.read(base + "Source.txt").decode("utf-8-sig").splitlines()
            source = {k: v for k, v in (line.split(":", 1) for line in details)}
            identifier = base.split("/")[-2]
            path = thumbnails / f"{identifier}.jpg"
            path.write_bytes(payload)
            result.append(dict(id=identifier, image=path.as_posix(), dimensions=[width, height],
                               sha256=sha256(payload).hexdigest(), archive_member=base + "Image.jpg",
                               source_origin=source["Source"], source_name=source["Name"].removeprefix("Source:"),
                               source_url="https://zenodo.org/records/4736111",
                               annotations=dict(format="publisher-instance-mask bounds", boxes=objects)))
    write_json(ROOT / "labpics-quarantine.json", dict(
        rows=result, archive_sha256=digest(LABPICS), depositor_license="MIT",
        scanned_source_scenes=len(annotation_names),
        limitation="Third-party YouTube/Instagram frame rights and capture lineage unresolved."))
    contact(sorted(result, key=lambda row: (-len(row["annotations"]["boxes"]), row["id"]))[:12], "labpics-audit")
    return result


def contact(rows, stem):
    directory = ROOT / "sheets"
    directory.mkdir(exist_ok=True)
    for start in range(0, len(rows), 12):
        page = Image.new("RGB", (1200, 900), "#f3f5f5")
        draw = ImageDraw.Draw(page)
        for index, row in enumerate(rows[start:start + 12]):
            left, top = 300 * (index % 4) + 8, 300 * (index // 4) + 8
            photo = ImageOps.contain(read_image(row["image"]), (284, 248))
            page.paste(photo, (left, top))
            for box in row["annotations"]["boxes"]:
                x0, y0, x1, y1 = box["bbox"]
                color = "#ea9040" if box.get("group_of") or box.get("depiction") else "#0ffa72"
                draw.rectangle((left + x0 * photo.width, top + y0 * photo.height,
                                left + x1 * photo.width, top + y1 * photo.height), outline=color, width=2)
            draw.text((left, top + 251), row["id"], fill="#202528")
            draw.text((left, top + 273), ", ".join(row.get("geometry_issues", ())) or
                      row.get("title", row.get("source_origin", ""))[:34], fill="#202528")
        page.save(directory / f"{stem}-{start // 12:02}.jpg", quality=91)


def publish():
    published = read_json(STAGED)
    rows = published["rows"]
    labpics = read_json(ROOT / "labpics-quarantine.json")
    lookup = {row["id"]: row for row in rows}
    reviews = read_json(REVIEW)["open_images"] if REVIEW.exists() else {}
    license_evidence = ROOT / "flickr-license-evidence.json"
    licenses = read_json(license_evidence) if license_evidence.exists() else {}
    checked, accepted = [], []
    for identifier, review in sorted(reviews.items()):
        row = lookup[identifier]
        if review["sha256"] != row["sha256"] or digest(row["image"]) != row["sha256"]:
            raise ValueError(f"Stale visual evidence for {identifier}")
        if review["status"] == "full-scene one-to-one":
            issues = set(row["geometry_issues"])
            if review.get("orientation_reviewed"):
                issues.discard("rotation")
            if issues or not all(valid_box(o["bbox"]) for o in row["annotations"]["boxes"]):
                raise ValueError(f"Unsuitable publisher geometry in {identifier}")
            checked.append(dict(row, review=review))
            if licenses.get(identifier, {}).get("license") == row["license"] and not review.get("unresolved_parent"):
                accepted.append(dict(row, review=review, license_checked_at_origin=True,
                                     origin_license_evidence=licenses[identifier]))
    write_json(ROOT / "visually-checked.json", dict(rows=checked, independent_human_review=False))
    write_json(ROOT / "origin-license-verified.json", dict(rows=accepted, independent_human_review=False))
    write_json(REPORT, dict(
        target=500, protected_universe_projects={p: sorted(s) for p, s in protected_projects().items()},
        protected_named_families=SOURCE_FAMILIES,
        open_images=dict(
            staged=len(rows), publisher_boxes=sum(len(row["annotations"]["boxes"]) for row in rows),
            aabb_candidates=sum(not row["geometry_issues"] for row in rows),
            exclusions=dict(Counter(r["reason"] for r in [*published["metadata_rejections"],
                                                         *published["raster_rejections"]])),
            source_license_checks=dict(Counter(row["status"] for row in licenses.values())),
            reviewed=len(reviews), visually_one_to_one=len(checked),
            origin_license_verified=len(accepted),
            visually_one_to_one_instances=sum(len(row["annotations"]["boxes"]) for row in checked),
            manifest=str(STAGED), manifest_sha256=digest(STAGED)),
        labpics=dict(scanned_source_scenes=labpics["scanned_source_scenes"], labeled_scenes=len(labpics["rows"]),
                     publisher_instance_masks=sum(len(row["annotations"]["boxes"]) for row in labpics["rows"]),
                     source_counts=dict(Counter(r["source_origin"] for r in labpics["rows"])),
                     multiple_instances=sum(len(row["annotations"]["boxes"]) > 1 for row in labpics["rows"]),
                     eligible=0, manifest=str(ROOT / "labpics-quarantine.json")),
        independent_human_approved=0,
        review_status="Local artifact and geometry checks plus agent visual review. Independent review remains pending.",
    ))
    print(read_json(REPORT)["open_images"], flush=True)


def probe_flickr(identifier, url):
    """Record source-side CC BY 2.0 evidence independently from provider metadata."""
    try:
        html = request(url, 4 * 1024**2).decode("utf-8", "replace")
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        return identifier, {"status": f"Source unavailable ({getattr(error, 'code', type(error).__name__)})"}
    match = re.search(r'<a[^>]*class="license-text"[^>]*>', html, re.I)
    markup = match.group() if match else ""
    if re.search(r'href="https?://creativecommons\.org/licenses/by/2\.0(?:/[^" ]*)?"', markup, re.I):
        return identifier, {"status": "Source page declares CC BY 2.0",
                            "license": "https://creativecommons.org/licenses/by/2.0/", "source_url": url,
                            "license_markup": markup}
    status = "All rights reserved at source" if 'title="All rights reserved"' in markup else "Source license unresolved"
    return identifier, {"status": status, "source_url": url, "license_markup": markup or None}


def license_check():
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = dict(pool.map(lambda row: probe_flickr(row["id"], row["source_url"]), read_json(STAGED)["rows"]))
    write_json(ROOT / "flickr-license-evidence.json", result)
    print(Counter(r["status"] for r in result.values()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "check-flickr"))
    action = parser.parse_args().action
    if action == "prepare":
        open_images_audit(*protection())
        labpics_audit()
    elif action == "check-flickr":
        license_check()
    else:
        publish()


if __name__ == "__main__":
    main()
