"""Stage licensed public syringe review candidates and verify the local manifest."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from shutil import copyfile, copyfileobj
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps
import yaml

from medsyringe.data import dhash, digest, read_image, read_json, valid_box, write_json
from scripts.experiments.transfer import detection_dental_challenge as dental
from scripts.experiments.public_data.detection_public_benchmark import (
    API, lexical_key, metadata, near_any, plain, prior_exposure, records, request, url_key,
)

DATA = Path("data")
HILL = DATA / "detection_hillclimb"
OUT = DATA / "public-annotated-collection"
INPUTS = OUT / "inputs"
MANIFEST = OUT / "manifest.json"
REVIEW = OUT / "visual_review.json"
SEED = "public-annotated-collection-2026-11-21"
MAX_BYTES = 18 * 1024**2
MAX_PIXELS = 32_000_000
SIDE = 1920
OI_CLASS = "/m/012ysf"
MEDBIN = "https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2"
CC_BY_40 = "https://creativecommons.org/licenses/by/4.0/"
DENTAL_SHA = "8d55211ee5d1250fc8883c3c9f09a86413a89281c4290babb5b715fcc4a495a1"
DENTAL_AUTHOR = "R Girija, Alpa Gupta, Rakshak Anand, Aravind Reddy Dhampetla, Tushank Kumar, Yash Kathait"
ATTRIBUTES = (
    "syringe_type", "markings_readable", "graduations_readable", "plunger_visible",
    "barrel_interior_visible", "liquid_visible",
)
COMMONS_SEEDS = (
    "Syringe details raw.jpeg", "Comparing insulin and tuberculin syringes.jpg",
    "Syringe-parts.jpg", "Influenza vaccine syringes.jpg", "Syringes injectienaalden.jpg",
    "Testing vaccine in laboratory. Holding syringe with protective medical gloves closeup. (51714051263).jpg",
    "Adjuvanted Quadrivalent Influenza Vaccine 2023.jpg", "Empty 5ml plastic syringe.jpg",
    "Syringe and Vaccine (14329622976).jpg", "Plastic syringe 10ml-01.jpg",
)
COMMONS_CATEGORIES = {
    "positive": (
        "Pre-filled syringes", "Disposable syringes", "Insulin syringes", "Syringes",
        "Syringe drivers", "Phlebotomy", "Vaccination", "Sharps waste",
        "Injection (medicine)", "Hypodermic needles", "Vaccination in the United States",
    ),
    "lookalike": (
        "Pasteur pipettes", "Air displacement pipettes", "Pipettes", "Ampoules",
        "Insulin pens", "Test tubes", "Droppers", "Mechanical pencils", "Ballpoint pens",
    ),
}
COMMONS_LIMITS = {"positive": 125, "lookalike": 90}
DENTAL_NEGATIVES = ("suction tube", "elevator", "tweezers", "needle holder", "tongue depressor")


def rank(identifier):
    return sha256(f"{SEED}:{identifier}".encode()).digest()


def cache_path(name, historical):
    target = INPUTS / name
    if not target.exists():
        copyfile(historical, target.with_suffix(".part"))
        target.with_suffix(".part").replace(target)
    return target


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def open_images():
    folder = HILL / "openimages"
    train_boxes = cache_path("oi-train-syringe-boxes.csv", folder / "boxes-syringe.csv")
    train_meta = cache_path("oi-train-syringe-metadata.csv", folder / "syringe-metadata.csv")
    train = defaultdict(list)
    for record in csv_rows(train_boxes):
        if record["LabelName"] == OI_CLASS:
            train[record["ImageID"]].append(record)
    test_snapshot = INPUTS / "oi-test-syringe.json"
    if not test_snapshot.exists():
        # The official test annotations also contain explicit image-level negatives.
        test = DATA / "openimages" / "test"
        boxes = defaultdict(list)
        for record in csv_rows(test / "boxes.csv"):
            if record["LabelName"] == OI_CLASS:
                boxes[record["ImageID"]].append(record)
        negatives = {
            r["ImageID"] for r in csv_rows(test / "labels.csv")
            if r["LabelName"] == OI_CLASS and r["Confidence"] == "0"
        } - boxes.keys()
        keep = boxes.keys() | negatives
        items = {r["ImageID"]: r for r in csv_rows(test / "metadata.csv") if r["ImageID"] in keep}
        write_json(test_snapshot, dict(boxes=boxes, negative_ids=sorted(negatives), metadata=items))
    test = read_json(test_snapshot)
    for partition, boxes, negatives, items in (
        ("train", train, set(), {r["ImageID"]: r for r in csv_rows(train_meta) if r["ImageID"] in train}),
        ("test", test["boxes"], set(test["negative_ids"]), test["metadata"]),
    ):
        for key, item in items.items():
            if item["License"].replace("http:", "https:").rstrip("/") + "/" != "https://creativecommons.org/licenses/by/2.0/":
                continue
            if not all(item[field] for field in ("Author", "AuthorProfileURL", "OriginalURL", "OriginalLandingURL")):
                continue
            if not re.fullmatch(r"[0-9a-f]{16}", key):
                raise ValueError("Unexpected Open Images ID")
            objects = []
            for r in boxes.get(key, []):
                box = [float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
                if not valid_box(box):
                    continue
                objects.append(dict(
                    label="Syringe", bbox=box, provider=r["Source"], occluded=r["IsOccluded"] == "1",
                    truncated=r["IsTruncated"] == "1", group_of=r["IsGroupOf"] == "1",
                    depiction=r["IsDepiction"] == "1", inside=r["IsInside"] == "1",
                ))
            if key not in negatives and not objects:
                continue
            rotation = item["Rotation"]
            url = f"https://open-images-dataset.s3.amazonaws.com/{partition}/{key}.jpg"
            yield dict(
                source="open-images", source_id=key, id=f"oi-{key}", publisher_partition=partition,
                title=item["Title"], source_url=item["OriginalLandingURL"], original_url=item["OriginalURL"],
                download_urls=[url], author=item["Author"], author_profile=item["AuthorProfileURL"],
                license="https://creativecommons.org/licenses/by/2.0/",
                group="open-images-profile:" + url_key(item["AuthorProfileURL"]).rstrip("/"),
                annotations=dict(format="open-images", boxes=objects, verified=False,
                                 annotations_license=CC_BY_40, publisher_rotation=rotation,
                                 coordinates_require_review=rotation not in ("0", "0.0"),
                                 image_level_syringe_absent=True if key in negatives else None),
                intent="lookalike" if key in negatives else "positive",
            )


def medbin():
    path = cache_path("medbin-public-catalog.json", HILL / "medbin-source-pool.json")
    rows = {r["id"]: r for page in read_json(path).values() for r in page if r["mediaType"] == "image"}
    for key, row in rows.items():
        url = row["url"]
        if not key.isalnum() or urlsplit(url).hostname != "source.roboflow.com":
            raise ValueError("Unexpected MedBin public asset reference")
        count = Counter()
        for label, number in row["annotations"]["classes"].items():
            count[label.strip().casefold()] += number
        if not count["syringe"] and not any(
            re.search("pipett|pettor|needle|tube|bottle|thermomet", label) for label in count
        ):
            continue
        yield dict(
            source="medbin-universe", id=f"medbin-{key}", source_id=key,
            group="medbin:unresolved-acquisition", author="medical waste (Roboflow project credit)",
            source_url=f"{MEDBIN}/images/{key}", original_url=url, download_urls=[url],
            license=CC_BY_40, publisher_partition=row["split"],
            annotations=dict(format="universe-class-counts", verified=False,
                             class_counts=dict(count), boxes=None),
            intent="positive" if count["syringe"] else "lookalike",
        )


def dental_archive():
    path = INPUTS / "dental-v1-yolov11.zip"
    if not path.exists():
        draft = path.with_suffix(".part")
        if dental.TARGET.exists():
            copyfile(dental.TARGET, draft)
        else:
            with ZipFile(dental.RangeSource()) as archive, archive.open(dental.MEMBER) as source, draft.open("wb") as out:
                copyfileobj(source, out)
        if digest(draft) != DENTAL_SHA:
            raise ValueError("Unexpected Mendeley dental export checksum")
        draft.replace(path)
    if digest(path) != DENTAL_SHA:
        raise ValueError("Unexpected Mendeley dental export checksum")
    return path


def dental_candidates(archive):
    names = yaml.safe_load(archive.read("data.yaml"))["names"]
    all_members = set(archive.namelist())
    for member in sorted(all_members):
        path = PurePosixPath(member)
        if path.parent.name != "images" or path.suffix.lower() not in (".jpeg", ".jpg", ".png"):
            continue
        label_member = str(path.parent.parent / "labels" / (path.stem + ".txt"))
        if label_member not in all_members:
            continue
        objects = [dict(label=names[key], bbox=box) for key, box in dental.labels(archive.read(label_member).decode())]
        positive = any("aspirating syringe" in o["label"].casefold() for o in objects)
        if not positive and not any(word in o["label"].casefold() for word in DENTAL_NEGATIVES for o in objects):
            continue
        parent = path.name.split(".rf.", 1)[0]
        yield dict(
            source="mendeley-dental-v1", id="dental-" + sha256(member.encode()).hexdigest()[:16],
            source_id=parent, parent=parent, archive_member=member,
            group="mendeley-dental-instruments-v1", author=DENTAL_AUTHOR,
            source_url="https://data.mendeley.com/datasets/mh6zpc7y9g/1",
            original_url=dental.API, license=CC_BY_40, publisher_partition=path.parts[0],
            annotations=dict(format="yolo-export", boxes=objects, verified=False),
            intent="positive" if positive else "lookalike",
        )


def commons_query(params):
    common = dict(action="query", prop="imageinfo", iiprop="url|size|mime|extmetadata|timestamp|sha1",
                  iiurlwidth=1280, format="json")
    result = json.loads(request(API + "?" + urlencode(common | params), 16 * 1024**2))
    if "error" in result:
        raise ValueError(result["error"]["code"])
    return result


def commons_candidates():
    snapshot = INPUTS / "commons-catalog.json"
    if not snapshot.exists():
        rows = {}
        searches = [("positive", "named detail seeds", dict(titles="|".join("File:" + n for n in COMMONS_SEEDS)))]
        searches += [
            (role, category, dict(generator="categorymembers", gcmtitle="Category:" + category,
                                  gcmtype="file", gcmlimit=50))
            for role, categories in COMMONS_CATEGORIES.items() for category in categories
        ]
        searches += [
            ("positive", term, dict(generator="search", gsrsearch=term, gsrnamespace=6, gsrlimit=50))
            for term in ('intitle:syringe "plunger"', 'intitle:syringe "liquid"', 'intitle:syringes "prefilled"')
        ]
        unavailable = []
        rate_limited = False
        for role, label, params in searches:
            continuation = {}
            for _ in range(2 if params.get("generator") == "categorymembers" else 1):
                try:
                    response = commons_query(params | continuation)
                except (OSError, ValueError) as error:
                    unavailable.append({"query": label, "error": str(error)})
                    rate_limited = isinstance(error, HTTPError) and error.code == 429
                    break
                for page in response.get("query", {}).get("pages", {}).values():
                    info = metadata(page, label, role)
                    if not info or not info["license"].startswith(("https://creativecommons.org/", "http://creativecommons.org/")):
                        continue
                    properties = page["imageinfo"][0]
                    ext = properties.get("extmetadata", {})
                    urls = [info["original_url"], info["thumbnail_url"]]
                    if not info["size"] or info["size"] > MAX_BYTES or max(info["source_dimensions"]) > 6500:
                        urls.reverse()
                    key = info.pop("id")
                    if key in rows:
                        rows[key]["discovery"].append(label)
                        continue
                    rows[key] = dict(
                        **info, id=f"commons-{key}", source_id=key, source="wikimedia-commons",
                        discovery=[label], download_urls=[u for u in urls if u], annotations=None,
                        credit=plain(ext.get("Credit", {}).get("value", "")) or None,
                        publisher_description=plain(ext.get("ImageDescription", {}).get("value", ""))[:800] or None,
                        photographer_date=plain(ext.get("DateTimeOriginal", {}).get("value", "")) or None,
                        publisher_sha1=properties.get("sha1"),
                    )
                continuation = response.get("continue")
                if not continuation:
                    break
            print(f"Commons {label}: {len(rows)} candidates so far", flush=True)
            if rate_limited:
                break
        # Supplement live queries with the repository's previously acquired public API catalogue.
        historical_catalog = HILL / "public" / "catalog.json"
        if historical_catalog.exists():
            for batch in read_json(historical_catalog).values():
                for item in batch:
                    if item["id"] in rows or not item["license"].startswith(
                        ("https://creativecommons.org/", "http://creativecommons.org/")
                    ):
                        continue
                    key = item["id"]
                    urls = [item["original_url"], item["thumbnail_url"]]
                    if not item["size"] or item["size"] > MAX_BYTES or max(item["source_dimensions"]) > 6500:
                        urls.reverse()
                    rows[key] = dict(
                        item, id=f"commons-{key}", source_id=key, source="wikimedia-commons",
                        intent="positive" if item["intent"] == "syringe" else "lookalike",
                        discovery=[item["category"]], download_urls=[u for u in urls if u],
                        annotations=None, credit=None, publisher_description=None, photographer_date=None,
                        publisher_sha1=None,
                    )
        write_json(snapshot, dict(rows=list(rows.values()), api_errors=unavailable))
    return read_json(snapshot)["rows"]


def historical():
    known = prior_exposure()
    known["dental_parents"] = set()
    scanned = list(Path("artifacts/detection_hillclimb").glob("*.json"))
    for path in sorted(scanned):
        for row in records(read_json(path)):
            known["ids"].add(str(row["id"]))
            known["urls"].update(url_key(row[k]) for k in ("source_url", "original_url", "image_url", "download_url")
                                 if row.get(k))
            known["hashes"].update(row[k] for k in ("sha256", "derived_sha256") if row.get(k))
            if re.fullmatch(r"[a-fA-F0-9]{16}", str(row.get("dhash"))):
                known["difference"].add(int(row["dhash"], 16))
            if row.get("source_parent") and ("dental" in path.name or str(row.get("group", "")).startswith("mendeley-dental")):
                known["dental_parents"].add(row["source_parent"])
    # These bytes are only read to deny a match. All acquisition parents are public allowlisted sources.
    for folder in (DATA / "original", DATA / "original-color", HILL / "medbin-original",
                   HILL / "catalog-prospect", DATA / "openimages" / "test"):
        for path in sorted(folder.iterdir()) if folder.is_dir() else ():
            if path.suffix.casefold() in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"):
                known["ids"].add(path.stem)
                known["hashes"].add(digest(path))
                try:
                    known["difference"].add(int(dhash(read_image(path)), 16))
                except OSError:
                    continue
    return known


def history_reason(row, known):
    if row["source"] == "mendeley-dental-v1":
        return "historical parent" if row["parent"] in known["dental_parents"] else None
    if row["source_id"] in known["ids"]:
        return "historical id"
    # Older MedBin rows cite the project root rather than their file page.
    if any(url_key(row[k]) in known["urls"] for k in ("source_url", "original_url")):
        return "historical url"
    if row["source"] == "wikimedia-commons" and lexical_key(row["title"]) in known["titles"]:
        return "historical title"
    return None


def get_blob(row, archive):
    if row.get("archive_member"):
        return archive.read(row["archive_member"]), row["original_url"]
    for url in row["download_urls"]:
        try:
            return request(url, MAX_BYTES), url
        except (OSError, ValueError):
            continue
    return None, None


def materialize(row, blob, url):
    if not blob:
        return None
    try:
        with Image.open(BytesIO(blob)) as original:
            if original.width * original.height > MAX_PIXELS:
                return None
            raster = ImageOps.exif_transpose(original).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if min(raster.size) < 180:
        return None
    width, height = raster.size
    miniature = ImageOps.contain(raster, (SIDE, SIDE), Image.Resampling.LANCZOS)
    miniature.info.clear()
    encoded = BytesIO()
    miniature.save(encoded, "JPEG", quality=94)
    payload = encoded.getvalue()
    path = OUT / "images" / (row["id"] + ".jpg")
    row = dict(row, image=path.relative_to(DATA).as_posix(), download_url=url,
               upstream_sha256=sha256(blob).hexdigest(), sha256=sha256(payload).hexdigest(),
               dhash=dhash(read_image(BytesIO(payload))), upstream_size=[width, height],
               size=list(miniature.size), attributes=dict.fromkeys(ATTRIBUTES))
    return row, miniature, path, payload


def ordered(candidates):
    def priority(row):
        if row["source"] == "wikimedia-commons":
            return (
                "named detail seeds" not in row["discovery"],
                bool(re.search(r"logo|diagram|illustration|poster|icon|patent|route map", row["title"], re.I)),
                rank(row["id"]),
            )
        if row["source"] == "mendeley-dental-v1":
            return (row["publisher_partition"] == "train", rank(row["id"]))
        return (rank(row["id"]),)
    return sorted(candidates, key=priority)


def stage(candidates, archive, known, prior_fingerprints, selected_hashes, limited, rejected):
    counts = Counter()
    parents = set()
    pending = []
    for candidate in ordered(candidates):
        reason = history_reason(candidate, known)
        if reason:
            rejected[reason] += 1
        else:
            pending.append(candidate)
    rows = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        # Fixed batches bound queued downloads and preserve the ranking.
        for start in range(0, len(pending), 15):
            batch = [r for r in pending[start:start + 15] if counts[r["intent"]] < limited[r["intent"]]]
            blobs = pool.map(lambda r: get_blob(r, archive), batch)
            for candidate, (blob, url) in zip(batch, blobs, strict=True):
                rejected["examined assets"] += 1
                role = candidate["intent"]
                parent = candidate.get("parent")
                if counts[role] >= limited[role] or parent in parents:
                    rejected["quota or duplicate filename parent"] += 1
                    continue
                prepared = materialize(candidate, blob, url)
                if prepared is None:
                    rejected["unavailable or invalid raster"] += 1
                    continue
                row, raster, path, payload = prepared
                if row["upstream_sha256"] in known["hashes"] or near_any(raster, prior_fingerprints):
                    rejected["historical exact or near image"] += 1
                    continue
                if row["upstream_sha256"] in selected_hashes["sha256"] or near_any(raster, selected_hashes["dhash"]):
                    rejected["internal exact or near image"] += 1
                    continue
                draft = path.with_suffix(".part")
                draft.write_bytes(payload)
                draft.replace(path)
                selected_hashes["sha256"].add(row["upstream_sha256"])
                selected_hashes["dhash"].add(int(row["dhash"], 16))
                if parent:
                    parents.add(parent)
                counts[role] += 1
                rows.append(row)
    return rows


def summary(rows):
    result = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        result[source] = dict(
            rows=len(subset), groups=len({r["group"] for r in subset}),
            intent=dict(Counter(r["intent"] for r in subset)),
            with_boxes=sum(bool((r["annotations"] or {}).get("boxes")) for r in subset),
            boxes=sum(len((r["annotations"] or {}).get("boxes") or []) for r in subset),
            with_class_counts=sum(bool((r["annotations"] or {}).get("class_counts")) for r in subset),
        )
    return result


def apply_review(manifest):
    rows = {r["id"]: r for r in manifest["rows"]}
    entries = read_json(REVIEW)["rows"] if REVIEW.exists() else {}
    coverage = {attribute: Counter() for attribute in ATTRIBUTES}
    usefulness = Counter()
    tags = Counter()
    roles = {}
    for row in rows.values():
        row["attributes"] = dict.fromkeys(ATTRIBUTES)
        row.pop("visual_review", None)
    for identifier, entry in entries.items():
        row = rows[identifier]
        attrs = entry["attributes"]
        if (entry["sha256"] != row["sha256"] or not set(attrs) <= set(ATTRIBUTES)
            or any(v is not None and not isinstance(v, str if k == "syringe_type" else bool)
                   for k, v in attrs.items())):
            raise ValueError(f"Invalid visual review binding for {identifier}")
        if entry["scene_role"] not in ("syringe", "hard-negative", "detail-only", "reject"):
            raise ValueError(f"Invalid scene role for {identifier}")
        row["attributes"].update(attrs)
        row["visual_review"] = entry
        usefulness[entry["scene_role"]] += 1
        roles[identifier] = entry["scene_role"]
        if entry["scene_role"] != "reject":
            tags.update(set(entry.get("tags", [])))
        for attribute, value in row["attributes"].items():
            if value is not None:
                coverage[attribute]["observed"] += 1
                coverage[attribute]["positive"] += int(bool(value))
    manifest["visual_review_summary"] = dict(
        examined=len(entries), useful=sum(count for role, count in usefulness.items() if role != "reject"),
        useful_groups=len({rows[k]["group"] for k, role in roles.items() if role != "reject"}),
        scene_roles=dict(usefulness), scene_tags=dict(sorted(tags.items())),
        attribute_coverage={k: dict(non_null=v["observed"], positive=v["positive"],
                                    unknown=len(rows) - v["observed"]) for k, v in coverage.items()},
    )
    return manifest


def contact_sheets(rows, folder):
    folder.mkdir(exist_ok=True)
    for start in range(0, len(rows), 24):
        surface = Image.new("RGB", (1440, 880), "#efefef")
        drawer = ImageDraw.Draw(surface)
        for i, row in enumerate(rows[start:start + 24]):
            x, y = 240 * (i % 6), 220 * (i // 6)
            picture = ImageOps.contain(read_image(DATA / row["image"]), (232, 190))
            left = x + (240 - picture.width) // 2
            top = y + (190 - picture.height) // 2
            surface.paste(picture, (left, top))
            drawer.text((x + 5, y + 192), row["id"], fill="#202020")
        surface.save(folder / f"{start // 24 + 1:02}.jpg", quality=88)


def collect():
    if MANIFEST.exists():
        verify()
        return
    INPUTS.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    known = historical()
    manifest = dict(schema_version=1, seed=SEED, rows=[], screening={}, history={
        key: len(values) for key, values in known.items()
    })
    selected_hashes = {"sha256": set(), "dhash": set()}
    with ZipFile(dental_archive()) as archive:
        for name, get_candidates, limits in (
            ("mendeley-dental-v1", lambda: dental_candidates(archive), {"positive": 50, "lookalike": 80}),
            ("open-images", open_images, {"positive": 120, "lookalike": 20}),
            ("medbin-universe", medbin, {"positive": 90, "lookalike": 75}),
            ("wikimedia-commons", commons_candidates, COMMONS_LIMITS),
        ):
            candidates = list(get_candidates())
            counts = Counter()
            rows = stage(candidates, archive, known, known["difference"], selected_hashes, limits, counts)
            manifest["rows"].extend(rows)
            manifest["screening"][name] = dict(metadata_candidates=len(candidates), **counts)
            print(f"{name}: {len(rows)} retained from {len(candidates)} metadata candidates", flush=True)
    manifest["inputs_sha256"] = {p.name: digest(p) for p in sorted(INPUTS.iterdir()) if p.is_file()}
    manifest["source_summary"] = summary(manifest["rows"])
    apply_review(manifest)
    by_source = defaultdict(list)
    for row in manifest["rows"]:
        by_source[row["source"]].append(row)
    folder = OUT / "sheets"
    folder.mkdir(exist_ok=True)
    for source, rows in by_source.items():
        contact_sheets(rows, folder / source)
    write_json(MANIFEST, manifest)


def verify():
    manifest = read_json(MANIFEST)
    ids = set()
    hashes = set()
    for row in manifest["rows"]:
        path = DATA / row["image"]
        if (row["id"] in ids or row["sha256"] in hashes or
            path != OUT / "images" / (row["id"] + ".jpg") or
            digest(path) != row["sha256"] or dhash(read_image(path)) != row["dhash"]):
            raise ValueError(f"Invalid candidate {row['id']}")
        annotation = row["annotations"] or {}
        if (not all(row[k] for k in ("group", "source_url", "original_url", "author", "license"))
            or row["source"] not in ("mendeley-dental-v1", "open-images", "medbin-universe", "wikimedia-commons")
            or "split" in row or annotation.get("verified")
            or set(row["attributes"]) != set(ATTRIBUTES)
            or any(not valid_box(b["bbox"]) for b in (annotation.get("boxes") or []))):
            raise ValueError(f"Invalid attributes or box for {row['id']}")
        ids.add(row["id"])
        hashes.add(row["sha256"])
    for name, checksum in manifest["inputs_sha256"].items():
        if digest(INPUTS / name) != checksum:
            raise ValueError(f"Source snapshot changed: {name}")
    if summary(manifest["rows"]) != manifest["source_summary"] or apply_review(manifest) != read_json(MANIFEST):
        raise ValueError("Manifest summaries or visual review changed")
    print(f"{len(ids)} thumbnails verified, {len({r['group'] for r in manifest['rows']})} provenance groups")
    print(json.dumps(manifest["visual_review_summary"], ensure_ascii=False))


def refresh_review():
    write_json(MANIFEST, apply_review(read_json(MANIFEST)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "apply-review", "verify"))
    action = parser.parse_args().action
    {"collect": collect, "apply-review": refresh_review, "verify": verify}[action]()
