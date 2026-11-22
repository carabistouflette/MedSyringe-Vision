"""Collect and review original photo assets from licensed public dataset catalogs."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import unescape
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
import re
import signal
import threading
import time

from PIL import Image, ImageDraw, ImageOps
import requests

from medsyringe.data import dhash, digest, read_image, read_json, write_json


DATA = Path("data")
ROOT = DATA / "public-bulk-syringe-collection"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
UNIVERSE = "https://universe.roboflow.com/"
PROJECTS = {
    "syringe-det": ("syringe-project/syringe-det", "syringe project", CC_BY),
    "biomedical-waste": (
        "biowastemanagement/biomedical-waste-detection-nv7fq-bpeeu", "Biowastemanagement", CC_BY
    ),
    "pharmaceutical-waste": (
        "medical-waste-disposal/pharmaceutical-biomedical-waste", "Medical Waste Disposal",
        "Public Domain (publisher declaration)",
    ),
    "vials-syringe": ("vial-0no6n/vials_syringe", "vial", CC_BY),
    "syringe-vial": ("syringelines/syringevial", "syringelines", CC_BY),
    "syringe-plus": ("vialsyringe/syringe_plus", "vialSyringe", CC_BY),
    "syringe-lines": ("line-segmentation/syringe-lines", "Line Segmentation", CC_BY),
    "syringe-project": ("michaels-workspace-ldsqh/syringe-project", "Michaels Workspace", CC_BY),
    "syringe-object": ("angels-workspace-busje/syringe-object-detection-gg123", "Angels Workspace", CC_BY),
    "syringe-needle": ("syringe-16nka/syringe-vyhfk", "Syringe", CC_BY),
    "mmt-waste": ("mmt-kcsv2/medical-waste-yanoy", "MMT", CC_BY),
    "uea-waste": ("uea-ltv8w/medical-waste-bahxi", "UEA", CC_BY),
    "top-syringe": ("tals-demo/top-syringe-detection", "Tals Demo", CC_BY),
    "medicine-syringe": ("medicine-gi9mg/syringe-aj8no-rnagc", "Medicine", CC_BY),
    "syringe-seg": ("ws-1jgjc/syringe-rpizw", "ws", CC_BY),
    "syringe-test": ("test-uugy6/syringe-test", "test", CC_BY),
    "vial-counting": ("vial-counting/syringes-and-vials-count", "vial counting", CC_BY),
    "syringe-poly": ("cs5800/data_set_syringe_polygon", "CS5800", CC_BY),
}
SYRINGE_CLASSES = {
    "syringe", "syringes", "used syringe", "syringe-needle", "pfs", "barrel", "syringe_body",
    "test-syringe", "syring",
}

SHARED_GROUPS = {
    "biomedical-waste": "shared-biomedical-waste",
    "pharmaceutical-waste": "shared-biomedical-waste",
    "syringe-det": "shared-syringe-object-catalogs",
    "syringe-plus": "shared-syringe-object-catalogs",
    "vials-syringe": "shared-syringe-object-catalogs",
    "syringe-lines": "shared-syringe-object-catalogs",
    "syringe-project": "michaels-workspace-ldsqh/syringe-detection-fnu4f",
}
HISTORY = [
    DATA / "public-annotated-collection/manifest.json",
    DATA / "public-web-collection/manifest.json",
    DATA / "detection_hillclimb/device-context-staged.json",
    Path("artifacts/detection_hillclimb/device_scene_manifest.json"),
    Path("artifacts/detection_hillclimb/device_context_manifest.json"),
    Path("artifacts/detection_hillclimb/medbin_held_out_manifest.json"),
    Path("artifacts/detection_hillclimb/openimages_additional_manifest.json"),
    Path("artifacts/zenodo_training_manifest.json"),
]
MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 32_000_000
HASH_RADIUS = 4
RASTER_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def catalog(source):
    directory = ROOT / "catalog" / source
    rows = {}
    total = None
    for path in sorted(directory.glob("*.json"), key=lambda path: int(path.stem)):
        page = read_json(path)
        total = page["total"]
        if page["offset"] != len(rows):
            raise ValueError(f"Missing or out-of-order catalog page in {directory}")
        for entry in page["results"]:
            if entry["id"] in rows:
                raise ValueError(f"Repeated catalog ID in {path}: {entry['id']}")
            rows[entry["id"]] = entry
    return rows, total


def publisher_classes(entry):
    annotations = entry.get("annotations")
    return annotations.get("classes", {}) if isinstance(annotations, dict) else {}


def publisher_syringes(entry):
    return sum(count for name, count in publisher_classes(entry).items()
               if name.casefold() in SYRINGE_CLASSES)


def publisher_positive(entry):
    return bool(publisher_syringes(entry)) or any(
        label.casefold() in SYRINGE_CLASSES for label in entry.get("labels", [])
    )


def native_url(entry):
    owner, image_id = entry["owner"], entry["id"]
    if not all(re.fullmatch(r"[a-zA-Z0-9]+", value) for value in (owner, image_id)):
        raise ValueError("Invalid catalog asset identifiers")
    expected = f"https://source.roboflow.com/{owner}/{image_id}/original.jpg"
    if entry["url"] != expected:
        raise ValueError(f"Unexpected public original URL for {image_id}")
    return expected


class DownloadPacer:
    def __init__(self, interval):
        self.interval = interval
        self.next_request = 0
        self.lock = threading.Lock()
        self.stopped = False

    def wait(self):
        # Serialize request starts across workers while allowing transfers to overlap.
        with self.lock:
            if self.stopped:
                return False
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + self.interval
            return True

    def stop(self):
        with self.lock:
            self.stopped = True


def download(task, pacer):
    source, entry = task
    try:
        url = native_url(entry) if source in PROJECTS else entry["original_url"]
    except ValueError:
        return source, "unsupported asset URL", entry["id"]
    target = ROOT / "images" / source / entry["id"]
    if any(target.with_suffix(suffix).exists() for suffix in RASTER_FORMATS.values()):
        return source, "cached", entry["id"]
    if not pacer.wait():
        return source, "paused", entry["id"]
    try:
        with requests.get(
            url, stream=True, timeout=(8, 25), allow_redirects=False,
            headers={"User-Agent": "MedSyringe-Vision/1.0 (public research photo collection)"},
        ) as response:
            if response.status_code in (429, 503):
                pacer.stop()
                retry = response.headers.get("Retry-After")
                if retry and not retry.isdigit():
                    retry = parsedate_to_datetime(retry).astimezone(timezone.utc).isoformat()
                return source, f"HTTP {response.status_code}; Retry-After={retry}", entry["id"]
            response.raise_for_status()
            if response.status_code != 200:
                return source, f"HTTP {response.status_code}", entry["id"]
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > MAX_BYTES:
                    return source, "byte limit", entry["id"]
        with Image.open(BytesIO(content)) as image:
            suffix = RASTER_FORMATS.get(image.format)
            if not suffix or image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
                return source, "format or pixel limit", entry["id"]
            image.verify()
        read_image(BytesIO(content))
        atomic_bytes(target.with_suffix(suffix), content)
        return source, "downloaded", entry["id"]
    except requests.HTTPError as error:
        return source, f"HTTP {error.response.status_code}", entry["id"]
    except (requests.RequestException, OSError, ValueError, Image.DecompressionBombError) as error:
        return source, type(error).__name__, entry["id"]


def fetch(sources, positives, controls, interval):
    tasks_by_source = {}
    for source in sources:
        rows, total = catalog(source)
        ranked = sorted(rows.values(), key=lambda entry: sha256(entry["id"].encode()).digest())
        positive = [entry for entry in ranked if publisher_positive(entry)]
        others = [entry for entry in ranked if not publisher_positive(entry)]
        tasks_by_source[source] = positive[:positives] + others[:controls]
        print(source, f"catalog {len(rows)}/{total}", f"labeled positives {len(positive)}",
              f"queued {len(tasks_by_source[source])}", flush=True)
    # Give each source a turn so an interrupted run still provides diverse material.
    tasks = [(source, entry) for batch in zip_longest(*tasks_by_source.values())
             for source, entry in zip(tasks_by_source, batch) if entry is not None]
    pacer = DownloadPacer(interval)
    # A requested stop leaves completed originals available for offline review.
    previous = signal.signal(signal.SIGTERM, lambda *_: setattr(pacer, "stopped", True))
    counts = defaultdict(Counter)
    errors = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, (source, status, image_id) in enumerate(
            executor.map(lambda task: download(task, pacer), tasks), start=1
        ):
            counts[source][status] += 1
            if status not in {"downloaded", "cached", "paused"}:
                errors.append(dict(source=source, id=image_id, result=status))
            if index % 100 == 0:
                print(f"{index}/{len(tasks)} completed", flush=True)
    signal.signal(signal.SIGTERM, previous)
    now = datetime.now(timezone.utc)
    write_json(ROOT / "fetch-runs" / f"{now:%Y%m%dT%H%M%S%fZ}.json", {
        "results": counts, "errors": errors, "seconds_between_starts": interval,
    })
    print(dict(counts), flush=True)


def page_value(html, pattern):
    match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
    return unescape(re.sub(r"<[^>]+>", "", match[1])).strip() if match else ""


def official_entry(html, url):
    dvids = re.fullmatch(r"https://www\.dvidshub\.net/image/(\d+)/[a-z0-9-]+", url)
    if dvids:
        image_id = dvids[1]
        if '<h4 class="public-domain">' not in html:
            return None
        asset = page_value(html, r'<meta property="og:image" content="([^"]+)"')
        author = page_value(html, r'<h3>Photo by <a [^>]+>([^<]+)</a>')
        virin = page_value(html, r"<td>VIRIN:</td>\s*<td>([^<]+)</td>")
        caption = page_value(html, r'<meta property="og:description" content="([^"]+)"')
        if not re.fullmatch(
            rf"https://d1ldvf68ux039x\.cloudfront\.net/thumbs/photos/\d+/{image_id}/1000w_q95\.jpg",
            asset,
        ) or not author or not re.fullmatch(r"\d{6}-[A-Z]-[A-Z0-9]+-\d+", virin):
            return None
        # The individual public-domain mark and an explicit U.S. agency attribution are required.
        if not re.search(r"\b(?:U\.S\.\s|DoD\b|Defense Department\b)", caption, re.IGNORECASE):
            return None
        return dict(id=image_id, source="dvids", source_url=url, original_url=asset,
                    author=f"{author} / DVIDS", group=f"dvids:{virin.rsplit('-', 1)[0]}",
                    license="Public Domain (U.S.)", caption=caption, virin=virin)
    cdc = re.fullmatch(r"https://wwwn\.cdc\.gov/phil/details\.aspx\?pid=(\d+)", url,
                       re.IGNORECASE)
    if not cdc:
        raise ValueError(f"Unsupported official photo page: {url}")
    image_id = cdc[1]
    restriction = page_value(html, r'<span id="lblCopyRight">(.*?)</span>')
    if not restriction.startswith("Permission is not required for public domain images"):
        return None
    asset = page_value(html, r'<img id="imgURL1"[^>]+src="([^"]+)"')
    if not re.fullmatch(rf"https://wwwn\.cdc\.gov/phil/+PHIL_Images/[\w/-]+/{image_id}_lores\.jpg",
                        asset, re.IGNORECASE):
        return None
    author = page_value(html, r'<span id="lblPhotoCredit">([^<]+)</span>')
    date = page_value(html, r'<span id="lblCretDt">([^<]+)</span>')
    caption = page_value(html, r'<span id="lblDesc">(.*?)</span>')
    return dict(id=image_id, source="cdc-phil", source_url=url, original_url=asset,
                author=f"CDC / {author}" if author else "CDC",
                group=f"cdc-phil:{author or 'CDC'}:{date or 'undated'}",
                license="Public Domain (CDC PHIL)", caption=caption)


def fetch_official():
    directory = ROOT / "catalog"
    output = directory / "official-assets.json"
    current = read_json(output) if output.exists() else {"rows": [], "attempts": {}}
    seen = {row["source_url"] for row in current["rows"]}
    pacer = DownloadPacer(1.0)
    for url in read_json(directory / "official-pages.json"):
        if url in seen:
            continue
        if not pacer.wait():
            break
        try:
            response = requests.get(url, timeout=(8, 15), allow_redirects=False)
            current["attempts"][url] = f"HTTP {response.status_code}"
            if response.status_code in (429, 503):
                pacer.stop()
                break
            if response.status_code != 200:
                continue
            entry = official_entry(response.text, url)
            if entry:
                current["rows"].append(entry)
                seen.add(url)
                write_json(output, current)
            else:
                current["attempts"][url] = "Photo-specific public-domain mark or credit not verified"
        except (requests.RequestException, ValueError) as error:
            current["attempts"][url] = type(error).__name__
    for row in current["rows"]:
        _, status, _ = download((row["source"], row), pacer)
        row["download_status"] = status
    write_json(output, current)
    print(f"{len(current['rows'])} rights-checked official records from "
          f"{len(current['attempts'])} pages", flush=True)


def manifest_row(source, entry, path, file_sha, fingerprint):
    if source in PROJECTS:
        slug, author, license_name = PROJECTS[source]
        provenance = dict(source_url=f"{UNIVERSE}{slug}", original_url=native_url(entry),
                          author=author, license=license_name,
                          group=f"roboflow:{SHARED_GROUPS.get(source, slug)}",
                          credit_basis="publisher attribution and dataset license")
    else:
        provenance = {key: entry[key] for key in
                      ("source_url", "original_url", "author", "license", "group", "caption")}
        provenance["credit_basis"] = "individual public photo page"
    return {
        "id": f"{source}-{entry['id']}",
        "image": str(path.relative_to(DATA)),
        "sha256": file_sha,
        "dhash": fingerprint["dhash"],
        **provenance,
        "source": source,
        "catalog_id": entry["id"],
        "publisher_partition": entry.get("split"),
        "publisher_classes": publisher_classes(entry),
        "publisher_labels": entry.get("labels", []),
        "publisher_positive": publisher_positive(entry) if source in PROJECTS else None,
        "publisher_syringe_instances": publisher_syringes(entry),
        "dimensions": fingerprint["dimensions"],
    }


def history():
    snapshot_path = ROOT / "history-index.json"
    if snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        return (snapshot["urls"], snapshot["sha256"], dict(snapshot["dhash"]), snapshot["manifests"])
    urls, hashes, perceptual = {}, {}, {}
    included = []
    for path in HISTORY:
        if not path.is_file():
            continue
        rows = read_json(path)["rows"]
        included.append({"manifest": str(path), "sha256": digest(path), "rows": len(rows)})
        for row in rows:
            key = row["id"]
            for field in ("original_url", "image_url", "download_url"):
                if row.get(field):
                    urls[row[field]] = key
            for field in ("sha256", "upstream_sha256"):
                if row.get(field):
                    hashes[row[field]] = key
            if re.fullmatch(r"[a-fA-F0-9]{16}", str(row.get("dhash"))):
                perceptual[int(row["dhash"], 16)] = key
    write_json(snapshot_path, {
        "urls": urls, "sha256": hashes, "dhash": sorted(perceptual.items()), "manifests": included,
    })
    return urls, hashes, perceptual, included


def oriented_hashes(raster):
    # Rotation or mirror checks only need the tiny grayscale image used for hashing.
    thumb = raster.convert("L").resize((96, 96), Image.Resampling.LANCZOS)
    for variant in (thumb, thumb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        for turns in (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                      Image.Transpose.ROTATE_270):
            yield int(dhash(variant.transpose(turns) if turns is not None else variant), 16)


class HashIndex:
    def __init__(self, known=()):
        self.buckets = defaultdict(dict)
        for value, key in known:
            self.add(value, key)

    @staticmethod
    def blocks(value):
        # At least one of five blocks must match for a distance of four or less.
        return ((shift, (value >> shift) & 8191) for shift in range(0, 64, 13))

    def add(self, value, key):
        for block in self.blocks(value):
            self.buckets[block][value] = key

    def near(self, variants):
        for value in sorted(variants):
            for block in self.blocks(value):
                for other, key in self.buckets.get(block, {}).items():
                    if (value ^ other).bit_count() <= HASH_RADIUS:
                        return key
        return None


def fingerprint_file(path, known):
    file_sha = digest(path)
    fingerprint = known.get(file_sha)
    if fingerprint is None:
        with Image.open(path) as original:
            ImageOps.exif_transpose(original, in_place=True)
            raster = original.convert("RGB")
        fingerprint = {
            "dhash": dhash(raster),
            "dimensions": list(raster.size),
            "oriented": sorted(set(oriented_hashes(raster))),
        }
    return path, file_sha, fingerprint


def inspect(only=None):
    urls, known_shas, prior_hashes, included = history()
    duplicates, reviewable, statistics = [], [], {}
    seen_shas, seen_hashes = {}, HashIndex()
    prior_hashes = HashIndex(prior_hashes.items())
    fingerprint_path = ROOT / "fingerprints.json"
    fingerprints = read_json(fingerprint_path) if fingerprint_path.exists() else {}
    reviews = read_reviews()
    sources = {}
    official_manifest = ROOT / "catalog" / "official-assets.json"
    if official_manifest.exists():
        official_rows = read_json(official_manifest)["rows"]
        for source in ("dvids", "cdc-phil"):
            entries = {entry["id"]: entry for entry in official_rows if entry["source"] == source}
            sources[source] = entries, len(entries)
    priority = ("top-syringe", "syringe-object", "syringe-test", "syringe-lines",
                "syringe-project", "syringe-needle", "syringe-seg")
    sources.update((source, catalog(source)) for source in dict.fromkeys((*priority, *PROJECTS)))
    for source, (rows, total) in sources.items():
        if only and source not in only:
            continue
        held = [path for path in (ROOT / "images" / source).glob("*")
                if path.suffix in RASTER_FORMATS.values() and path.stem in rows]
        if not rows and not held:
            continue
        counts = Counter()
        counts["catalog_rows"] = len(rows)
        counts["catalog_total"] = total
        counts["publisher_positive_images"] = sum(publisher_positive(entry) for entry in rows.values())
        counts["publisher_positive_instances"] = sum(publisher_syringes(entry) for entry in rows.values())
        # Keep a verified representative when later downloads add copies of it.
        held.sort(key=lambda path: (
            reviews.get(f"{source}-{path.stem}", {}).get("visual_status") != "positive",
            not publisher_positive(rows[path.stem]),
            path.stem,
        ))
        with ThreadPoolExecutor(max_workers=4) as executor:
            scored = list(executor.map(lambda path: fingerprint_file(path, fingerprints), held))
        for path, file_sha, fingerprint in scored:
            fingerprints[file_sha] = fingerprint
            row = manifest_row(source, rows[path.stem], path, file_sha, fingerprint)
            counts["downloaded"] += 1
            variants = {int(row["dhash"], 16), *fingerprint["oriented"]}
            match = duplicate_match(row, variants, urls, known_shas, prior_hashes,
                                    seen_shas, seen_hashes)
            if match:
                reason, key = match
                duplicates.append(dict(row, disposition=reason, matched_id=key))
                counts[reason] += 1
                continue
            reviewable.append(row)
            seen_shas[row["sha256"]] = row["id"]
            seen_hashes.add(int(row["dhash"], 16), row["id"])
        statistics[source] = counts
        print(source, f"{counts['downloaded']} local originals recorded", flush=True)
    write_json(fingerprint_path, fingerprints)
    return reviewable, duplicates, statistics, included


def stage(sources=None):
    rows, duplicates, statistics, included = inspect(sources)
    write_json(ROOT / "duplicate-log.json", {"rows": duplicates, "public_history": included})
    write_json(ROOT / "inventory.json", {"rows": rows, "sources": statistics, "scoped": bool(sources)})
    print(f"{len(rows)} review candidates, {len(duplicates)} automated exclusions", flush=True)


def duplicate_match(row, variants, urls, known_shas, prior_hashes, seen_shas, seen_hashes):
    for reason, key in (
        ("historical_url", urls.get(row["original_url"])),
        ("historical_exact", known_shas.get(row["sha256"])),
        ("exact", seen_shas.get(row["sha256"])),
    ):
        if key is not None:
            return reason, key
    for reason, index in (("historical_near", prior_hashes), ("near", seen_hashes)):
        hit = index.near(variants)
        if hit is not None:
            return reason, hit
    return None


def sheets(sources=None):
    inventory = read_json(ROOT / "inventory.json")["rows"]
    directory = ROOT / "sheets"
    directory.mkdir(exist_ok=True)
    by_source = defaultdict(list)
    previous = read_reviews()
    for row in inventory:
        if row["id"] not in previous and (not sources or row["source"] in sources):
            by_source[row["source"]].append(row)
    width, height, columns, lines = 168, 138, 7, 7
    page_size = columns * lines
    for source, rows in by_source.items():
        # Make each sheet's name content-bound so later downloads cannot change its index.
        for offset in range(0, len(rows), page_size):
            batch = rows[offset:offset + page_size]
            suffix = sha256(" ".join(row["id"] for row in batch).encode()).hexdigest()[:10]
            stem = f"{source}-{offset // page_size + 1:03}-{suffix}"
            board = Image.new("RGB", (columns * width, ((len(batch) + columns - 1) // columns) * height),
                              "#f0f0f0")
            draw = ImageDraw.Draw(board)
            for index, row in enumerate(batch):
                x, y = index % columns * width, index // columns * height
                with Image.open(DATA / row["image"]) as original:
                    original.draft("RGB", (width * 2, height * 2))
                    raster = ImageOps.exif_transpose(original).convert("RGB")
                raster.thumbnail((width - 6, height - 22), Image.Resampling.LANCZOS)
                board.paste(raster, (x + (width - raster.width) // 2, y + 2))
                claimed = {None: "?", True: "+", False: "-"}[row["publisher_positive"]]
                draw.text((x + 4, y + height - 17), f"{index + 1}  {claimed}", fill="#111")
            board.save(directory / f"{stem}.jpg", quality=88)
            write_json(directory / f"{stem}.json", {
                "rows": [{"id": row["id"], "sha256": row["sha256"]} for row in batch],
            })
    print({source: len(rows) for source, rows in by_source.items()})


def read_reviews():
    reviews = {}
    for path in sorted((ROOT / "reviews").glob("*.json")):
        for row in read_json(path)["rows"]:
            if row["id"] in reviews and reviews[row["id"]] != row:
                raise ValueError(f"Conflicting visual reviews for {row['id']}")
            reviews[row["id"]] = row
    return reviews


def parse_indices(value):
    indices = set()
    for token in value.split(","):
        if not token:
            continue
        first, _, last = token.partition("-")
        indices.update(range(int(first), int(last or first) + 1))
    return indices


def review(sheet_name, positive, negative, ambiguous, burst, excluded):
    if Path(sheet_name).name != sheet_name:
        raise ValueError("Use the local sheet basename")
    rows = read_json(ROOT / "sheets" / f"{sheet_name}.json")["rows"]
    decisions = {}
    for disposition, value in (("positive", positive), ("negative", negative),
                               ("ambiguous", ambiguous), ("burst", burst), ("excluded", excluded)):
        for index in parse_indices(value):
            if index in decisions or index < 1 or index > len(rows):
                raise ValueError(f"Invalid or repeated contact-sheet index: {index}")
            decisions[index] = disposition
    reviewed = [dict(rows[index - 1], visual_status=disposition, sheet=sheet_name, cell=index)
                for index, disposition in sorted(decisions.items())]
    write_json(ROOT / "reviews" / f"{sheet_name}.json", {"rows": reviewed})


def publish():
    inventory = read_json(ROOT / "inventory.json")
    if inventory.get("scoped"):
        raise ValueError("Stage all sources before publishing the collection")
    reviews = read_reviews()
    accepted, reviewed_out = [], []
    for row in inventory["rows"]:
        decision = reviews.get(row["id"])
        if decision and decision["sha256"] != row["sha256"]:
            raise ValueError(f"Review refers to different bytes for {row['id']}")
        status = decision["visual_status"] if decision else "unreviewed"
        row = dict(row, visual_status=status)
        if decision:
            row.update(review_sheet=decision["sheet"], review_cell=decision["cell"])
        inventory["sources"][row["source"]][status] = (
            inventory["sources"][row["source"]].get(status, 0) + 1
        )
        if status == "positive":
            exact = digest(DATA / row["image"])
            if exact != row["sha256"]:
                raise ValueError(f"File changed after review: {row['id']}")
            accepted.append(row)
        else:
            reviewed_out.append(row)
    write_json(ROOT / "review-inventory.json", {"rows": reviewed_out})
    write_json(ROOT / "manifest.json", {
        "scope": "Visually confirmed original public syringe photographs after duplicate screening.",
        "rows": accepted,
        "sources": inventory["sources"],
        "positive_count": len(accepted),
        "provisional_group_count": len({row["group"] for row in accepted}),
        "dhash_radius": HASH_RADIUS,
    })
    print(f"{len(accepted)} verified positives; {len(reviewed_out)} other downloaded candidates")


def verify_fingerprint(row):
    path = DATA / row["image"]
    images = ROOT / "images"
    if not path.is_relative_to(images) or not path.resolve().is_relative_to(images.resolve()):
        raise ValueError(f"Invalid image path: {path}")
    _, file_sha, fingerprint = fingerprint_file(path, {})
    return file_sha, fingerprint


def verify():
    manifest = read_json(ROOT / "manifest.json")
    rows = manifest["rows"]
    ids, hashes, perceptual = set(), set(), HashIndex()
    reviews = read_reviews()
    with ThreadPoolExecutor(max_workers=4) as executor:
        verified = list(executor.map(verify_fingerprint, rows))
    for row, (file_sha, fingerprint) in zip(rows, verified, strict=True):
        if not (file_sha == row["sha256"] == reviews[row["id"]]["sha256"]
                and fingerprint["dhash"] == row["dhash"]
                and fingerprint["dimensions"] == row["dimensions"]
                and reviews[row["id"]]["visual_status"] == row["visual_status"] == "positive"):
            raise ValueError(f"Integrity or review mismatch: {row['id']}")
        if row["id"] in ids or row["sha256"] in hashes:
            raise ValueError(f"Repeated ID or SHA-256: {row['id']}")
        if not all(row.get(key) for key in ("group", "source_url", "original_url", "author", "license")):
            raise ValueError(f"Missing source metadata: {row['id']}")
        variants = {int(row["dhash"], 16), *fingerprint["oriented"]}
        if perceptual.near(variants) is not None:
            raise ValueError(f"Perceptual duplicate: {row['id']}")
        ids.add(row["id"])
        hashes.add(row["sha256"])
        perceptual.add(int(row["dhash"], 16), row["id"])
    if (manifest["positive_count"] != len(rows)
            or manifest["provisional_group_count"] != len({row["group"] for row in rows})):
        raise ValueError("Manifest totals do not match rows")
    print(f"Verified {len(rows)} reviewed images, {len(hashes)} unique SHA-256 values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    downloader = commands.add_parser("fetch")
    downloader.add_argument("--source", choices=PROJECTS, action="append")
    downloader.add_argument("--positives-per-source", type=int, default=1000)
    downloader.add_argument("--controls-per-source", type=int, default=12)
    downloader.add_argument("--interval", type=float, default=0.35)
    for action in ("fetch-official", "publish", "verify"):
        commands.add_parser(action)
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--source", action="append")
    sheets_parser = commands.add_parser("sheets")
    sheets_parser.add_argument("--source", action="append")
    reviewer = commands.add_parser("review")
    reviewer.add_argument("sheet")
    for status in ("positive", "negative", "ambiguous", "burst", "excluded"):
        reviewer.add_argument(f"--{status}", default="")
    args = parser.parse_args()
    if args.action == "fetch":
        if args.interval < 0.25 or min(args.positives_per_source, args.controls_per_source) < 0:
            parser.error("Use at least 0.25 seconds between request starts and nonnegative limits")
        fetch(args.source or PROJECTS, args.positives_per_source, args.controls_per_source, args.interval)
    elif args.action == "review":
        review(args.sheet, args.positive, args.negative, args.ambiguous, args.burst, args.excluded)
    elif args.action == "sheets":
        sheets(args.source)
    elif args.action == "stage":
        stage(args.source)
    else:
        {"fetch-official": fetch_official, "publish": publish, "verify": verify}[args.action]()


if __name__ == "__main__":
    main()
