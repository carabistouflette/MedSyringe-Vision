"""Collect public syringe-photo candidates and audit their local provenance."""
import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from fcntl import LOCK_EX, LOCK_NB, flock
from hashlib import sha256
from html import unescape
from io import BytesIO
import json
from pathlib import Path
import re
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image, ImageDraw, ImageOps

from medsyringe.data import dhash, digest, read_image, read_json, write_json
from scripts.detection_public_benchmark import transformed_hashes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "public-web-collection"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
AGENT = "MedSyringeVisionCollectionBot/1.0 (https://github.com/carabistouflette/MedSyringe-Vision)"
FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
LIMIT = 5 * 1024 * 1024
THRESHOLD = 4
QUOTA = {"commons": 48, "openverse": 20}
# Lists are ordered for round-robin sampling. A query's stratum is a retrieval hint.
COMMONS = {
    "crowded": (
        "intitle:syringes", "intitle:seringues", "intitle:spritzen",
        "intitle:syringe intitle:set", "intitle:syringe intitle:disposal",
        "intitle:sharps intitle:container", "incategory:Disposable_syringes",
        "incategory:Syringes", "intitle:seringas", "intitle:jeringas",
    ),
    "transparent": (
        "intitle:plastic intitle:syringe", "intitle:glass intitle:syringe",
        "intitle:disposable intitle:syringe", "intitle:clear intitle:syringe",
        "intitle:syringe intitle:water", "incategory:Insulin_syringes",
        "intitle:seringue", "intitle:strzykawki", "intitle:glass intitle:syringes",
    ),
    "occluded": (
        "intitle:syringe intitle:hand", "intitle:syringe intitle:preparation",
        "intitle:prepares intitle:vaccine", "intitle:injection intitle:draw",
        "intitle:phlebotomy", "intitle:syringe intitle:vial",
        "intitle:administers intitle:vaccine",
    ),
    "lookalike": (
        "intitle:pipette", "intitle:autoinjector", "intitle:insulin intitle:pen",
        "intitle:cannula", "intitle:ampoule", "intitle:dropper",
        "intitle:epoxy intitle:applicator", "intitle:specimen intitle:tube",
        "intitle:mechanical intitle:pencil",
    ),
    "attributes": (
        "intitle:syringe intitle:plunger", "intitle:syringe intitle:graduated",
        "intitle:syringe intitle:5ml", "intitle:syringe intitle:10ml",
        "intitle:syringe intitle:filled", "intitle:syringe intitle:liquid",
        "intitle:syringe intitle:insulin", "intitle:syringe intitle:ml",
        "intitle:strzykawka", "intitle:seringue intitle:graduée",
        "intitle:spritze intitle:ml", "intitle:syringe intitle:closeup",
    ),
}
OPENVERSE = {
    "crowded": ("syringes", "discarded syringes", "syringes medical", "syringes tray", "needle exchange",
                "vaccination syringes", "syringe pack"),
    "transparent": ("clear syringe", "glass syringe", "plastic syringe", "disposable syringe", "syringe",
                    "vaccine syringe"),
    "occluded": ("syringe in hand", "filling syringe", "syringe injection", "preparing vaccine", "syringe vial"),
    "lookalike": ("pipette", "autoinjector", "insulin pen", "dropper", "cannula", "ampoule"),
    "attributes": ("syringe plunger", "syringe graduation", "filled syringe", "syringe closeup", "insulin syringe",
                   "syringe barrel", "syringe 10 ml", "syringe medical"),
}
URL_FIELDS = ("source_url", "original_url", "image_url", "thumbnail_url", "source_image_url",
              "foreign_landing_url", "url")
ATTRIBUTES = (
    "type_visible", "markings_visible", "marking_text", "graduations_visible",
    "plunger_visible", "barrel_transparency", "barrel_interior_visible", "liquid_visibility",
)
REVIEW = OUT / "visual_review.json"


class Throttled(Exception):
    pass


class SameEndpointRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        before, after = urlsplit(req.full_url), urlsplit(newurl)
        if (before.scheme, before.netloc) != (after.scheme, after.netloc):
            raise ValueError("Cross-origin download redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicAPI:
    def __init__(self, offline):
        self.offline = offline
        self.last = {}
        self.opener = build_opener(SameEndpointRedirect())

    def get(self, url, route):
        if self.offline:
            raise ValueError("This operation requires a network request.")
        parsed = urlsplit(url)
        allowed = {
            "commons_search": {"commons.wikimedia.org"},
            "commons_thumbnail": {"upload.wikimedia.org", "thumb.wikimedia.org"},
            "openverse_search": {"api.openverse.org"},
            "openverse_thumbnail": {"api.openverse.org"},
        }[route]
        if parsed.scheme != "https" or parsed.netloc not in allowed:
            raise ValueError("URL outside the configured public endpoints.")
        interval = {"openverse_search": 3.2, "commons_search": 0.8}.get(route, 0.35)
        for retry in range(2):
            sleep(max(0, interval - (monotonic() - self.last.get(route, 0))))
            self.last[route] = monotonic()
            try:
                with self.opener.open(Request(url, headers={"User-Agent": AGENT}), timeout=16) as response:
                    body = response.read(LIMIT + 1)
                if len(body) > LIMIT:
                    raise ValueError("Response exceeds the download limit.")
                return body
            except HTTPError as error:
                retry_after = error.headers.get("Retry-After")
                if not retry and (retry_after or error.code in {502, 503, 504}):
                    try:
                        delay = int(retry_after) if retry_after else 8
                    except ValueError:
                        delay = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
                    if delay <= 45:
                        sleep(max(delay, 1))
                        continue
                if retry_after or error.code == 429:
                    raise Throttled(f"HTTP {error.code}, Retry-After={retry_after}") from error
                raise


def normalized(value):
    return " ".join(unquote(str(value)).replace("_", " ").casefold().split())


def plain(value):
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value or "")).split()) or None


def url_key(value):
    if not isinstance(value, str) or not value.startswith(("http://", "https://", "//")):
        return None
    url = urlsplit(value)
    host = url.hostname
    if not host:
        return None
    path = unquote(url.path).rstrip("/")
    if host == "commons.wikimedia.org" and path.startswith("/wiki/File:"):
        return "commons:file:" + normalized(path.removeprefix("/wiki/File:"))
    if host in {"upload.wikimedia.org", "thumb.wikimedia.org"}:
        file = re.search(r"/wikipedia/commons/(?:thumb/)?[a-f0-9]/[a-f0-9]{2}/([^/]+)", path)
        if file:
            return "commons:file:" + normalized(file[1])
    if host in {"www.flickr.com", "flickr.com"}:
        photo = re.match(r"/photos/[^/]+/(\d+)", path)
        if photo:
            return "flickr:photo:" + photo[1]
    if host.endswith(".staticflickr.com"):
        photo = re.match(r"/\d+/(\d+)_", path)
        if photo:
            return "flickr:photo:" + photo[1]
    return host.removeprefix("www.") + path


def row_urls(row):
    return {key for field in URL_FIELDS if (key := url_key(row.get(field)))}


def license_ok(url):
    return bool(url and re.fullmatch(
        r"(?:https?:)?//creativecommons\.org/(?:licenses/by(?:-sa)?/[1-4]\.\d|"
        r"publicdomain/(?:zero|mark)/1\.0)(?:/[^?#]*)?(?:[?#].*)?", url, re.IGNORECASE
    ))


def creator_key(row):
    if row.get("author"):
        return "public-author:" + normalized(row["author"])
    return "unresolved:" + row["id"]


def commons_record(page):
    if not page.get("imageinfo"):
        return None, "missing_imageinfo"
    info = page["imageinfo"][0]
    if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
        return None, "non_raster"
    meta = {k: v["value"] for k, v in info.get("extmetadata", {}).items()}
    label = plain(meta.get("LicenseShortName"))
    license_url = plain(meta.get("LicenseUrl"))
    public_domain = (meta.get("License", "").casefold() in {"pd", "public-domain"}
                     and (label or "").casefold() == "public domain")
    if not (license_ok(license_url) or public_domain):
        return None, "license_filter"
    native_id = str(page["pageid"])
    artist = meta.get("Artist", "")
    return dict(
        id="commons-" + native_id, native_id=native_id, api="commons", provider="wikimedia",
        title=page["title"], source_url=info.get("descriptionurl"), original_url=info.get("url"),
        thumbnail_url=info.get("thumburl"), author=plain(artist), author_url=None,
        author_aliases=sorted({normalized(name) for name in
                               re.findall(r'(?:/wiki/|title=)User:([^"&?#/]+)', artist, re.I)}),
        credit=plain(meta.get("Credit")), attribution=plain(meta.get("Attribution")),
        license=label or license_url, license_url=license_url,
        source_timestamp=info.get("timestamp"), source_sha1=info.get("sha1"),
        source_dimensions=[info.get("width"), info.get("height")],
    ), None


def openverse_record(item):
    if item.get("source") == "wikimedia" or item.get("provider") == "wikimedia":
        return None, "wikimedia_cross_index"
    if not license_ok(item.get("license_url")):
        return None, "license_filter"
    native_id = item["id"]
    return dict(
        id="openverse-" + native_id, native_id=native_id, api="openverse",
        provider=item.get("provider"), catalog_source=item.get("source"), title=item.get("title"),
        source_url=item.get("foreign_landing_url"), original_url=item.get("url"),
        thumbnail_url=item.get("thumbnail"), author=item.get("creator"), author_url=item.get("creator_url"),
        author_aliases=[], credit=None, attribution=item.get("attribution"),
        license=item.get("license"), license_url=item.get("license_url"),
        license_version=item.get("license_version"), source_timestamp=None, source_sha1=None,
        source_dimensions=[item.get("width"), item.get("height")],
    ), None


def query_url(source, term):
    if source == "openverse":
        return OPENVERSE_API + "?" + urlencode(dict(
            q=term, page_size=16, category="photograph", license="by,by-sa,cc0,pdm",
            excluded_source="wikimedia",
        ))
    return COMMONS_API + "?" + urlencode(dict(
        action="query", format="json", formatversion=2, generator="search", gsrnamespace=6,
        gsrsearch=term + " filetype:bitmap", gsrlimit=36, prop="imageinfo",
        iiprop="url|size|mime|timestamp|sha1|extmetadata", iiurlwidth=800,
        iiextmetadatafilter="Artist|Credit|License|LicenseShortName|LicenseUrl|Attribution",
    ))


def clean_author(author):
    if not author or normalized(author) in {"unknown", "own work", "anonymous", "inconnu", "n/a"}:
        return None
    return author


def discover(client):
    rows, queries = {}, []
    queues = defaultdict(list)
    for source, plan in (("commons", COMMONS), ("openverse", OPENVERSE)):
        throttled = False
        for stratum, terms in plan.items():
            for term in terms:
                url = query_url(source, term)
                path = OUT / "api" / (sha256(url.encode()).hexdigest() + ".json")
                query = dict(api=source, stratum=stratum, term=term, url=url,
                             snapshot=path.relative_to(OUT).as_posix())
                queries.append(query)
                try:
                    if path.exists():
                        response = read_json(path)
                    elif client.offline or throttled:
                        query["error"] = "no cached response"
                        continue
                    else:
                        payload = json.loads(client.get(url, source + "_search"))
                        if "error" in payload:
                            raise ValueError(str(payload["error"]))
                        response = dict(fetched_at=datetime.now(UTC).isoformat(), payload=payload)
                        write_json(path, response)
                except (OSError, ValueError, Throttled) as error:
                    query["error"] = str(error)
                    throttled |= isinstance(error, Throttled) or (
                        isinstance(error, HTTPError) and error.code in {401, 403}
                    )
                    print(f"{source} | {term} | {error}", flush=True)
                    continue
                payload = response["payload"]
                if source == "commons":
                    items = sorted(payload.get("query", {}).get("pages", []),
                                   key=lambda page: page.get("index", 0))
                    extract = commons_record
                else:
                    items = payload["results"]
                    extract = openverse_record
                ids, excluded = [], Counter()
                for item in items:
                    row, reason = extract(item)
                    if reason:
                        excluded[reason] += 1
                        continue
                    if not row["thumbnail_url"]:
                        excluded["missing_thumbnail"] += 1
                        continue
                    row_id = row["id"]
                    if row_id not in rows:
                        row["author"] = clean_author(row["author"])
                        row["group"] = creator_key(row)
                        row["queries"] = []
                        rows[row_id] = row
                    rows[row_id]["queries"].append(dict(stratum=stratum, term=term))
                    ids.append(row_id)
                queries[-1].update(fetched_at=response["fetched_at"], sha256=digest(path),
                                   returned=len(items), eligible=len(ids), excluded=dict(excluded))
                queues[source, stratum].append(ids)
                print(f"{source} | {stratum} | {term} | {len(ids)}/{len(items)}", flush=True)
    return rows, queues, queries


def historical_records(obj):
    if isinstance(obj, dict):
        if ("id" in obj or "image" in obj or "source" in obj) and any(
            k in obj for k in (*URL_FIELDS[:-1], "sha256", "dhash", "derived_sha256", "author")
        ):
            yield obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from historical_records(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from historical_records(value)


def historical_index():
    paths = set((ROOT / "artifacts").rglob("*.json"))
    paths.update((DATA / "detection_hillclimb" / "public").rglob("*.json"))
    paths.difference_update({DATA / "detection_hillclimb" / "public" / "catalog.json"})
    paths.update((DATA / "rfdetr").glob("*-candidates.json"))
    paths.update((DATA / "rfdetr").glob("*-images.json"))
    hill = DATA / "detection_hillclimb"
    raster_dirs = [*DATA.glob("commons*"), DATA / "original", DATA / "original-color",
                   DATA / "expanded", DATA / "sam3" / "public", DATA / "rfdetr" / "public",
                   DATA / "openimages" / "test", hill / "public" / "images", hill / "device-native",
                   hill / "device-context", hill / "dental" / "images", hill / "openimages" / "images",
                   hill / "medbin-original"]
    private_dirs = {DATA / "original", DATA / "original-color", DATA / "expanded"}
    private_manifests = {"artifacts/source_inventory.json", "artifacts/expanded/inventory.json"}
    index = {field: defaultdict(set) for field in ("url", "id", "author", "sha256", "dhash", "fil_rouge_dhash")}
    summary = dict(metadata_files={}, raster_files={}, unreadable=[])
    for path in sorted(paths):
        location = path.relative_to(ROOT).as_posix()
        try:
            records = list(historical_records(read_json(path)))
        except (ValueError, OSError):
            summary["unreadable"].append(location)
            continue
        if not records:
            continue
        summary["metadata_files"][location] = len(records)
        for row in records:
            row_id = str(row.get("id", ""))
            urls = row_urls(row)
            group = str(row.get("group", ""))
            authors = {normalized(row["author"])} if row.get("author") else set()
            authors.update(normalized(alias) for alias in row.get("profile_keys", []))
            if group.startswith("commons:"):
                authors.add(normalized(group.removeprefix("commons:")))
            ref = location + ("#" + row_id if row_id else "")
            keys = {
                "url": urls,
                "id": [row_id] if row_id and (group.startswith("commons:") or
                        any(key.startswith("commons:file:") for key in urls)) else [],
                "author": authors,
                "sha256": [row.get("sha256"), row.get("derived_sha256")],
                "dhash": [row.get("dhash")],
            }
            for field, values in keys.items():
                for value in filter(None, values):
                    if field == "dhash":
                        if not re.fullmatch(r"[a-fA-F0-9]{16}", str(value)):
                            continue
                        value = int(value, 16)
                    index[field][value].add(ref)
                    if field == "dhash" and location in private_manifests:
                        index["fil_rouge_dhash"][value].add(ref)
    for folder in raster_dirs:
        scanned = 0
        for path in sorted(folder.glob("*")):
            if path.suffix.casefold() not in SUFFIXES or not path.is_file():
                continue
            location = path.relative_to(ROOT).as_posix()
            index["sha256"][digest(path)].add(location)
            try:
                fingerprint = int(dhash(read_image(path)), 16)
                index["dhash"][fingerprint].add(location)
                if folder in private_dirs:
                    index["fil_rouge_dhash"][fingerprint].add(location)
            except (OSError, ValueError):
                summary["unreadable"].append(location)
            scanned += 1
        if scanned:
            summary["raster_files"][folder.relative_to(ROOT).as_posix()] = scanned
    summary["index_sizes"] = {field: len(values) for field, values in index.items()}
    return index, summary


def references(index, field, keys):
    return sorted({ref for key in keys for ref in index[field].get(key, ())})


def near_matches(variants, values):
    return sorted(
        (distance, key) for key in values
        if (distance := min((variant ^ key).bit_count() for variant in variants)) <= THRESHOLD
    )


def read_thumbnail(payload):
    with Image.open(BytesIO(payload)) as image:
        if image.format not in FORMAT_EXT or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Unsupported raster format.")
        if min(image.size) < 120 or max(image.size) > 2000:
            raise ValueError("Thumbnail dimensions outside 120–2000 px.")
        return FORMAT_EXT[image.format], ImageOps.exif_transpose(image).convert("RGB")


def thumbnail(client, row):
    assets = sorted((OUT / "images").glob(row["id"] + ".*"))
    if assets:
        path = assets[0]
        payload = path.read_bytes()
    else:
        if client.offline:
            return None
        payload = client.get(row["thumbnail_url"], row["api"] + "_thumbnail")
        path = None
    ext, image = read_thumbnail(payload)
    return path or OUT / "images" / (row["id"] + ext), payload, image


def round_robin(queues):
    for rank in range(max(map(len, queues), default=0)):
        for queue in queues:
            if rank < len(queue):
                yield queue[rank]


def collect(client, catalog, queues, history):
    checked, urls, checksums = set(), set(), set()
    rows, rejected = [], []
    counts, group_counts = Counter(), Counter()
    paused = set()
    # Alternate intent buckets. A candidate reached through several terms retains all query hints.
    selectors = {key: iter(round_robin(queue)) for key, queue in queues.items()}
    while selectors:
        for key, choices in list(selectors.items()):
            source, stratum = key
            if counts[key] >= QUOTA[source] or source in paused:
                del selectors[key]
                continue
            for row_id in choices:
                row = catalog[row_id]
                if row_id in checked or group_counts[key, row["group"]] >= 5:
                    continue
                matched_urls = row_urls(row)
                known = references(history, "url", matched_urls)
                if source == "commons":
                    known = sorted(set(known) | set(references(history, "id", [row["native_id"]])))
                checked.add(row_id)
                notes = dict(id=row_id, api=source, stratum=stratum, source_url=row["source_url"])
                if known or matched_urls & urls:
                    rejected.append(dict(notes, reason="known_source" if known else "duplicate_source",
                                         historical_references=known[:12], historical_reference_count=len(known)))
                    continue
                try:
                    result = thumbnail(client, row)
                    if result is None:
                        rejected.append(dict(notes, reason="uncached_thumbnail"))
                        continue
                    path, payload, image = result
                except Throttled as error:
                    paused.add(source)
                    rejected.append(dict(notes, reason="throttled", error=str(error)))
                    break
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    rejected.append(dict(notes, reason="thumbnail_error", error=str(error)))
                    continue
                checksum = sha256(payload).hexdigest()
                known = references(history, "sha256", [checksum])
                if known or checksum in checksums:
                    rejected.append(dict(notes, reason="historical_sha256" if known else "duplicate_sha256",
                                         historical_references=known[:12], historical_reference_count=len(known)))
                    continue
                private_matches = near_matches(transformed_hashes(image), history["fil_rouge_dhash"])
                if private_matches:
                    known = references(history, "fil_rouge_dhash", [value for _, value in private_matches])
                    rejected.append(dict(notes, reason="fil_rouge_dhash", historical_references=known[:12],
                                         historical_reference_count=len(known)))
                    continue
                row = dict(row, image=path.relative_to(DATA).as_posix(), sha256=checksum, dhash=dhash(image),
                           width=image.width, height=image.height, stratum=stratum)
                if not path.exists():
                    temporary = path.with_suffix(".part")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                rows.append(row)
                urls.update(matched_urls)
                checksums.add(checksum)
                counts[key] += 1
                group_counts[key, row["group"]] += 1
                if len(rows) % 20 == 0:
                    print(f"{len(rows)} thumbnails retained", flush=True)
                break
            else:
                del selectors[key]
    return rows, rejected


def audit(rows, history):
    by_hash = defaultdict(set)
    for row in rows:
        by_hash[int(row["dhash"], 16)].add(row["id"])
    for row in rows:
        variants = transformed_hashes(read_image(DATA / row["image"]))
        historical_near = near_matches(variants, history["dhash"])
        recent_near = near_matches(variants, by_hash)
        aliases = row["author_aliases"][:]
        if row["author"]:
            aliases.append(normalized(row["author"]))
        historical_author = references(history, "author", aliases)
        row["historical"] = dict(
            author_reference_count=len(historical_author), author_references=historical_author[:12],
            dhash_matches=[dict(dhash=f"{key:016x}", distance=distance,
                                reference_count=len(history["dhash"][key]),
                                references=sorted(history["dhash"][key])[:5])
                           for distance, key in historical_near],
        )
        row["candidate_dhash_matches"] = sorted(
            (dict(id=other_id, distance=distance) for distance, key in recent_near
             for other_id in by_hash[key] if other_id != row["id"]),
            key=lambda match: (match["distance"], match["id"]),
        )


def sheets(rows):
    folder = OUT / "sheets"
    folder.mkdir(exist_ok=True)
    for stratum in COMMONS:
        selection = [row for row in rows if row["stratum"] == stratum]
        for start in range(0, len(selection), 36):
            page = selection[start:start + 36]
            sheet = Image.new("RGB", (6 * 204, ((len(page) + 5) // 6) * 156), "#e9e9e9")
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(page):
                x, y = index % 6 * 204, index // 6 * 156
                thumb = ImageOps.contain(read_image(DATA / row["image"]), (198, 128))
                sheet.paste(thumb, (x + (204 - thumb.width) // 2, y))
                draw.text((x + 5, y + 130), f"{row['review_index']:03d}  {row['api']}", fill="black")
            sheet.save(folder / f"{stratum}-{start // 36 + 1}.jpg", quality=88)


def apply_review(manifest):
    review = read_json(REVIEW)["rows"] if REVIEW.exists() else []
    by_id = {item["id"]: item for item in review}
    rows = manifest["rows"]
    if len(by_id) != len(review) or by_id.keys() - {row["id"] for row in rows}:
        raise ValueError("Unknown or duplicate image in visual review.")
    for row in rows:
        item = by_id.get(row["id"])
        values = item.get("attributes", {}) if item else {}
        if item and row["sha256"] != item["sha256"]:
            raise ValueError(f"Visual review targets different raster bytes: {row['id']}")
        if values.keys() - set(ATTRIBUTES):
            raise ValueError(f"Unknown attribute for {row['id']}")
        row["attributes"] = {field: values.get(field) for field in ATTRIBUTES}
        row["visual_review"] = item.get("observations") if item else None
    manifest["visual_review"] = {
        "source_sha256": digest(REVIEW) if REVIEW.exists() else None,
        "reviewed_rows": len(by_id),
        "known_by_attribute": {
            field: sum(row["attributes"][field] is not None for row in rows) for field in ATTRIBUTES
        },
    }


def local_image(row):
    path = DATA / row["image"]
    if not path.resolve().is_relative_to(OUT / "images"):
        raise ValueError(f"Image outside the collection: {path}")
    return path


def restore(client):
    restored = 0
    manifest = read_json(OUT / "manifest.json")
    for row in manifest["rows"]:
        path = local_image(row)
        if path.exists():
            continue
        payload = client.get(row["thumbnail_url"], row["api"] + "_thumbnail")
        if sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"The provider no longer serves the pinned bytes for {row['id']}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
        restored += 1
    print(f"Restored {restored} pinned thumbnails.")


def verify():
    manifest = read_json(OUT / "manifest.json")
    rows = manifest["rows"]
    if len(rows) != len({row["id"] for row in rows}) or len(rows) != len({row["sha256"] for row in rows}):
        raise ValueError("Duplicate ID or exact checksum in candidate manifest.")
    for row in rows:
        path = local_image(row)
        if digest(path) != row["sha256"]:
            raise ValueError(f"SHA-256 differs for {path}.")
        image = read_image(path)
        if dhash(image) != row["dhash"] or image.size != (row["width"], row["height"]):
            raise ValueError(f"Appearance fingerprint or dimensions differ for {path}.")
        if row.keys() & {"split", "objects", "parents"} or set(row["attributes"]) != set(ATTRIBUTES):
            raise ValueError(f"Unexpected review schema for {row['id']}.")
    for query in read_json(OUT / "queries.json")["queries"]:
        if "sha256" in query and digest(OUT / query["snapshot"]) != query["sha256"]:
            raise ValueError(f"API snapshot checksum differs for {query['snapshot']}.")
    print(f"Verified {len(rows)} images (SHA-256, dHash and dimensions).")


def write_collection(offline):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    client = PublicAPI(offline)
    catalog, queues, queries = discover(client)
    history, historical_summary = historical_index()
    rows, rejected = collect(client, catalog, queues, history)
    audit(rows, history)
    # Stable numbers for the local sheets, with one block of numbers per retrieval intent.
    rows.sort(key=lambda row: list(COMMONS).index(row["stratum"]))
    for number, row in enumerate(rows, start=1):
        row["review_index"] = number
    counts = {source: {stratum: sum(row["api"] == source and row["stratum"] == stratum for row in rows)
                       for stratum in COMMONS} for source in QUOTA}
    report = dict(
        schema_version=1, image_base="data", status="unannotated_candidate_pool",
        code_sha256=digest(__file__), quota_per_stratum=QUOTA, counts=counts,
        query_count=len(queries), catalog_unique=len(catalog), historical=historical_summary,
        rejected_counts=dict(Counter(row["reason"] for row in rejected)), rows=rows,
    )
    apply_review(report)
    write_json(OUT / "queries.json", dict(queries=queries))
    write_json(OUT / "rejected.json", dict(rows=rejected))
    write_json(OUT / "manifest.json", report)
    sheets(rows)
    print(json.dumps(dict(counts=counts, rejected=report["rejected_counts"]), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "review", "restore", "verify"))
    parser.add_argument("--offline", action="store_true", help="Replay cached API responses and thumbnails.")
    args = parser.parse_args()
    if args.command != "verify":
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / ".lock").open("a") as lock:
            try:
                flock(lock, LOCK_EX | LOCK_NB)
            except BlockingIOError:
                parser.exit(1, "Another collection process is writing to this directory.\n")
            if args.command == "collect":
                write_collection(args.offline)
            elif args.command == "review":
                manifest = read_json(OUT / "manifest.json")
                apply_review(manifest)
                write_json(OUT / "manifest.json", manifest)
            else:
                restore(PublicAPI(args.offline))
    verify()


if __name__ == "__main__":
    main()
