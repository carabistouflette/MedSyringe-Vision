"""Build and verify a photo-only, manually reviewed public syringe benchmark."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from html import unescape
from io import BytesIO
import json
from mimetypes import guess_type
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageOps

from medsyringe.data import author_key, dhash, digest, leakage, read_image, read_json, valid_box, write_json

ROOT = Path("data")
WORK = ROOT / "detection_hillclimb" / "public"
ART = Path("artifacts/detection_hillclimb")
CATALOG = WORK / "catalog.json"
POOL = WORK / "pool.json"
REVIEWS = ART / "public_reviews.json"
MANIFEST = ART / "public_manifest.json"
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MedSyringeVision/1.0 (local academic image benchmark; single-user research)"
SEED = "independent-2026-11-21-v1"
CATEGORIES = {
    "syringe": ("Disposable syringes", "Syringes", "Insulin syringes", "Hypodermic needles",
                "Injection (medicine)", "Phlebotomy", "COVID-19 vaccination", "COVID-19 vaccination in the United States",
                "Vaccination in the United States", "Vaccination", "Syringe drivers"),
    "lookalike": ("Pipettes", "Pasteur pipettes", "Air displacement pipettes", "Insulin pens",
                  "Mechanical pencils", "Ballpoint pens", "Fountain pens"),
}
LIMIT = 9 * 1024 * 1024


def lexical_key(value):
    return " ".join(unquote(value).replace("_", " ").casefold().split())


def url_key(value):
    parsed = urlsplit(value)
    return lexical_key(parsed.netloc + parsed.path)


def order(value):
    return sha256(f"{SEED}:{value}".encode()).digest()


def records(value):
    if isinstance(value, dict):
        if "id" in value and any(k in value for k in ("author", "source_url", "sha256")):
            yield value
        else:
            for child in value.values():
                yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def prior_exposure():
    manifests = [*Path("artifacts").glob("*.json"), *Path("artifacts/expanded").glob("*.json"),
                 *Path("data/rfdetr").glob("*-candidates.json")]
    rows = []
    for p in sorted(set(manifests)):
        rows.extend(records(read_json(p)))
    paths = {p for folder in [*ROOT.glob("commons*"), ROOT / "sam3" / "public"] if folder.is_dir()
             for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    ids = {str(r["id"]) for r in rows} | {p.stem for p in paths}
    urls = {url_key(r[k]) for r in rows for k in ("source_url", "image_url", "original_url") if r.get(k)}
    titles = {lexical_key(r["title"]) for r in rows if r.get("title")}
    authors = {lexical_key(r["author"]) for r in rows if r.get("author")}
    authors |= {lexical_key(r["group"].split(":", 1)[1]) for r in rows
                if str(r.get("group", "")).startswith("commons:")}
    hashes = {r[k] for r in rows for k in ("sha256", "derived_sha256") if r.get(k)}
    difference = {int(r["dhash"], 16) for r in rows if re.fullmatch(r"[0-9a-fA-F]{16}", str(r.get("dhash", "")))}
    for p in sorted(paths):
        hashes.add(digest(p))
        try:
            difference.add(int(dhash(read_image(p)), 16))
        except OSError:
            continue
    return dict(ids=ids, urls=urls, titles=titles, authors=authors, hashes=hashes, difference=difference)


def request(url, size_limit):
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=20) as response:
        if int(response.headers.get("Content-Length", 0)) > size_limit:
            raise ValueError("Response exceeds the benchmark download bound.")
        payload = response.read(size_limit + 1)
    if len(payload) > size_limit:
        raise ValueError("Response exceeds the benchmark download bound.")
    return payload


def plain(value):
    return " ".join(unescape(re.sub("<[^>]+>", " ", value)).split())


def metadata(page, category, stratum):
    if not page.get("imageinfo") or page.get("ns") != 6:
        return None
    info = page["imageinfo"][0]
    mime = info.get("mime") or guess_type(page["title"])[0]
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        return None
    if min(info.get("width", 0), info.get("height", 0)) < 220:
        return None
    ext = info.get("extmetadata", {})
    artist = ext.get("Artist", {}).get("value", "")
    author = plain(artist)
    tag = plain(ext.get("LicenseShortName", {}).get("value", ""))
    license_url = plain(ext.get("LicenseUrl", {}).get("value", ""))
    token = ext.get("License", {}).get("value", "").casefold()
    license_key = url_key(license_url)
    qualified = re.fullmatch(r"creativecommons\.org/licenses/(by|by-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0)(?:/.*)?",
                             license_key)
    cc_zero = license_key.startswith("creativecommons.org/publicdomain/zero/1.0")
    public_domain = token in {"pd", "public-domain"} and tag.casefold() == "public domain"
    if not (qualified or cc_zero or public_domain):
        return None
    if not author or len(author) > 145 or author.casefold() in {"unknown", "own work", "anonymous"}:
        return None
    profile_keys = set()
    for name in re.findall(r'(?:/wiki/|title=)User:([^"&?#/]+)', artist, re.IGNORECASE):
        profile_keys.add(lexical_key(name))
    return dict(id=str(page["pageid"]), title=page["title"], category=category, intent=stratum,
                author=author, profile_keys=sorted(profile_keys), group="commons:" + author_key(author),
                source_url=info["descriptionurl"], original_url=info["url"], thumbnail_url=info.get("thumburl"),
                license=license_url or tag, license_label=tag, file_timestamp=info.get("timestamp"),
                source_dimensions=[info["width"], info["height"]], mime=mime, size=info.get("size"))


def withheld(row, exposure):
    keys = {lexical_key(row["author"]), *row.get("profile_keys", [])}
    return (row["id"] in exposure["ids"] or lexical_key(row["title"]) in exposure["titles"]
            or bool(keys & exposure["authors"]) or
            any(url_key(row[k]) in exposure["urls"] for k in ("source_url", "original_url")))


def discover():
    found = read_json(CATALOG) if CATALOG.exists() else {}
    inherited = read_json("data/rfdetr/lookalike-pages.json")
    for stratum, categories in CATEGORIES.items():
        for category in categories:
            if found.get(category):
                continue
            params = dict(action="query", generator="categorymembers", gcmtitle="Category:"+category,
                          gcmtype="file", gcmlimit="50", prop="imageinfo",
                          iiprop="url|size|mime|extmetadata|timestamp", iiurlwidth="960", format="json")
            try:
                result = json.loads(request(API+"?"+urlencode(params), 12 * 1024 * 1024))
                pages = list(result.get("query", {}).get("pages", {}).values())
                pages = [p for p in pages if "imageinfo" in p]
                if not pages and category in inherited:
                    pages = inherited[category]
            except (ValueError, HTTPError, URLError, TimeoutError) as error:
                print(f"Catalogue {category}: {error}", flush=True)
                if category not in inherited:
                    continue
                pages = inherited[category]
            found[category] = [row for p in pages if (row := metadata(p, category, stratum))]
            write_json(CATALOG, found)
            print(f"Catalogue {category}: {len(found[category])} eligible metadata records.", flush=True)
    return found


def transformed_hashes(image):
    variants = [image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
    variants.extend(src.transpose(turn) for src in tuple(variants)
                    for turn in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270))
    return {int(dhash(src), 16) for src in variants}


def near_any(image, known):
    return any((variant ^ other).bit_count() <= 4 for variant in transformed_hashes(image) for other in known)


def get_asset(row):
    original = row["original_url"]
    if row["size"] and row["size"] < LIMIT:
        urls = (original, row["thumbnail_url"])
    elif row["size"]:
        urls = (row["thumbnail_url"],)
    else:
        urls = (row["thumbnail_url"], original)
    extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[row["mime"]]
    path = WORK / "images" / (row["id"] + extension)
    receipt = WORK / "images" / (row["id"] + ".json")
    if path.is_file() and receipt.is_file():
        acquired = read_json(receipt)
        if acquired["sha256"] != digest(path):
            raise ValueError(f"Cached downloaded bytes changed for {row['id']}.")
        return acquired, read_image(path)
    for url in dict.fromkeys(u for u in urls if u):
        try:
            payload = request(url, LIMIT)
            with Image.open(BytesIO(payload)) as im:
                pixels = ImageOps.exif_transpose(im).convert("RGB")
        except (OSError, HTTPError, URLError, TimeoutError, ValueError):
            continue
        if min(pixels.size) < 220:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        acquired = dict(**row, image=path.relative_to(ROOT).as_posix(), image_url=url, original_bytes=url == original,
                        width=pixels.width, height=pixels.height, sha256=digest(path), dhash=dhash(pixels))
        write_json(receipt, acquired)
        return acquired, pixels
    return None


def pool(positives, negatives):
    if MANIFEST.exists():
        raise ValueError("A frozen benchmark already exists.")
    found = discover()
    old = prior_exposure()
    picked = {r["id"]: r for r in (read_json(POOL)["rows"] if POOL.exists() else [])}
    used_groups = {r["group"] for r in picked.values()}
    used_profiles = {p for r in picked.values() for p in r.get("profile_keys", [])}
    known = {*old["difference"], *[int(r["dhash"], 16) for r in picked.values()]}
    counters = Counter(r["intent"] for r in picked.values())
    discarded = Counter()
    def rank(row):
        title = row["title"].casefold()
        terms = r"syring|spritze|injection|vaccin|injec|pipette|pen"
        illustration = r"poster|diagram|drawing|icon|svg|illustration|machine|high pressure|logo"
        return (bool(re.search(illustration, title)), not bool(re.search(terms, title)), order(row["id"]))

    offers = {category: iter(sorted(found.get(category, ()), key=rank))
              for categories in CATEGORIES.values() for category in categories}
    for stratum, target in (("syringe", positives), ("lookalike", negatives)):
        categories = CATEGORIES[stratum]
        while counters[stratum] < target:
            progressed = False
            for category in categories:
                candidate = next(offers[category], None)
                if candidate is None:
                    continue
                progressed = True
                if withheld(candidate, old) or candidate["id"] in picked or candidate["group"] in used_groups:
                    discarded["previous or same group"] += 1
                    continue
                if used_profiles.intersection(candidate["profile_keys"]):
                    discarded["same profile"] += 1
                    continue
                asset = get_asset(candidate)
                if not asset:
                    discarded["download or raster invalid"] += 1
                    continue
                row, raster = asset
                if row["sha256"] in old["hashes"] or near_any(raster, known):
                    discarded["exact or transformed near duplicate"] += 1
                    continue
                picked[row["id"]] = row
                used_groups.add(row["group"])
                used_profiles.update(row["profile_keys"])
                counters[stratum] += 1
                known.add(int(row["dhash"], 16))
                write_json(POOL, dict(rows=list(picked.values()), discarded=dict(discarded)))
                print(f"{stratum} {counters[stratum]:2}/{target}: {row['id']} {row['title']}", flush=True)
                if counters[stratum] >= target:
                    break
            if not progressed:
                break
    print(f"Pool: {dict(counters)}; filtered: {dict(discarded)}", flush=True)


def sheet(selected, tag, columns, rows_per_sheet):
    folder = WORK / "sheets"
    folder.mkdir(parents=True, exist_ok=True)
    geometry = {}
    count = columns * rows_per_sheet
    for start in range(0, len(selected), count):
        page = selected[start:start+count]
        width, height = 288, 253
        surface = Image.new("RGB", (columns*width, ((len(page)+columns-1)//columns)*height), "#efefef")
        draw = ImageDraw.Draw(surface)
        for j, row in enumerate(page):
            x0, y0 = j % columns * width + 8, j // columns * height + 5
            im = ImageOps.contain(read_image(ROOT / row["image"]), (width-16, height-36))
            x = x0 + (width-16-im.width)//2
            y = y0 + (height-36-im.height)//2
            surface.paste(im, (x,y))
            draw.text((x0, y0+height-30), row["id"]+" | "+row["title"][5:45], fill="black")
            draw.rectangle((x-1,y-1,x+im.width,y+im.height), outline="#555555")
            geometry[row["id"]] = [start//count, x, y, im.width, im.height]
        surface.save(folder / f"{tag}-{start//count:02}.jpg", quality=92)
    write_json(folder / f"{tag}-coordinates.json", geometry)


def review_sheets():
    selected = read_json(POOL)["rows"]
    for label in CATEGORIES:
        sheet([r for r in selected if r["intent"] == label], label, 4, 3)


def split_for(rows):
    """Balance support strata by whole photographer group before reading model output."""
    buckets = defaultdict(list)
    for r in rows:
        buckets["positive" if r["objects"] else "lookalike"].append(r)
    result = {}
    for bucket in sorted(buckets):
        groups = defaultdict(list)
        for row in buckets[bucket]:
            groups[row["group"]].append(row)
        for index, group in enumerate(sorted(groups, key=order)):
            for row in groups[group]:
                result[row["id"]] = "public-development" if index % 3 == 2 else "public-audit"
    return result


def finalize():
    if MANIFEST.exists():
        raise ValueError("Benchmark is frozen. Verify it without changing the references.")
    old = prior_exposure()
    seen = set(old["difference"])
    selected = {r["id"]: r for r in read_json(POOL)["rows"]}
    reviews = read_json(REVIEWS)
    rows = []
    for identifier, reviewed in reviews["photos"].items():
        row = selected[identifier]
        image = ROOT / row["image"]
        pixels = read_image(image)
        if withheld(row, old) or digest(image) != row["sha256"] or near_any(pixels, seen):
            raise ValueError(f"Previously exposed or changed image {identifier}.")
        seen.add(int(row["dhash"],16))
        values = reviewed["objects"]
        others = reviewed["lookalikes"]
        if not (values or others):
            raise ValueError(f"No visually verified benchmark subject on {identifier}.")
        if any(not valid_box(record["bbox"]) for record in [*values,*others]):
            raise ValueError(f"Invalid normalized box on {identifier}.")
        rows.append(dict(**row, **reviewed, reference_review="manual single-pass full-image review without predictions"))
    assignments = split_for(rows)
    for row in rows:
        row["split"] = assignments[row["id"]]
    if leakage(rows) or len({r["group"] for r in rows}) != len(rows):
        raise ValueError("Acquisition group or image leakage.")
    report = {}
    for split in sorted(set(assignments.values())):
        partition = [r for r in rows if r["split"] == split]
        report[split] = dict(images=len(partition), groups=len({r["group"] for r in partition}),
                             positives=sum(bool(r["objects"]) for r in partition),
                             negatives=sum(not r["objects"] for r in partition),
                             boxes=sum(len(r["objects"]) for r in partition),
                             multi=sum(len(r["objects"])>1 for r in partition),
                             partial=sum(any(o.get("occluded") or o.get("truncated")
                                             for o in r["objects"]) for r in partition))
    write_json(MANIFEST, dict(rows=sorted(rows,key=lambda r:(r["split"],r["id"])), partitions=report,
                              prior_exposure_counts={k:len(v) for k,v in old.items()},
                              review_sha256=digest(REVIEWS), assignment_seed=SEED,
                              assignment="Sort photographer groups by a seeded SHA-256 in each phenotype stratum. "
                                         "Every third group enters development; all others enter audit.",
                              domain="Independent Wikimedia Commons photographs with author and file-level attribution.",
                              exclusion_threshold="Exact hashes and dHash at Hamming distance <=4 over all reflected "
                                                  "and rotated orientations.",
                              benchmark_scope="Curated stress set. Operating precision depends on this class prevalence."))
    print(report, flush=True)


def verify():
    m = read_json(MANIFEST)
    if digest(REVIEWS) != m["review_sha256"]:
        raise ValueError("The independent visual references changed.")
    old = prior_exposure()
    seen = set(old["difference"])
    if leakage(m["rows"]):
        raise ValueError("A cross-split leakage pair was found.")
    for row in m["rows"]:
        path = ROOT / row["image"]
        pixels = read_image(path)
        if (withheld(row, old) or digest(path) != row["sha256"] or near_any(pixels, seen) or
                list(pixels.size) != [row["width"],row["height"]]):
            raise ValueError(f"Independent image integrity failure for {row['id']}.")
        seen.add(int(row["dhash"],16))
        if any(not valid_box(v["bbox"]) for v in row["objects"] + row["lookalikes"]):
            raise ValueError(f"Invalid persisted box for {row['id']}.")
    print(m["partitions"], flush=True)


def restore():
    """Restore only the exact renditions already bound to the review."""
    for row in read_json(MANIFEST)["rows"]:
        path = ROOT / row["image"]
        if path.exists():
            if digest(path) != row["sha256"]:
                raise ValueError(f"Refusing to overwrite different bytes for {row['id']}.")
            continue
        payload = request(row["image_url"], LIMIT)
        if sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"The publisher now serves different bytes for {row['id']}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "sheets", "freeze", "restore", "verify"))
    parser.add_argument("--positive-candidates", type=int, default=24)
    parser.add_argument("--lookalike-candidates", type=int, default=18)
    args = parser.parse_args()
    if args.stage == "fetch":
        pool(args.positive_candidates, args.lookalike_candidates)
    else:
        {"sheets": review_sheets, "freeze": finalize, "restore": restore, "verify": verify}[args.stage]()
