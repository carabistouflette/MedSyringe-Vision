"""Materialize visually confirmed public syringe photos at provider resolution."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from medsyringe.data import dhash, digest, read_image, read_json, write_json
from scripts.public_training_intake import close, raster_hashes, references

ROOT = Path("data/public-syringe-dataset")
LIMIT = 24 * 1024 * 1024
FORMATS = {"JPEG": "jpg", "MPO": "mpo", "PNG": "png", "WEBP": "webp"}


def checked_reviews(paths):
    accepted, excluded = {}, {}
    for path in paths:
        report = read_json(path)
        key = digest(path)
        for identifier in report["confirmed"]:
            if identifier in accepted:
                raise ValueError(f"Two review batches cover photo {identifier}.")
            accepted[identifier] = key
        excluded.update(report["excluded"])
    if set(accepted) & set(excluded):
        raise ValueError("Review batches disagree about a photo.")
    return accepted, excluded


def download(row, folder):
    address = row["original_url"]
    host = urlsplit(address)
    if host.scheme != "https" or host.hostname != "source.roboflow.com" or not row["id"].isalnum():
        return None, "Invalid provider asset locator"
    thumbnail = Path("data") / row["image"]
    try:
        if digest(thumbnail) != row["sha256"]:
            return None, "Reviewed preview checksum differs"
        cached = [folder / f"{row['id']}.{extension}" for extension in FORMATS.values()]
        matches = [path for path in cached if path.is_file()]
        if len(matches) > 1:
            return None, "Ambiguous cached provider raster"
        if matches:
            image = matches[0]
        else:
            with urlopen(Request(address, headers={"User-Agent": "MedSyringe-Vision/1.0 research dataset"}),
                         timeout=25) as response:
                payload = response.read(LIMIT + 1)
            if len(payload) > LIMIT:
                return None, "Provider image exceeds the local file-size limit"
            with Image.open(BytesIO(payload)) as picture:
                extension = FORMATS.get(picture.format)
                if extension is None or picture.width * picture.height > 42_000_000:
                    return None, "Unsupported provider raster"
                picture.load()
            image = folder / f"{row['id']}.{extension}"
            pending = image.with_suffix(".part")
            pending.write_bytes(payload)
            pending.replace(image)
        picture = read_image(image)
        width, height = picture.size
        visual_hash = dhash(picture)
        return {**row, "image": str(image.relative_to("data")), "sha256": digest(image),
                "width": width, "height": height, "dhash": visual_hash, "preview_sha256": row["sha256"],
                "preview_image": row["image"]}, None
    except HTTPError as problem:
        return None, f"HTTP {problem.code}"
    except (OSError, URLError, ValueError, Image.DecompressionBombError) as problem:
        return None, type(problem).__name__


def stable(row):
    # Interleave a reproducible mixture across projects instead of taking the first burst.
    return sha256(("medsyringe-positive-v1/" + row["id"]).encode()).hexdigest()


def build(sources, reviews, minimum):
    output = ROOT / "manifest.json"
    if output.exists():
        raise FileExistsError("The final photo manifest already exists.")
    approved, rejected = checked_reviews(reviews)
    candidates = [row for source in sources for row in read_json(source)["rows"]]
    all_rows = {row["id"]: row for row in candidates}
    if len(all_rows) != len(candidates):
        raise ValueError("Multiple catalogs contain the same provider ID.")
    if not (set(approved) | set(rejected)).issubset(all_rows):
        raise ValueError("A review refers to a photo outside the catalog.")
    groups = {}
    for r in sorted((all_rows[i] for i in approved), key=stable):
        groups.setdefault(r["group"], []).append(r)
    photos = ROOT / "images"
    photos.mkdir(parents=True, exist_ok=True)
    forbidden, user_hashes, user_dhashes = references()
    active, removed = [], []
    order = [row for group in zip_longest(*(groups[k] for k in sorted(groups))) for row in group if row]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for step in range(0, len(order), 24):
            batch = order[step:step + 24]
            for record, (row, failure) in zip(batch, pool.map(lambda r: download(r, photos), batch), strict=True):
                if row is None:
                    removed.append(dict(id=record["id"], reason=failure))
                    continue
                visual = raster_hashes(Path("data") / row["image"])
                protected = row["sha256"] in user_hashes or close(visual, user_dhashes)
                duplicate = row["sha256"] in forbidden["hashes"] or close(visual, forbidden["difference"])
                if protected or duplicate:
                    removed.append(dict(id=row["id"], reason="Reference or duplicate overlap at provider resolution"))
                    continue
                forbidden["hashes"].add(row["sha256"])
                forbidden["difference"].update(visual)
                active.append({**row, "positive_review_sha256": approved[row["id"]],
                               "label_status": "Syringe presence reviewed. Instance and attribute annotations pending."})
            print("Provider-resolution positive photos", len(active), "of", len(order), "reviewed", flush=True)
    result = dict(rows=active, provider_photos=len(active), target=minimum, target_reached=len(active) >= minimum,
                  sources={str(p): digest(p) for p in sources}, reviews={str(p): digest(p) for p in reviews},
                  photos_by_source=dict(Counter(row["group"] for row in active)),
                  rejected_by_review=rejected, rejected_at_original_resolution=removed,
                  acquisition_status="Keep each named source together pending acquisition and specimen lineage review.",
                  scope="Public real positive scenes. Duplicate and Fil Rouge screening rerun at provider resolution. "
                        "No synthetic cases, reviewed negatives or unlabeled image variants included.")
    write_json(output if result["target_reached"] else ROOT / "partial.json", result)
    print("Photo target met" if result["target_reached"] else "More visually confirmed photos are required",
          len(active), "/", minimum, flush=True)


def verify():
    report = read_json(ROOT / "manifest.json")
    rows = report["rows"]
    paths = [Path("data") / row["image"] for row in rows]
    if len(rows) != report["provider_photos"] or len(rows) < report["target"] or not report["target_reached"]:
        raise ValueError("The final photo manifest falls below its declared target.")
    if len({row["sha256"] for row in rows}) != len(rows) or len(set(paths)) != len(paths):
        raise ValueError("The final photo manifest contains duplicate assets.")
    if any(digest(path) != row["sha256"] for path, row in zip(paths, rows, strict=True)):
        raise ValueError("A retained photo is missing or differs from the final manifest.")
    for field in ("sources", "reviews"):
        if any(digest(Path(path)) != checksum for path, checksum in report[field].items()):
            raise ValueError(f"A bound {field} snapshot changed.")
    print("Verified", len(rows), "provider-resolution positive photos", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="*", type=Path)
    parser.add_argument("--review", action="append", type=Path, default=[])
    parser.add_argument("--minimum", type=int, default=500)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.source and args.review and args.minimum > 0:
        build(args.source, args.review, args.minimum)
    else:
        parser.error("Supply catalog paths, at least one visual review, and a positive target.")
