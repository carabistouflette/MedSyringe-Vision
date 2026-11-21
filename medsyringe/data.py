"""Read-only source audit, deterministic dataset assembly and COCO export."""
import argparse
from collections import Counter, defaultdict
import csv
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import io
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
CLASS = "/m/012ysf"
OPEN = "https://storage.googleapis.com/openimages"
EXIF_FIELDS = (
    "FileName", "ImageWidth", "ImageHeight", "Orientation", "DateTimeOriginal",
    "Make", "Model", "LensModel", "ExposureTime", "ISO", "FNumber", "ColorSpace",
)


def read_image(source):
    with Image.open(source) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def digest(path):
    with Path(path).open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def dhash(image):
    thumb = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = thumb.get_flattened_data()
    bits = (values[y * 9 + x] > values[y * 9 + x + 1] for y in range(8) for x in range(8))
    return f"{int(''.join(str(int(bit)) for bit in bits), 2):016x}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def audit(source, output, inventory):
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Source directory is empty.")
    exif = subprocess.run(
        ["exiftool", "-r", "-csv", *["-" + field for field in EXIF_FIELDS], str(source)],
        check=True, capture_output=True, text=True,
    )
    meta = {Path(r["SourceFile"]).resolve(): r for r in csv.DictReader(io.StringIO(exif.stdout))}
    seen = {}
    rows = []
    for p in sorted(files, key=lambda p: (p.name.count("("), str(p))):
        relative = str(p.relative_to(source))
        checksum = digest(p)
        duplicate = seen.setdefault(checksum, relative)
        info = {k: meta.get(p.resolve(), {}).get(k) for k in EXIF_FIELDS}
        row = dict(source=relative, sha256=checksum, bytes=p.stat().st_size, metadata=info)
        if p.suffix.lower() in (".heic", ".heif", ".jpg", ".jpeg", ".png"):
            image = read_image(p)
            row.update(width=image.width, height=image.height, dhash=dhash(image))
            if relative != duplicate:
                row.update(duplicate_of=duplicate, split="excluded")
            else:
                # One group for these known source acquisitions, regardless of EXIF gaps.
                transformed = Path("original") / Path(relative).with_suffix(".png")
                target = output / transformed
                target.parent.mkdir(parents=True, exist_ok=True)
                image.info.clear()
                image.save(target, compress_level=3)
                row.update(id=p.stem, group="local-2026-11-07-session", split="train",
                           image=transformed.as_posix(), derived_sha256=digest(target))
        else:
            row.update(split="excluded", reason="unsupported format")
        rows.append(row)
    write_json(inventory, sorted(rows, key=lambda r: r["source"]))


def download(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--retry", "2", "--max-time", "120", "-o", str(temporary), url],
                       check=True, capture_output=True)
        temporary.replace(path)


def author_key(name):
    return " ".join(name.casefold().split())


def openimages_test(data, inventory, blocked):
    split = "test"
    directory = data / "openimages" / split
    sources = {
        "boxes": f"{OPEN}/v5/{split}-annotations-bbox.csv",
        "labels": f"{OPEN}/v5/{split}-annotations-human-imagelabels-boxable.csv",
        "metadata": f"{OPEN}/2018_04/{split}/{split}-images-with-rotation.csv",
    }
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda item: download(item[1], directory / f"{item[0]}.csv"), sources.items()))
    rejected = {}
    boxed = defaultdict(list)
    with (directory / "boxes.csv").open() as file:
        for row in csv.DictReader(file):
            if row["LabelName"] == CLASS:
                boxed[row["ImageID"]].append(row)
    verified_negative = set()
    with (directory / "labels.csv").open() as file:
        for row in csv.DictReader(file):
            if row["LabelName"] == CLASS and row["Confidence"] == "0" and row["ImageID"] not in boxed:
                verified_negative.add(row["ImageID"])
    candidates = set(boxed) | verified_negative
    info = {}
    with (directory / "metadata.csv").open() as file:
        for row in csv.DictReader(file):
            if row["ImageID"] in candidates:
                info[row["ImageID"]] = row
    blocked_groups = {r["group"] for r in blocked}
    blocked_source = {r.get("source_url") for r in blocked}
    blocked_authors = {author_key(r["author"]) for r in blocked if r.get("author")}
    blocked_hashes = {r["sha256"] for r in blocked}
    blocked_dhash = [int(r["dhash"], 16) for r in blocked if r.get("dhash")]

    def prepare(identifier):
        boxes = boxed.get(identifier, [])
        if any(int(row[key]) for row in boxes for key in ("IsGroupOf", "IsDepiction", "IsInside")):
            return identifier, "ambiguous group, depiction or inside"
        meta = info.get(identifier, {})
        license_url = meta.get("License", "").replace("http://", "https://").rstrip("/")
        author = meta.get("Author")
        landing = meta.get("OriginalLandingURL")
        profile = meta.get("AuthorProfileURL", "").replace("http://", "https://").rstrip("/")
        if not author or not landing or not profile or license_url != "https://creativecommons.org/licenses/by/2.0":
            return identifier, "attribution or CC BY 2.0 metadata unavailable"
        group = "openimages:" + profile
        if group in blocked_groups or landing in blocked_source or author_key(author) in blocked_authors:
            return identifier, "author or original URL occurs in an earlier partition"
        relative = Path("openimages") / split / f"{identifier}.jpg"
        url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{identifier}.jpg"
        try:
            download(url, data / relative)
            image = read_image(data / relative)
        except (OSError, subprocess.CalledProcessError):
            return identifier, "image unavailable or unreadable"
        checksum, visual_hash = digest(data / relative), dhash(image)
        if checksum in blocked_hashes or any(
            (int(visual_hash, 16) ^ other).bit_count() <= 4 for other in blocked_dhash
        ):
            return identifier, "exact or near-hash collision with an earlier partition"
        objects = [{
            "bbox": [float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")],
            "occluded": bool(int(row["IsOccluded"])), "truncated": bool(int(row["IsTruncated"])),
        } for row in boxes]
        return identifier, dict(id=identifier, group=group, split=split,
            image=relative.as_posix(), sha256=checksum, dhash=visual_hash,
            width=image.width, height=image.height, objects=objects, author=author,
            source_url=landing, download_url=url, license=license_url, title=meta.get("Title"))

    rows = []
    with ThreadPoolExecutor(8) as pool:
        selected = sorted(boxed) + sorted(verified_negative)[:40]
        for identifier, result in pool.map(prepare, selected):
            if isinstance(result, str):
                rejected[identifier] = result
            else:
                rows.append(result)
    write_json(inventory, dict(rows=rows, rejected=rejected, sources=sources,
                               candidate_negatives=len(verified_negative), candidate_positives=len(boxed),
                               csv_sha256={k: digest(directory / f"{k}.csv") for k in sources}))
    return rows


def commons_validation(data, inventory, design):
    rows = []
    for entry in read_json(design):
        relative = Path("commons") / f"{entry['id']}.jpg"
        path = data / relative
        if not path.exists():
            for url in (entry["image_url"], entry["original_url"]):
                try:
                    download(url, path)
                except subprocess.CalledProcessError:
                    continue
                if digest(path) == entry["sha256"]:
                    break
                path.unlink()
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Commons image hash mismatch for {entry['id']}")
        image = read_image(path)
        rows.append(dict(**entry, image=relative.as_posix(), split="validation",
                         group="commons:" + entry["author"].casefold(),
                         width=image.width, height=image.height, dhash=dhash(image)))
    write_json(inventory, dict(rows=rows, metadata="Wikimedia Commons file pages and extmetadata",
                               reference_status="single-pass visual draft, external to training"))
    return rows


def leakage(rows, distance=4):
    pairs = []
    for i, first in enumerate(rows):
        for second in rows[i+1:]:
            if first["split"] == second["split"]:
                continue
            same = [k for k in ("id", "group", "sha256", "source_url")
                    if first.get(k) and first[k] == second.get(k)]
            if first.get("author") and second.get("author") and (
                author_key(first["author"]) == author_key(second["author"])
            ):
                same.append("author")
            if first.get("dhash") and second.get("dhash"):
                if (int(first["dhash"], 16) ^ int(second["dhash"], 16)).bit_count() <= distance:
                    same.append("dhash")
            if same:
                pairs.append(dict(images=[first["id"], second["id"]], evidence=same))
    return pairs


def valid_box(box):
    return (len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def coco(rows):
    images, annotations = [], []
    for index, row in enumerate(rows, 1):
        w, h = row["width"], row["height"]
        images.append(dict(id=index, file_name=row["image"], width=w, height=h))
        for obj in row["objects"]:
            if not valid_box(obj["bbox"]):
                raise ValueError(f"Invalid box for {row['id']}")
            x, y, r, b = obj["bbox"]
            bbox = [x*w, y*h, (r-x)*w, (b-y)*h]
            attributes = {key: value for key, value in obj.items() if key != "bbox"}
            annotations.append(dict(id=len(annotations)+1, image_id=index, category_id=1,
                                    bbox=bbox, area=bbox[2]*bbox[3], iscrowd=0, attributes=attributes))
    return dict(images=images, annotations=annotations, categories=[dict(id=1, name="syringe")])


def train_rows(root):
    inventory = read_json(root / "artifacts/source_inventory.json")
    annotations = read_json(root / "artifacts/annotations.json")
    rows = [r for r in inventory if r["split"] == "train"]
    if set(annotations) != {r["id"] for r in rows}:
        raise ValueError("Draft annotations do not cover exactly the unique training images.")
    for row in rows:
        row["objects"] = annotations[row["id"]]["objects"]
        row["review_status"] = annotations[row["id"]]["status"]
        for obj in row["objects"]:
            if not valid_box(obj["bbox"]):
                raise ValueError(f"Invalid box for {row['id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "validation", "test", "verify", "coco"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    data, artifacts = root / "data", root / "artifacts"
    inventory = artifacts / "source_inventory.json"
    if args.action == "audit":
        if not args.source:
            parser.error("audit requires --source")
        audit(args.source, data, inventory)
        print(dict(Counter(row["split"] for row in read_json(inventory))))
        return
    originals = [row for row in read_json(inventory) if row["split"] == "train"]
    validation = artifacts / "validation_manifest.json"
    test = artifacts / "test_manifest.json"
    if args.action == "validation":
        rows = commons_validation(data, validation, artifacts / "commons_annotations.json")
    elif args.action == "test":
        freeze = read_json(artifacts / "freeze.json")
        if freeze["validation_sha256"] != digest(validation):
            raise ValueError("Validation manifest differs from the frozen selection.")
        rows = openimages_test(data, test, [*originals, *read_json(validation)["rows"]])
    elif args.action == "verify":
        rows = [*originals, *read_json(validation)["rows"], *read_json(test)["rows"]]
        violations = leakage(rows)
        missing = [r["id"] for r in rows if digest(data / r["image"]) !=
                   r.get("derived_sha256", r["sha256"])]
        source_status = "not checked"
        if args.source:
            before = {r["source"]: r["sha256"] for r in read_json(inventory)}
            after = {str(p.relative_to(args.source)): digest(p)
                     for p in args.source.rglob("*") if p.is_file()}
            if after != before:
                raise ValueError("Files or hashes in the original source directory differ from the audit.")
            source_status = f"{len(before)} original files match their initial SHA-256"
        if violations or missing:
            raise ValueError(dict(leakage=violations, hash_mismatch=missing))
        write_json(artifacts / "integrity.json", dict(images=dict(Counter(r["split"] for r in rows)),
                                                      leakage=[], hash_mismatch=[], source=source_status))
    else:
        rows = train_rows(root)
        write_json(artifacts / "train_coco.json", coco(rows))
        for path in (validation, test):
            if path.exists():
                write_json(artifacts / f"{path.stem.split('_')[0]}_coco.json",
                           coco(read_json(path)["rows"]))
    print(dict(images=len(rows), action=args.action))


if __name__ == "__main__":
    main()
