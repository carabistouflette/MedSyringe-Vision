"""Materialize public Roboflow Universe positive image records from catalog snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from medsyringe.data import dhash, digest, read_json, write_json
from scripts.experiments.public_data.detection_public_benchmark import sheet
from scripts.experiments.public_data.public_training_intake import close, raster_hashes, references

ROOT = Path("data/public-universe-scale")
CATALOG = ROOT / "catalog.json"
MANIFEST = ROOT / "manifest.json"
ASSETS = ROOT / "images"
MAX_BYTES = 15 * 1024 * 1024


def download(source, item, assets):
    original = item["url"]
    url = urlsplit(original)
    identifier = item["id"]
    if url.scheme != "https" or url.hostname != "source.roboflow.com" or not identifier.isalnum():
        return None, "Unsupported public asset address"
    thumbnail = original.removesuffix("/original.jpg") + "/thumb.jpg"
    target = assets / f"{identifier}.jpg"
    chosen = thumbnail
    try:
        if not target.exists():
            error = "Asset could not be fetched"
            for location in (thumbnail, original):
                try:
                    request = Request(location, headers={"User-Agent": "MedSyringe-Vision/1.0 research dataset"})
                    with urlopen(request, timeout=18) as response:
                        payload = response.read(MAX_BYTES + 1)
                    if len(payload) > MAX_BYTES:
                        error = "Asset exceeds local collection size bound"
                        break
                    with Image.open(BytesIO(payload)) as probe:
                        if probe.format != "JPEG" or probe.width * probe.height > 30_000_000:
                            error = "Unsupported raster"
                            break
                        probe.verify()
                    target.write_bytes(payload)
                    chosen = location
                    break
                except HTTPError as problem:
                    error = f"HTTP {problem.code}"
                    if problem.code in (401, 403, 429):
                        return None, error
            if not target.exists():
                return None, error
        with Image.open(target) as probe:
            width, height = probe.size
            visual_hash = dhash(probe)
        return dict(id=identifier, image=str(target.relative_to("data")), sha256=digest(target), dhash=visual_hash,
                    width=width, height=height, source_url=source["url"], original_url=original, image_url=chosen,
                    author=source["credit"], license=source["license"], group=source["url"], title=source["name"],
                    publisher_split=item.get("split"), annotations=item["annotations"],
                    admission="publisher-positive, visual instance annotation pending"), None
    except (OSError, URLError, ValueError, Image.DecompressionBombError) as error:
        return None, type(error).__name__


def positives(items, classes=("syringe",)):
    labels = {k.casefold() for k in classes}
    return [item for item in items if item.get("mediaType") == "image" and any(
        k.strip().casefold() in labels and n for k, n in item.get("annotations", {}).get("classes", {}).items())]


def collect(quota, root=ROOT, excluded_manifests=()):
    manifest = root / "manifest.json"
    catalog = root / "catalog.json"
    assets = root / "images"
    if manifest.exists():
        raise FileExistsError("The catalog snapshot has already been materialized.")
    assets.mkdir(parents=True, exist_ok=True)
    known, originals, differences = references()
    for previous in excluded_manifests:
        for r in read_json(previous)["rows"]:
            known["hashes"].add(r["sha256"])
            known["difference"].add(int(r["dhash"], 16))
    report, accepted, rejected = read_json(catalog), [], []
    by_source = {}
    for source in report["projects"]:
        distinct = {entry["id"]: entry for entry in positives(
            source["items"], source.get("positive_classes", ("syringe",)))}
        local = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            ordered = list(distinct.values())
            for start in range(0, len(ordered), 24):
                batch = ordered[start:start + 24]
                for item, (row, issue) in zip(batch, pool.map(lambda item: download(source, item, assets), batch),
                                               strict=True):
                    if row is None:
                        rejected.append(dict(id=item["id"], source=source["url"], reason=issue))
                        continue
                    path = Path("data") / row["image"]
                    transformed = raster_hashes(path)
                    if row["sha256"] in originals or close(transformed, differences):
                        rejected.append(dict(id=row["id"], source=source["url"], reason="Fil Rouge provenance collision"))
                        continue
                    if row["sha256"] in known["hashes"] or close(transformed, known["difference"]):
                        rejected.append(dict(id=row["id"], source=source["url"], reason="Existing or duplicate photo"))
                        continue
                    local.append(row)
                    known["hashes"].add(row["sha256"])
                    known["difference"].update(transformed)
                if len(local) >= quota:
                    break
        accepted.extend(local)
        by_source[source["url"]] = dict(catalog_rows=len(source["items"]), publisher_positive=len(distinct),
                                        candidate_photos=len(local))
        if local:
            sheet(local, f"universe-large-{source['url'].split('/')[-1]}", 6, 5)
        print(source["name"], len(local), "checksummed positive candidates", flush=True)
    write_json(manifest, dict(rows=accepted, excluded=rejected, sources=by_source, catalog_sha256=digest(catalog),
                              prior_manifests={str(p): digest(p) for p in excluded_manifests},
                              status="Provider-labeled candidate photos. Verify provenance, distinct acquisition "
                                     "groups, and every proposed training or evaluation label."))
    print("Unique publisher-positive local photos", len(accepted), flush=True)


def verify(root=ROOT):
    evidence = read_json(root / "manifest.json")
    if evidence["catalog_sha256"] != digest(root / "catalog.json"):
        raise ValueError("The provider catalog changed.")
    if any(digest(Path("data") / r["image"]) != r["sha256"] for r in evidence["rows"]):
        raise ValueError("A collected photo is missing or differs.")
    print("Verified", len(evidence["rows"]), "local provider-positive photos", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "verify"))
    parser.add_argument("--quota", type=int, default=125)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "collect":
        collect(args.quota, args.root, args.exclude_manifest)
    else:
        verify(args.root)
