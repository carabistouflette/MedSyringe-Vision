"""Bind human-reviewed instance boxes to public images and export reviewed COCO."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.detection_public_benchmark import lexical_key, url_key
from scripts.instance_next_batch import CATALOG, OUT
from scripts.public_catalog_exposure_audit import project_links, protected_projects
from scripts.public_training_intake import close, raster_hashes, references

CROWD = Path("artifacts/detection_hillclimb/crowd_next_reference_manifest.json")
PRIOR_CROWD = Path("artifacts/detection_hillclimb/crowded_sam_prospective_reference.json")
AUDIT = Path("artifacts/detection_hillclimb/public_instance_review_batch.json")


def current_crowd():
    crowd = read_json(CROWD)["images"]
    authors = {lexical_key(x) for r in crowd for x in [r["author"], *r.get("author_aliases", [])] if x}
    urls = {url_key(r[field]) for r in crowd for field in ("source_url", "original_url", "asset_url") if r.get(field)}
    photos = {r["sha256"] for r in crowd}
    visual = set().union(*(raster_hashes(Path("data") / r["file"]) for r in crowd))
    return authors, urls, photos, visual


def previously_scored_groups():
    rows = read_json(PRIOR_CROWD)["rows"]
    return ({r["group"] for r in rows}, {r["source_url"] for r in rows}, {r["sha256"] for r in rows},
            set().union(*(raster_hashes(Path("data") / r["image"]) for r in rows)))


def checked_raster(candidate, source, cache):
    raw = Path(candidate["image"])
    if not raw.is_file():
        raw = Path("data") / raw
    expected = source["sha256"]
    if candidate["sha256"] != expected or digest(raw) != expected:
        raise ValueError(f"Reviewed raster has changed for {source['id']}.")
    cached = cache / f"{expected}.json"
    pixels = set(read_json(cached)) if cached.exists() else raster_hashes(raw)
    if not cached.exists():
        write_json(cached, sorted(pixels))
    return raw, pixels


def export(queue, catalog, review_paths, output):
    files = read_json(catalog)
    inventory = {r["id"]: r for r in files["rows"]}
    bank = read_json(queue / "proposals.json")
    if bank["binding"]["catalog"] != digest(catalog):
        raise ValueError("Proposals refer to another catalog.")
    review_sheets = sorted(queue.glob("review-*.json"))
    placements = {r["index"]: r for sheet in review_sheets for r in read_json(sheet)["rows"]}
    if len(placements) != len(bank["rows"]) or {r["id"] for r in placements.values()} != set(bank["rows"]):
        raise ValueError("Exactly one tile is required per proposal photograph.")
    seen = set()
    selected = []
    for path in review_paths:
        review = read_json(path)
        if not review["reviewer"] or not review["basis"]:
            raise ValueError("Each approval needs its review source.")
        for i in review["approved_indices"]:
            if i not in placements or i in seen:
                raise ValueError(f"Missing or conflicting approval for tile {i}.")
            seen.add(i)
            placed = placements[i]
            selected.append((placed, review["reviewer"], None))
        for key, correction in review.get("corrected", {}).items():
            i = int(key)
            if i not in placements or i in seen:
                raise ValueError(f"Missing or conflicting correction for tile {i}.")
            seen.add(i)
            selected.append((placements[i], review["reviewer"], correction))
    previous, private, private_visual = references()
    authors, urls, reserved, reserve_visual = current_crowd()
    prior_groups, prior_urls, prior_hashes, prior_visual = previously_scored_groups()
    selected.sort(key=lambda row: row[0]["index"])
    known_projects = protected_projects()
    rows, rejected = [], {}
    accepted_visual, accepted_rasters = set(), set()
    cache = queue / "raster-signatures-v1"
    with ThreadPoolExecutor(max_workers=3) as worker:
        rasters = list(worker.map(lambda entry: checked_raster(bank["rows"][entry[0]["id"]],
                                                               inventory[entry[0]["id"]], cache), selected))
    for (placed, reviewer, corrected), (raw, pixels) in zip(selected, rasters, strict=True):
        key = placed["id"]
        source, candidate = inventory[key], bank["rows"][key]
        if (source["sha256"] in private | previous["hashes"] | reserved | prior_hashes or project_links(source, known_projects)
                or lexical_key(source["author"]) in authors
                or any(url_key(source[k]) in urls for k in ("source_url", "original_url"))
                or close(pixels, private_visual | previous["difference"] | reserve_visual | prior_visual)):
            rejected[key] = "Protected reference raster, author, project or visual neighbor"
            continue
        if source["sha256"] in accepted_rasters or close(pixels, accepted_visual):
            rejected[key] = "Conservative same-catalog visual duplicate of another approved review"
            continue
        if corrected is None:
            boxes = [candidate["proposals"][i]["bbox"] for i in placed["proposed_indices"]]
        else:
            boxes = corrected
        if not boxes or any(not valid_box(box) for box in boxes) or len({tuple(box) for box in boxes}) != len(boxes):
            raise ValueError(f"Invalid reviewed instance geometry for {key}.")
        prior_group = source["group"] in prior_groups or source["source_url"] in prior_urls
        rows.append({**source, "image": str(raw), "reviewer": reviewer, "review_index": placed["index"],
                     "raster_signatures": sorted(pixels),
                     "previously_scored_acquisition": prior_group,
                     "training_status": "historical reference family" if prior_group else "pending individual image lineage review",
                     "objects": [dict(kind="syringe", bbox=box) for box in boxes]})
        accepted_rasters.add(source["sha256"])
        accepted_visual.update(pixels)
    coco = dict(images=[], annotations=[], categories=[dict(id=1, name="syringe")])
    for i, row in enumerate(rows, 1):
        w, h = (row.get("width"), row.get("height")) if row.get("width") else row["dimensions"]
        coco["images"].append(dict(id=i, file_name=row["image"], width=w, height=h, license=row["license"]))
        for obj in row["objects"]:
            l, t, r, b = obj["bbox"]
            a, z = (r - l) * w, (b - t) * h
            coco["annotations"].append(dict(id=len(coco["annotations"]) + 1, image_id=i, category_id=1,
                                            bbox=[l * w, t * h, a, z], area=a * z, iscrowd=0))
    write_json(output, dict(status="Visually reviewed boxes, grouped and deduplicated. Training admission pending.",
                            catalog=digest(catalog), proposals=digest(queue / "proposals.json"),
                            reviews={str(p): digest(p) for p in review_paths},
                            review_sheets={str(p): digest(p) for p in review_sheets},
                            protected_crowd=digest(CROWD), protected_prior_crowd=digest(PRIOR_CROWD),
                            approvals=len(seen), rows=rows, exclusions=rejected,
                            per_acquisition=dict(Counter(row["group"] for row in rows)),
                            instances=len(coco["annotations"])))
    write_json(queue / "reviewed-instances.coco.json", coco)
    print("Approved boxes:", len(rows), "photos", len(coco["annotations"]), "instances", len(rejected), "excluded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", nargs="+", type=Path)
    parser.add_argument("--queue", type=Path, default=OUT)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, default=AUDIT)
    args = parser.parse_args()
    export(args.queue, args.catalog, args.reviews, args.output)
