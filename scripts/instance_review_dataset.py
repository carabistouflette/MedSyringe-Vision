"""Combine separately reviewed public instances and assign whole-group partitions."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from medsyringe.data import author_key, digest, leakage, read_json, valid_box, write_json
from scripts.detection_public_benchmark import lexical_key, url_key
from scripts.instance_review_export import CROWD
from scripts.public_acquisition_partition import partition
from scripts.public_training_intake import close, raster_hashes

SUPPLEMENT = Path("data/instance-source-review/reviewed-annotations.json")
REVIEWED = Path("data/instance-review-dataset")
MINIMUM = 500
STRATA = ("hard-negative", "crowded", "syringe-positive", "source-verified-positive", "source-verified-negative")


def join(paths):
    entries = [(row, i == 0) for i, path in enumerate(paths) for row in read_json(path)["rows"]]
    by_author, groups = defaultdict(set), {}
    for row, _ in entries:
        if row["author"]:
            by_author[("author", author_key(row["author"]))].add(row["group"])
            by_author[("alias", lexical_key(row["author"]))].add(row["group"])
        by_author[("source", row["source_url"])].add(row["group"])
        groups[row["group"]] = row["group"]

    def root(group):
        while groups[group] != group:
            group = groups[group]
        return group

    def link(first, second):
        a, b = sorted((root(first), root(second)))
        groups[b] = a

    for projects in by_author.values():
        lead = min(projects)
        for group in projects:
            link(lead, group)

    retained, excluded, seen = [], [], set()
    urls, hashes, visual = {}, {}, []
    for row, rights in entries:
        key = row["id"]
        image = Path(row["image"])
        checksum = row["sha256"]
        if digest(image) != checksum or any(not valid_box(obj["bbox"]) for obj in row["objects"]):
            raise ValueError(f"Changed raster or invalid reviewed box geometry for {key}.")
        signatures = row.get("raster_signatures") or raster_hashes(image)
        asset = url_key(row["original_url"]) if row.get("original_url") else None
        # Source acquisition membership survives the removal of a related raster.
        related = hashes.get(checksum) or urls.get(asset)
        if related is None:
            related = next((old for old, old_signatures in visual if close(signatures, old_signatures)), None)
        if related is not None:
            excluded.append(dict(id=key, representative=related["id"], reason="Same URL, bytes or perceptual neighbor"))
            link(related["group"], row["group"])
            continue
        if key in seen:
            raise ValueError(f"One image identifier refers to two different verified rasters: {key}.")
        seen.add(key)
        if asset:
            urls[asset] = row
        hashes[checksum] = row
        visual.append((row, signatures))
        presence = "syringe-positive" if row["objects"] else "hard-negative"
        traits = [presence]
        if len(row["objects"]) >= 2:
            traits.append("crowded")
        if rights:
            traits.append("source-verified-positive" if row["objects"] else "source-verified-negative")
        retained.append({**row, "rights_status": "individual source verified" if rights else "publisher declared",
                         "eligible_for_fit": rights, "conditions": traits})

    represented = defaultdict(list)
    for row in retained:
        row["source_group"] = row["group"]
        row["group"] = root(row["group"])
        represented[row["group"]].append(row)
    blocked = {root(row["group"]) for row, _ in entries if row.get("previously_scored_acquisition")}
    roles, coverage = partition({g: r for g, r in represented.items() if g not in blocked}, STRATA, development_groups=2)
    roles.update({group: "historical-reference-only" for group in blocked})
    for row in retained:
        row["split"] = roles[row["group"]]
        row["eligible_for_fit"] &= row["split"] == "public-fit"
    if leakage(retained):
        raise ValueError("A public evaluation boundary shares a near duplicate with another role.")
    return retained, excluded, roles, coverage


def coco_export(rows):
    licenses = {key: i for i, key in enumerate(sorted({row["license"] for row in rows}), 1)}
    data = dict(info={"description": "MedSyringe Vision reviewed public instances"},
                licenses=[{"id": token, "name": name, "url": name if name.startswith("https://") else ""}
                          for name, token in licenses.items()],
                images=[], annotations=[], categories=[dict(id=1, name="syringe", supercategory="device")])
    for i, row in enumerate(rows, 1):
        w, h = (row.get("width"), row.get("height")) if row.get("width") else row["dimensions"]
        data["images"].append(dict(id=i, file_name=row["image"], width=w, height=h,
                                   license=licenses[row["license"]], split=row["split"]))
        for obj in row["objects"]:
            l, t, r, b = obj["bbox"]
            bw, bh = (r-l)*w, (b-t)*h
            data["annotations"].append(dict(id=len(data["annotations"]) + 1, image_id=i, category_id=1,
                                            bbox=[l*w, t*h, bw, bh], area=bw*bh, iscrowd=0))
    return data


def export(paths, target):
    data, excluded, roles, coverage = join([SUPPLEMENT, *paths])
    pos = [row for row in data if row["objects"]]
    summary = dict(target=MINIMUM, positive_images=len(pos), target_met=len(pos) >= MINIMUM,
                   negative_images=len(data)-len(pos), instance_boxes=sum(len(row["objects"]) for row in pos),
                   photos_with_multiple_syringes=sum(len(row["objects"]) >= 2 for row in pos),
                   rights_status=dict(Counter(row["rights_status"] for row in pos)),
                   positive_split=dict(Counter(row["split"] for row in pos)),
                   train_or_new_evaluation_positive_images=sum(row["split"] != "historical-reference-only" for row in pos),
                   split_groups=dict(Counter(roles.values())), excluded=len(excluded),
                   eligible_positive_fitting_images=sum(row["eligible_for_fit"] for row in pos),
                   source_verified_positive_split=dict(Counter(
                       row["split"] for row in pos if row["rights_status"] == "individual source verified")))
    write_json(target / "manifest.json", dict(rows=data, groups=roles, coverage=coverage,
               hard_holdout_sha256=digest(CROWD), excluded=excluded, summary=summary,
               sources={str(p): digest(p) for p in (SUPPLEMENT, *paths)},
               coordinates="Normalized xyxy on EXIF-oriented source raster"))
    write_json(target / "instances.coco.json", coco_export(data))
    write_json(target / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=REVIEWED)
    args = parser.parse_args()
    export(args.reviews, args.output)
