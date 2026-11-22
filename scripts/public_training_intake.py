"""Join separately harvested public sources into a review queue with protected references."""
import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageOps

from medsyringe.data import dhash, digest, read_image, read_json, write_json
from scripts.public_catalog_exposure_audit import project_links, protected_projects
from scripts.detection_medbin_data import history
from scripts.detection_public_benchmark import lexical_key, url_key

FIL_ROUGE = (Path("artifacts/source_inventory.json"), Path("artifacts/expanded/inventory.json"))
EXTRA_PROTECTED = tuple(Path("artifacts/detection_hillclimb") / f"{name}.json" for name in (
    "device_scene_manifest", "device_context_manifest", "commons_crowds_reserve", "crowd_field_manifest"))


def records(file):
    payload = read_json(file)
    return payload if isinstance(payload, list) else payload["rows"]


def normalized_url(url):
    if not url:
        return None
    value = urlsplit(url)
    if value.scheme not in ("https", "http") or not value.hostname:
        return None
    return value.hostname.casefold() + unquote(value.path).rstrip("/")


def raster_hashes(path):
    image = read_image(path).convert("L")
    hashes = set()
    for turn in (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270):
        patch = image if turn is None else image.transpose(turn)
        hashes.update(int(dhash(view), 16) for view in (patch, ImageOps.mirror(patch)))
    return hashes


def close(hashes, protected):
    return any((first ^ second).bit_count() <= 4 for first in hashes for second in protected)


def references():
    protected = history()
    for manifest in EXTRA_PROTECTED:
        if manifest.exists():
            for item in records(manifest):
                protected["hashes"].add(item["sha256"])
                protected["difference"].add(int(item["dhash"], 16))
                protected["urls"].update(url_key(item[k]) for k in ("source_url", "original_url") if item.get(k))
    hashes = {row[field] for file in FIL_ROUGE for row in records(file)
              for field in ("sha256", "derived_sha256") if row.get(field)}
    difference = {int(row["dhash"], 16) for file in FIL_ROUGE for row in records(file) if row.get("dhash")}
    return protected, hashes, difference


def intake(inputs, output):
    if output.exists():
        raise FileExistsError("Write the combined intake to a new path.")
    existing, fil_rouge, private_difference = references()
    protected_groups = protected_projects()
    candidates, rejected = [], []
    for manifest in sorted(inputs):
        for row in records(manifest):
            provenance = [row.get(key) for key in ("author", "license", "group", "source_url", "original_url")]
            image = Path(row.get("image") or "")
            if not image.is_absolute() and not image.is_file():
                image = next((root / image for root in (manifest.parent, Path("data"))
                              if (root / image).is_file()), image)
            identifier = f"{manifest.parent.name}:{row.get('id')}"
            if not all(provenance) or not image.is_file() or not normalized_url(row["original_url"]):
                rejected.append(dict(id=identifier, cause="Missing local raster or acquisition provenance"))
                continue
            if overlaps := project_links(row, protected_groups):
                rejected.append(dict(id=identifier, cause="Previously evaluated public project or provenance group",
                                     protected_projects=overlaps))
                continue
            checksum = digest(image)
            if checksum not in (row.get("sha256"), row.get("derived_sha256")):
                rejected.append(dict(id=identifier, cause="Local raster does not match its manifest"))
                continue
            hashes = raster_hashes(image)
            reasons = []
            if checksum in fil_rouge or row.get("sha256") in fil_rouge or close(hashes, private_difference):
                reasons.append("Protected Fil Rouge match or visual proximity")
            archived_url = any(url_key(row[k]) in existing["urls"] for k in ("source_url", "original_url"))
            contributor = ("commons.wikimedia.org" in row["source_url"] and
                           lexical_key(row["author"]) in existing["authors"])
            if (checksum in existing["hashes"] or row.get("sha256") in existing["hashes"] or archived_url
                    or contributor or close(hashes, existing["difference"])):
                reasons.append("Previously used public or simulator raster")
            if reasons:
                rejected.append(dict(id=identifier, cause=reasons))
                continue
            candidates.append(dict(item={**row, "id": identifier, "image": str(image),
                                         "raster_sha256": checksum, "admission": "awaiting visual review"},
                                   hashes=hashes, url=normalized_url(row["original_url"])))
    groups = defaultdict(set)
    contributor_groups = defaultdict(set)
    duplicate_edges = defaultdict(set)
    for index, candidate in enumerate(candidates):
        row = candidate["item"]
        group = row["group"]
        groups[group].add(group)
        contributor = lexical_key(row["author"])
        for other_group in contributor_groups[contributor]:
            groups[group].add(other_group)
            groups[other_group].add(group)
        contributor_groups[contributor].add(group)
        for other, previous in enumerate(candidates[:index]):
            old = previous["item"]
            if (row["raster_sha256"] == old["raster_sha256"] or candidate["url"] == previous["url"]
                    or close(candidate["hashes"], previous["hashes"])):
                groups[group].add(old["group"])
                groups[old["group"]].add(group)
                duplicate_edges[index].add(other)
                duplicate_edges[other].add(index)
    unique = {}
    for group in groups:
        if group in unique:
            continue
        component, pending = set(), [group]
        while pending:
            current = pending.pop()
            if current not in component:
                component.add(current)
                pending.extend(groups[current] - component)
        alias = min(component)
        unique.update({item: alias for item in component})
    queue, assigned = [], set()
    for index, item in enumerate(candidates):
        if index in assigned:
            continue
        component, pending = set(), [index]
        while pending:
            current = pending.pop()
            if current not in component:
                component.add(current)
                pending.extend(duplicate_edges[current] - component)
        assigned.update(component)
        retained = max(component, key=lambda k: (candidates[k]["item"].get("width", 0) or 0, -k))
        row = candidates[retained]["item"]
        rejected.extend(dict(id=candidates[i]["item"]["id"], cause="Cross-collection duplicate candidate",
                             retained=row["id"]) for i in sorted(component - {retained}))
        queue.append({**row, "protected_acquisition": unique[row["group"]],
                      "candidate_group": row["group"], "split": "annotation-review"})
    write_json(output, dict(rows=queue, rejected=rejected,
                            sources={str(file): digest(file) for file in inputs},
                            protected_fil_rouge={str(file): digest(file) for file in FIL_ROUGE},
                            acquisitions=len({row["protected_acquisition"] for row in queue}),
                            protocol="Candidate rasters are eligible for review only. Verify annotation and license "
                                     "before fitting. Reserve acquisition groups intact. Fil Rouge is blocked by "
                                     "image hashes and transformed visual fingerprints."))
    print("Public intake", len(queue), "review candidates in",
          len({r["protected_acquisition"] for r in queue}), "acquisition groups", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    intake(args.manifest, args.output)
