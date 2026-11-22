"""Freeze whole-acquisition roles after visual review of the public intake."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from medsyringe.data import digest, leakage, read_json, valid_box, write_json
from scripts.public_catalog_exposure_audit import project_links, protected_projects

CONDITIONS = ("crowded", "transparent", "occluded", "hard-negative", "readable-markings", "visible-piston")
ROLES = ("public-reserve", "public-development", "public-fit")
SEED = "medsyringe-public-acquisitions-v1"


def priority(group):
    return sha256(f"{SEED}/{group}".encode()).hexdigest()


def partition(groups, conditions=CONDITIONS, development_groups=1):
    support = {name: {group for group, rows in groups.items() if any(
        name in row["conditions"] for row in rows)} for name in conditions}
    assigned = {}
    coverage = defaultdict(dict)
    for role in ROLES[:-1]:
        for condition, available in sorted(support.items(), key=lambda item: (len(item[1]), item[0])):
            if role == ROLES[0]:
                target = 2 if len(available) >= 5 else int(len(available) >= 3)
            else:
                target = min(development_groups, max(0, len(available) - 3))
            while sum(assigned.get(group) == role for group in available) < target:
                eligible = [group for group in available if group not in assigned and all(
                    sum(g not in assigned and g != group for g in cohort) >= min(len(cohort), 2)
                    for cohort in support.values() if group in cohort)]
                if not eligible:
                    break
                group = min(eligible, key=priority)
                assigned[group] = role
            coverage[condition][role] = sum(assigned.get(g) == role for g in available)
    for group in groups:
        assigned.setdefault(group, "public-fit")
    for condition, candidates in support.items():
        for role in ROLES:
            coverage[condition][role] = sum(assigned[g] == role for g in candidates)
    return assigned, dict(coverage)


def make(intake, review, output):
    if output.exists():
        raise FileExistsError("Choose a new path for the frozen public partition.")
    source = read_json(intake)
    confirmation = read_json(review)
    if confirmation["intake_sha256"] != digest(intake):
        raise ValueError("The annotation review does not match the proposed intake.")
    inventory = {row["id"]: row for row in source["rows"]}
    for filename, expected in source["protected_fil_rouge"].items():
        if digest(filename) != expected:
            raise ValueError("The original Fil Rouge reference inventory changed.")
    if not set(confirmation["cases"]).issubset(inventory):
        raise ValueError("A review refers to an image outside the public intake.")
    protected_groups = protected_projects()
    grouping = defaultdict(list)
    for identifier, result in confirmation["cases"].items():
        if result["decision"] != "accepted":
            continue
        row = inventory[identifier]
        if (row["split"] != "annotation-review" or not result.get("reviewer") or not result.get("license_checked")
                or digest(row["image"]) != row["raster_sha256"]):
            raise ValueError(f"Approved metadata or local raster is missing for {identifier}.")
        if project_links(row, protected_groups):
            raise ValueError(f"Public project or provenance group for {identifier} is already protected.")
        objects = result["objects"]
        if any(not valid_box(obj["bbox"]) or obj["kind"] != "syringe" for obj in objects):
            raise ValueError(f"Invalid or ambiguous reference boxes for {identifier}.")
        conditions = result["conditions"]
        if "crowded" in conditions and len(objects) < 2 or "hard-negative" in conditions and objects:
            raise ValueError(f"Inconsistent reference condition for {identifier}.")
        grouping[row["protected_acquisition"]].append(
            {**row, "group": row["protected_acquisition"], "annotations_reviewed": True,
             "reviewer": result["reviewer"], "objects": objects, "conditions": conditions})
    roles, condition_groups = partition(grouping)
    allocation = [{**r, "split": roles[group]} for group, rows in grouping.items() for r in rows]
    if not allocation or leakage(allocation):
        raise ValueError("The reviewed source pool is empty or split boundaries overlap.")
    write_json(output, dict(rows=allocation, acquisition_roles=roles, condition_groups=condition_groups,
                            review_sha256=digest(review), intake_sha256=digest(intake),
                            not_reviewed=len(inventory.keys() - confirmation["cases"].keys()),
                            protected_fil_rouge=source["protected_fil_rouge"], seed=SEED,
                            method="Keep acquisition groups intact. Prioritize rare deployment conditions for "
                                   "reserve coverage while retaining distinct fitting groups. Sparse conditions "
                                   "are reported explicitly. Only reviewed and license-checked rows are admitted."))
    print("Approved public photos", len(allocation), "in", len(grouping), "acquisitions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intake", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make(args.intake, args.review, args.output)
