"""Keep public project groups exposed by an earlier audit out of expanded fitting."""
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from medsyringe.data import digest, read_json, write_json

ART = Path("artifacts/detection_hillclimb")
INPUTS = (Path("data/public-syringe-dataset/manifest.json"),
          Path("data/public-bulk-syringe-collection/manifest.json"))
PROTECTED = tuple(ART / name for name in (
    "device_scene_manifest.json", "device_context_manifest.json", "medbin_held_out_manifest.json",
    "commons_crowds_reserve.json", "crowd_field_manifest.json"))
OUTPUT = ART / "expanded_public_catalog_exposure.json"


def universe_project(address):
    uri = urlsplit(address)
    segments = uri.path.strip("/").split("/")
    if uri.hostname != "universe.roboflow.com" or len(segments) < 2:
        return None
    return "/".join(segments[:2]).lower()


def protected_projects():
    protected = {}
    for path in PROTECTED:
        for row in read_json(path)["rows"]:
            project = universe_project(row.get("source_url") or "")
            if project:
                protected.setdefault(project, set()).add(path.name)
    return protected


def project_links(row, protected):
    project = universe_project(row["source_url"])
    alias = row.get("candidate_group", row["group"]).removeprefix("roboflow:")
    return sorted({key for key in (project, alias) if key in protected})


def run():
    protected = protected_projects()
    reports = {}
    for path in INPUTS:
        records = read_json(path)["rows"]
        matches = {}
        for row in records:
            project = universe_project(row["source_url"])
            links = project_links(row, protected)
            if links:
                manifests = sorted({name for key in links for name in protected[key]})
                matches[row["id"]] = dict(project=project, matched_projects=links, group=row["group"],
                                          previous_manifests=manifests)
        groups = {proof["group"] for proof in matches.values()}
        pending = [row["id"] for row in records if row["group"] in groups and row["id"] not in matches]
        reports[str(path)] = dict(manifest_sha256=digest(path), rows=len(records), known_overlap=matches,
                                  retained_for_separate_review_only=pending,
                                  blocked_by_project=dict(Counter(item["project"] for item in matches.values())),
                                  blocked_by_group=sorted(groups))
    write_json(OUTPUT, dict(catalogs=reports, protected={str(p): digest(p) for p in PROTECTED},
                            rule="A public project eligible for a prior development or reserve experiment remains "
                                 "excluded as a whole from new fitting. Unblocked records still need annotation, "
                                 "cross-catalog pixel screening and individual lineage review."))
    print({path: report["blocked_by_project"] for path, report in reports.items()}, flush=True)


if __name__ == "__main__":
    run()
