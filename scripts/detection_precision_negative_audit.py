"""Test the fixed precision policy on new photographer-disjoint lookalikes."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.parse import urlencode

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.evaluate import evaluate
from scripts.detection_consensus_corpus import MANIFEST as CONSENSUS, prior
from scripts.detection_crop_semantics import VisualSemantics
from scripts.detection_dental_evaluate import runner
from scripts.detection_hillclimb_experiment import ART, DATA, EXISTING, FLOOR
from scripts.detection_precision_confirmation import CAPTURE, LOCK, RESULT as REGRESSIONS, verify
from scripts.detection_public_benchmark import (
    API, ROOT, WORK, get_asset, lexical_key, metadata, near_any, order, request, sheet, url_key, withheld,
)
from scripts.detection_semantics_audit import apply_rules, crop_frame
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

CANDIDATES = WORK / "precision-confirmation-candidates.json"
REVIEW = ART / "precision_negative_review.json"
MANIFEST = ART / "precision_negative_manifest.json"
FREEZE = ART / "precision_negative_freeze.json"
RESULT = ART / "precision_negative_results.json"
CACHE = DATA / "proposals/precision-negative-comparison.json"
QUERIES = (
    'intitle:"blood collection" filetype:bitmap',
    'intitle:"intravenous cannula" filetype:bitmap',
    'intitle:"micropipette" filetype:bitmap',
    'intitle:"autoinjector" filetype:bitmap',
    'intitle:"pipette tip" filetype:bitmap',
    'intitle:"insulin pen" filetype:bitmap',
    'intitle:"dental mirror" filetype:bitmap',
    'intitle:"test tube" filetype:bitmap',
    'intitle:"applicator" filetype:bitmap',
    'intitle:"dental instruments" filetype:bitmap',
    'intitle:"dispenser" "laboratory" filetype:bitmap',
    'intitle:"mechanical pencil" filetype:bitmap',
    'intitle:"medical tubing" filetype:bitmap',
)


def protect(known, row):
    known["ids"].add(row["id"])
    known["titles"].add(lexical_key(row["title"]))
    known["authors"].update({lexical_key(row["author"]), *row.get("profile_keys", [])})
    known["hashes"].add(row["sha256"])
    known["difference"].add(int(row["dhash"], 16))
    known["urls"].update(url_key(row[k]) for k in ("source_url", "original_url", "image_url"))


def exposure():
    known = prior()
    for row in read_json(CONSENSUS)["rows"]:
        protect(known, row)
    return known


def overlaps(known, row, image):
    return withheld(row, known) or row["sha256"] in known["hashes"] or near_any(image, known["difference"])


def discover(limit):
    if FREEZE.exists() or CACHE.exists():
        raise ValueError("The negative set can no longer be changed.")
    known = exposure()
    groups = []
    for query in QUERIES:
        params = dict(action="query", generator="search", gsrlimit="35", gsrnamespace="6", gsrsearch=query,
                      prop="imageinfo", iiprop="url|size|mime|extmetadata|timestamp", iiurlwidth="960", format="json")
        try:
            pages = json.loads(request(API + "?" + urlencode(params), 8 * 1024 * 1024))
        except (OSError, ValueError) as exc:
            print("Unavailable query", query, exc, flush=True)
            continue
        rows = [row for page in pages.get("query", {}).get("pages", {}).values()
                if (row := metadata(page, query, "lookalike"))]
        groups.append(sorted(rows, key=lambda row: order(row["id"])))
    selected = []
    for level in zip_longest(*groups):
        for candidate in level:
            if candidate is None or withheld(candidate, known):
                continue
            asset = get_asset(candidate)
            if asset is None:
                continue
            row, picture = asset
            if overlaps(known, row, picture):
                continue
            protect(known, row)
            selected.append(row)
            if len(selected) == limit:
                break
        if len(selected) == limit:
            break
    write_json(CANDIDATES, dict(rows=selected))
    sheet(selected, "precision-negative-candidates", 4, 3)
    print("Candidate scenes", len(selected), flush=True)


def freeze():
    if FREEZE.exists() or CACHE.exists():
        raise ValueError("The negative confirmation has already started.")
    plan = verify()
    reviewed = read_json(REVIEW)
    if reviewed["candidate_sha256"] != digest(CANDIDATES):
        raise ValueError("Reference review and candidate images differ.")
    known = exposure()
    rows = []
    for row in read_json(CANDIDATES)["rows"]:
        confirmed = reviewed["confirmed"].get(row["id"])
        if confirmed is None:
            continue
        path = ROOT / row["image"]
        if digest(path) != row["sha256"] or overlaps(known, row, read_image(path)):
            raise ValueError(f"Duplicate or changed asset {row['id']}.")
        protect(known, row)
        rows.append({**row, "objects": [], "lookalike_type": confirmed, "split": "precision-negative-audit"})
    if len(rows) < 20 or len(rows) != len(reviewed["confirmed"]):
        raise ValueError("At least 20 newly reviewed negative source groups are required.")
    write_json(MANIFEST, dict(rows=rows, review_sha256=digest(REVIEW)))
    paths = (Path(__file__), MANIFEST, REVIEW, LOCK, REGRESSIONS, EXISTING, FLOOR)
    write_json(FREEZE, dict(
        inputs={str(path): digest(path) for path in paths},
        components=plan["components"],
        primary_endpoint="More newly audited scenes with zero accepted predictions and fewer false-positive boxes.",
        qualification="Both primary endpoints must improve, no negative scene may gain a false alarm, all fixed "
                      "positive-cohort matches and full-capture counts must be retained.",
    ))
    print("Frozen independent negative scenes", len(rows), flush=True)


def bound_inputs():
    lock = read_json(FREEZE)
    if any(digest(path) != expected for path, expected in lock["inputs"].items()):
        raise ValueError("The frozen sample or detector evidence has changed.")
    return lock, read_json(MANIFEST)["rows"]


def collect():
    configure_cpu()
    lock, rows = bound_inputs()
    key = digest(FREEZE)
    saved = read_json(CACHE) if CACHE.exists() else dict(freeze_sha256=key, frames={})
    if saved["freeze_sha256"] != key:
        raise ValueError("The inference cache is bound to another experiment.")
    remaining = [row for row in rows if row["id"] not in saved["frames"]]
    if not remaining:
        return
    detector, close, _ = runner("baseline")
    try:
        for row in remaining:
            old = detector(source(row))
            saved["frames"][row["id"]] = dict(before=old["ranked"], floor=old["floor"])
            write_json(CACHE, saved)
            print("Independent negatives", len(saved["frames"]), "/", len(rows), flush=True)
    finally:
        close()


def apply():
    configure_cpu()
    lock, rows = bound_inputs()
    saved = read_json(CACHE)
    if saved["freeze_sha256"] != digest(FREEZE) or set(saved["frames"]) != {r["id"] for r in rows}:
        raise ValueError("Collection in the Intel environment must finish first.")
    encoder = None
    for row in rows:
        frame = saved["frames"][row["id"]]
        old = dict(ranked=frame["before"], floor=frame["floor"])
        if "after" in frame:
            continue
        active = [p for p in old["ranked"] if p["score"] >= old["floor"]]
        begin = perf_counter()
        if active:
            if encoder is None:
                encoder = VisualSemantics()
            patches = crop_frame(encoder, source(row), dict(ranked=active, floor=old["floor"]))
            new, _ = apply_rules(old, patches, lock["components"])
        else:
            new = old["ranked"]
        frame.update(after=new, extra_seconds=perf_counter()-begin)
        write_json(CACHE, saved)


def measure():
    if RESULT.exists():
        raise ValueError("The untouched negative cohort has already been scored.")
    _, rows = bound_inputs()
    saved = read_json(CACHE)
    if (saved["freeze_sha256"] != digest(FREEZE) or set(saved["frames"]) != {r["id"] for r in rows}
        or any("after" not in f for f in saved["frames"].values())):
        raise ValueError("The predictions do not represent the complete frozen cohort.")
    floor = read_json(FLOOR)["cutoff"]
    previous, candidate = (
        evaluate(rows, {k: p[stage] for k, p in saved["frames"].items()}, floor)
        for stage in ("before", "after")
    )
    earlier_errors = {r["id"] for r in previous["false_positives"]}
    new_errors = {r["id"] for r in candidate["false_positives"]}
    regressions = read_json(REGRESSIONS)["outcomes"]
    retained = all(x["guard"]["passed"] and x["candidate"]["fp"] <= x["previous"]["fp"] and
                   (name not in CAPTURE or x["candidate"]["exact_count"] == x["previous"]["exact_count"])
                   for name, x in regressions.items())
    passed = retained and new_errors < earlier_errors and candidate["fp"] < previous["fp"]
    write_json(RESULT, dict(passed=passed, selection_sha256=digest(LOCK), freeze_sha256=digest(FREEZE),
                           regression_sha256=digest(REGRESSIONS), all_regression_guards_passed=retained,
                           previous=short(previous), candidate=short(candidate),
                           corrected_scene_ids=sorted(earlier_errors - new_errors),
                           new_false_alarm_scene_ids=sorted(new_errors - earlier_errors),
                           errors=dict(previous=previous["false_positives"], candidate=candidate["false_positives"]),
                           median_extra_seconds=median(r["extra_seconds"] for r in saved["frames"].values())))
    print("Independent negative confirmation", passed, short(previous), short(candidate), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("discover", "freeze", "collect", "apply", "measure"))
    parser.add_argument("--limit", type=int, default=42)
    args = parser.parse_args()
    if args.action == "discover":
        discover(args.limit)
    else:
        {"freeze": freeze, "collect": collect, "apply": apply, "measure": measure}[args.action]()
