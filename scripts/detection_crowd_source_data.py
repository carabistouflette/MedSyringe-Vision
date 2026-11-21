"""Screen attributable, acquisition-disjoint public photographs for crowded scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
import json
from pathlib import Path
from urllib.parse import urlencode

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from scripts.detection_context_data import RELEASE as CONTEXT
from scripts.detection_device_scene_data import MANIFEST as DEVICE
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_medbin_data import MANIFEST as MEDBIN, history
from scripts.detection_precision_negative_audit import overlaps, protect
from scripts.detection_public_benchmark import API, get_asset, metadata, order, request, sheet, withheld

POOL = DATA / "commons-crowds-screen.json"
MODERN_POOL = DATA / "commons-crowds-modern-screen.json"
REVIEW = ART / "commons_crowds_reference_review.json"
MANIFEST = ART / "commons_crowds_reserve.json"
QUERIES = (
    'intitle:syringes vaccine "prepared" filetype:bitmap',
    'intitle:syringes "tray" filetype:bitmap',
    'intitle:syringes "several" filetype:bitmap',
    'intitle:syringes "table" filetype:bitmap',
    'intitle:syringes "prefilled" filetype:bitmap',
    'intitle:syringes "vials" filetype:bitmap',
    'intitle:syringes "collection" filetype:bitmap',
    'intitle:syringes "sizes" filetype:bitmap',
    'intitle:spritzen medizintechnik filetype:bitmap',
    'intitle:seringues "vaccin" filetype:bitmap',
)
CAMPAIGNS = (
    '"syringes" "prepared" covid filetype:bitmap',
    '"syringes" "Kadena" "vaccine" filetype:bitmap',
    '"prefilled syringes" "influenza" filetype:bitmap',
    '"syringes" "preparation" "COVID-19" filetype:bitmap',
    '"syringes" "waiting" "vaccine" filetype:bitmap',
    '"vaccination" "filled syringes" filetype:bitmap',
)
TITLES = (
    "201228-O-YB478-140.jpg", "Moderna COVID-19 Vaccination 6473801.jpg",
    "Influenza vaccine syringes.jpg", "Fluzone intradermal.jpg",
    "Syringes rest on a clean table in a clinic.jpg", "Syringes.jpg",
    "US Navy 021019-N-9593M-007 Flu shot preparations.jpg", "Preparations for vaccination.jpg",
    "Clexane box of prefilled syringes.jpg", "Aripiprazole 400mg injection.jpg",
    "Syringe with Certolizumab pegol-1800.jpg",
)


def used_sources():
    known = history()
    for dataset in (MEDBIN, DEVICE, CONTEXT):
        for row in read_json(dataset)["rows"]:
            protect(known, {**row, "original_url": row.get("original_url", row["image_url"])})
    return known


def lookup(query):
    base = dict(action="query", prop="imageinfo", iiprop="url|size|mime|extmetadata|timestamp",
                iiurlwidth=1280, format="json")
    if query == "reviewed titles":
        params = dict(**base, titles="|".join(f"File:{name}" for name in TITLES))
    else:
        params = dict(**base, generator="search", gsrlimit=45, gsrnamespace=6, gsrsearch=query)
    try:
        raw = json.loads(request(API+"?"+urlencode(params), 12*1024*1024))
    except (OSError, ValueError) as exc:
        print("Catalogue search did not return results", query, str(exc), flush=True)
        return []
    pages = raw.get("query", {}).get("pages", {}).values()
    found = [row for page in pages if (row := metadata(page, query, "crowded syringe scene"))]
    return sorted(found, key=lambda r: (not r["title"].casefold().startswith("file:syring"), order(r["id"])))


def discover(limit, modern):
    target = MODERN_POOL if modern else POOL
    if MANIFEST.exists() or target.exists():
        raise ValueError("The crowded source pool is already saved.")
    known = used_sources()
    records = []
    terms = ("reviewed titles", *CAMPAIGNS) if modern else QUERIES
    with ThreadPoolExecutor(max_workers=2) as threads:
        entries = list(threads.map(lookup, terms))
    for offers in zip_longest(*entries):
        for item in offers:
            if item is None or withheld(item, known):
                continue
            downloaded = get_asset(item)
            if downloaded is None:
                continue
            row, picture = downloaded
            if overlaps(known, row, picture):
                continue
            protect(known, row)
            records.append(row)
            if len(records) == limit:
                break
        if len(records) == limit:
            break
    groups = len({r["group"] for r in records})
    write_json(target, dict(rows=records, queries=terms, explicit_titles=TITLES if modern else [], studied_groups=groups))
    sheet(records, "commons-crowds-modern" if modern else "commons-crowds-screen", 4, 3)
    print("Crowd sources staged", len(records), groups, flush=True)


def freeze():
    if MANIFEST.exists():
        raise ValueError("The independently reviewed source pool is sealed.")
    decision = read_json(REVIEW)
    pool = MODERN_POOL if decision.get("pool") == "modern" else POOL
    if decision["pool_sha256"] != digest(pool):
        raise ValueError("The reference decision does not match the source pool.")
    records, known = [], used_sources()
    for row in read_json(pool)["rows"]:
        evidence = decision["rows"].get(row["id"])
        if evidence is None:
            continue
        picture = Path("data") / row["image"]
        if (digest(picture) != row["sha256"] or withheld(row, known) or overlaps(known, row, read_image(picture))
            or not all(valid_box(v["bbox"]) for v in evidence["objects"])):
            raise ValueError(f"Reviewed image does not meet the frozen source criteria: {row['id']}.")
        protect(known, row)
        records.append({**row, **evidence, "split": "commons-crowds-reserve"})
    multi = sum(len(r["objects"]) >= 2 for r in records)
    if len(records) != len(decision["rows"]) or multi < 2:
        raise ValueError("At least two fully countable multi-device photographs from different groups are required.")
    write_json(MANIFEST, dict(rows=records, review_sha256=digest(REVIEW),
                            protocol="One scene per identified photographer group. No scene or candidate detector "
                                     "output was used in annotation. Group and overlap checks separate these sources "
                                     "from every previously exposed cohort. This small cohort is a descriptive "
                                     "overlap challenge and cannot establish a population-level estimate."))
    print("Independent crowd photographs sealed", len(records), multi, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("discover", "freeze"))
    parser.add_argument("--limit", type=int, default=58)
    parser.add_argument("--modern", action="store_true")
    args = parser.parse_args()
    if args.action == "discover":
        discover(args.limit, args.modern)
    else:
        freeze()
