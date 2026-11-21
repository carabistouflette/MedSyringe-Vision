"""Prepare a fresh, source-disjoint audit for the restricted recovery policy."""
import argparse
from itertools import zip_longest
import json
from urllib.parse import urlencode

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from scripts.detection_dental_transfer_data import MANIFEST as DENTAL, previous
from scripts.detection_hillclimb_experiment import ART
from scripts.detection_precision_negative_audit import overlaps, protect
from scripts.detection_public_benchmark import API, ROOT, WORK, get_asset, metadata, order, request, sheet, withheld

POOL = WORK / "empty-scene-audit-pool.json"
REFERENCES = ART / "empty_scene_audit_review.json"
MANIFEST = ART / "empty_scene_audit_manifest.json"
QUERIES = (
    'intitle:vaccination influenza filetype:bitmap',
    'intitle:"vaccine syringe" filetype:bitmap',
    'intitle:"syringe" "vial" filetype:bitmap',
    'intitle:"filling" syringe filetype:bitmap',
    'intitle:"injection" medical filetype:bitmap',
    'intitle:"syringe" "work" filetype:bitmap',
    'intitle:"syringe" "10 ml" filetype:bitmap',
    'intitle:"syringe" "training" filetype:bitmap',
    'intitle:"vacuum" "blood" filetype:bitmap',
    'intitle:"catheter" "luer" filetype:bitmap',
    'intitle:"blood collection" "holder" filetype:bitmap',
)


def exposure():
    known = previous()
    for row in read_json(DENTAL)["rows"]:
        protect(known, row)
    return known


def discover(limit):
    if MANIFEST.exists():
        raise ValueError("The new audit is already sealed.")
    known = exposure()
    offers = []
    for query in QUERIES:
        params = dict(action="query", generator="search", gsrlimit="30", gsrnamespace="6", gsrlimitnt=0,
                      gsrsearch=query, prop="imageinfo", iiprop="url|size|mime|extmetadata|timestamp",
                      iiurlwidth="960", format="json")
        try:
            pages = json.loads(request(API + "?" + urlencode(params), 8 * 1024 * 1024))
        except (OSError, ValueError) as exc:
            print(query, exc, flush=True)
            continue
        entries = [row for p in pages.get("query", {}).get("pages", {}).values() if (row:=metadata(p,query,"mixed"))]
        offers.append(sorted(entries,key=lambda r:order(r["id"])))
    accepted = []
    for round_ in zip_longest(*offers):
        for proposal in round_:
            if proposal is None or withheld(proposal,known):
                continue
            candidate = get_asset(proposal)
            if candidate is None:
                continue
            row,raster = candidate
            if overlaps(known,row,raster):
                continue
            protect(known,row)
            accepted.append(row)
            if len(accepted)>=limit:
                break
        if len(accepted)>=limit:
            break
    write_json(POOL,dict(rows=accepted))
    sheet(accepted,"empty-scene-audit",4,3)
    print("Fresh recovery candidates",len(accepted),flush=True)


def freeze():
    if MANIFEST.exists():
        raise ValueError("The new audit is already sealed.")
    references = read_json(REFERENCES)
    if references["pool_sha256"]!=digest(POOL):
        raise ValueError("The new annotator references do not match the candidate pool.")
    known = exposure()
    rows = []
    for row in read_json(POOL)["rows"]:
        note = references["images"].get(row["id"])
        if note is None:
            continue
        path = ROOT/row["image"]
        if digest(path)!=row["sha256"] or overlaps(known,row,read_image(path)) or not all(
            valid_box(p["bbox"]) for p in note["objects"]
        ):
            raise ValueError("New audit reference or lineage check failed.")
        protect(known,row)
        rows.append({**row,**note,"split":"empty-scene-audit"})
    if (len(rows) != len(references["images"]) or sum(bool(r["objects"]) for r in rows) < 12 or
        sum(not r["objects"] for r in rows) < 4):
        raise ValueError("At least 12 target scenes and four lookalike scenes must be independently reviewed.")
    write_json(MANIFEST,dict(rows=rows,review_sha256=digest(REFERENCES)))
    print("Frozen independent acquisition groups",len(rows),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("discover","freeze"))
    parser.add_argument("--limit",type=int,default=50)
    args=parser.parse_args()
    discover(args.limit) if args.action=="discover" else freeze()
