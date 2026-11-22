"""Stage new photographic lookalikes before the extended detector is selected."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from medsyringe.data import digest, leakage, read_image, read_json, write_json
from scripts.experiments.detector_training.detection_hillclimb_corpus import ART
from scripts.experiments.public_data.detection_public_benchmark import (API, ROOT, WORK, get_asset, lexical_key, metadata, near_any,
                                                order, prior_exposure, request, sheet, url_key, withheld)

QUERIES = (
    'intitle:pipette filetype:bitmap', 'intitle:vacutainer filetype:bitmap',
    'intitle:"insulin pen" filetype:bitmap', 'intitle:"catheter" filetype:bitmap',
    'intitle:"test tube" filetype:bitmap', 'intitle:"syringe pump" filetype:bitmap',
    'intitle:"mechanical pencil" filetype:bitmap', 'intitle:"thermal paste" filetype:bitmap',
)
CANDIDATES = WORK / "negative-audit-candidates.json"
MANIFEST = ART / "external_lookalike_manifest.json"
REVIEW = ART / "external_lookalike_reference_review.json"


def exposure():
    known = prior_exposure()
    for p in ("public_manifest.json","public_additional_fit.json","dental_manifest.json",
              "phlebotomy.json","openimages_additional_manifest.json"):
        for r in read_json(ART/p)["rows"]:
            known["ids"].add(r["id"])
            known["hashes"].add(r["sha256"])
            known["difference"].add(int(r["dhash"],16))
            known["authors"].update(r.get("profile_keys",[]))
            if r.get("author"):
                known["authors"].add(lexical_key(r["author"]))
            known["urls"].update(url_key(r[k]) for k in ("source_url","original_url","image_url") if r.get(k))
    return known


def discover(limit):
    if MANIFEST.exists():
        raise ValueError("Keep the existing independent lookalike manifest frozen.")
    seen = exposure()
    offers = []
    for query in QUERIES:
        parameters = dict(action="query",generator="search",gsrlimit="26",gsrnamespace="6",gsrsearch=query,
                          prop="imageinfo",iiprop="url|size|mime|extmetadata|timestamp",iiurlwidth="960",format="json")
        try:
            pages = json.loads(request(API+"?"+urlencode(parameters),8*1024*1024)).get("query",{}).get("pages",{})
        except (OSError,ValueError) as exc:
            print(query,"unavailable",exc,flush=True)
            continue
        rows = [row for page in pages.values() if (row:=metadata(page,query,"lookalike"))]
        offers.append(iter(sorted(rows,key=lambda r:order(r["id"]))))
    chosen = []
    while len(chosen)<limit:
        progressed = False
        for candidates in offers:
            item = next(candidates,None)
            if item is None:
                continue
            progressed = True
            if withheld(item,seen):
                continue
            seen["ids"].add(item["id"])
            asset = get_asset(item)
            if not asset:
                continue
            row,raster = asset
            if row["sha256"] in seen["hashes"] or near_any(raster,seen["difference"]):
                continue
            seen["hashes"].add(row["sha256"])
            seen["difference"].add(int(row["dhash"],16))
            seen["authors"].update({lexical_key(row["author"]),*row.get("profile_keys",[])})
            chosen.append(row)
            if len(chosen)>=limit:
                break
        if not progressed:
            break
    write_json(CANDIDATES,dict(rows=chosen))
    sheet(chosen,"reserved-lookalikes-next",4,3)
    print("Additional lookalike candidates",len(chosen),flush=True)


def freeze():
    if MANIFEST.exists():
        raise ValueError("Keep the existing independent lookalike manifest frozen.")
    checked = read_json(REVIEW)
    if checked["candidates_sha256"]!=digest(CANDIDATES):
        raise ValueError("The candidate and manual review records differ.")
    known = exposure()
    rows = []
    for r in read_json(CANDIDATES)["rows"]:
        kind = checked["confirmed"].get(r["id"])
        if not kind:
            continue
        file = ROOT/r["image"]
        if (withheld(r,known) or r["sha256"] in known["hashes"] or digest(file)!=r["sha256"] or
            near_any(read_image(file),known["difference"])):
            raise ValueError(f"Protected source overlap on {r['id']}.")
        rows.append({**r,"objects":[],"lookalike_type":kind,"split":"negative-audit"})
    if len(rows)!=len(checked["confirmed"]) or len({r["group"] for r in rows})!=len(rows) or leakage(rows):
        raise ValueError("Confirmed negative scenes must have separate photographer groups.")
    write_json(MANIFEST,dict(rows=rows,visual_review_sha256=digest(REVIEW),count=len(rows),
                             scope="Supplemental negative-only audit. Use for scene false alarms and specificity."))
    print("Frozen negative-only audit",len(rows),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("discover","freeze"))
    parser.add_argument("--limit",type=int,default=26)
    args=parser.parse_args()
    discover(args.limit) if args.action=="discover" else freeze()
