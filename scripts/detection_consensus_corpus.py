"""Collect another author-disjoint audit before inferring the consensus policy."""
import argparse
from collections import Counter
import json
from urllib.parse import urlencode

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from scripts.detection_hillclimb_corpus import ART
from scripts.detection_lookalike_audit_data import MANIFEST as NEGATIVES, exposure
from scripts.detection_public_benchmark import (API, ROOT, WORK, get_asset, lexical_key, metadata, near_any, order,
                                                request, sheet, url_key, withheld)

CANDIDATES = WORK/"consensus-audit-candidates.json"
REVIEW = ART/"consensus_audit_reference_review.json"
MANIFEST = ART/"consensus_audit_manifest.json"
QUERIES = (
    'intitle:injecting vaccine filetype:bitmap', 'intitle:vaccination syringe health filetype:bitmap',
    'intitle:syringe hypodermic filetype:bitmap', 'intitle:syringe "luer" filetype:bitmap',
    'intitle:syringe "needle" filetype:bitmap', 'intitle:phlebotomy syringe filetype:bitmap',
    'intitle:syringes vaccine filetype:bitmap', 'intitle:"prepared syringe" filetype:bitmap',
    'intitle:"administers" vaccination filetype:bitmap', 'intitle:aspirating syringe filetype:bitmap',
    'intitle:"dental syringe" filetype:bitmap', 'intitle:insulin injection filetype:bitmap',
)


def prior():
    seen = exposure()
    for row in read_json(NEGATIVES)["rows"]:
        seen["ids"].add(row["id"])
        seen["hashes"].add(row["sha256"])
        seen["difference"].add(int(row["dhash"],16))
        seen["authors"].update({lexical_key(row["author"]),*row.get("profile_keys",[])})
        seen["urls"].update(url_key(row[k]) for k in ("source_url","original_url","image_url"))
    return seen


def find(limit):
    if MANIFEST.exists():
        raise ValueError("The consensus reference set has been frozen.")
    known = prior()
    iterators = []
    for query in QUERIES:
        args = dict(action="query",generator="search",gsrlimit="44",gsrnamespace="6",gsrsearch=query,
                    prop="imageinfo",iiprop="url|size|mime|extmetadata|timestamp",iiurlwidth="960",format="json")
        try:
            pages = json.loads(request(API+"?"+urlencode(args),10*1024*1024)).get("query",{}).get("pages",{})
        except (OSError,ValueError) as exc:
            print("Unavailable",query,exc,flush=True)
            continue
        relevant = [x for p in pages.values() if (x:=metadata(p,query,"syringe"))]
        iterators.append(iter(sorted(relevant,key=lambda row:order(row["id"]))))
    checked = []
    while len(checked)<limit:
        progressed = False
        for items in iterators:
            p = next(items,None)
            if p is None:
                continue
            progressed = True
            if withheld(p,known):
                continue
            known["ids"].add(p["id"])
            picked = get_asset(p)
            if not picked:
                continue
            r,image = picked
            if r["sha256"] in known["hashes"] or near_any(image,known["difference"]):
                continue
            checked.append(r)
            known["hashes"].add(r["sha256"])
            known["difference"].add(int(r["dhash"],16))
            known["authors"].update({lexical_key(r["author"]),*r.get("profile_keys",[])})
            if len(checked)>=limit:
                break
        if not progressed:
            break
    write_json(CANDIDATES,dict(rows=checked))
    sheet(checked,"consensus-public-candidates",4,3)
    print("Sourced",len(checked),"new candidate photographs",flush=True)


def freeze():
    if MANIFEST.exists():
        raise ValueError("The consensus reference set has been frozen.")
    checked = read_json(REVIEW)
    picked = read_json(CANDIDATES)["rows"]
    if checked["candidates_sha256"]!=digest(CANDIDATES):
        raise ValueError("The manual review does not match the staged source set.")
    known = prior()
    included = []
    for r in picked:
        value = checked["photos"].get(r["id"])
        if value is None:
            continue
        item = ROOT/r["image"]
        if (digest(item)!=r["sha256"] or r["sha256"] in known["hashes"] or
            withheld(r,known) or near_any(read_image(item),known["difference"])):
            raise ValueError("An audit image overlaps protected sources or differs from its review.")
        if any(not valid_box(o["bbox"]) for o in (*value["objects"],*value["lookalikes"])):
            raise ValueError("An audit reference box is invalid.")
        included.append({**r,**value,"split":"consensus-audit"})
    if len({r["group"] for r in included})!=len(included) or len(included)!=len(checked["photos"]):
        raise ValueError("Duplicated photographer or incomplete source metadata.")
    negatives = read_json(NEGATIVES)["rows"]
    rows = [*included,*negatives]
    tally = Counter("positive" if r["objects"] else "negative" for r in rows)
    write_json(MANIFEST,dict(rows=rows,counts=dict(tally),source="Wikimedia Commons",
                             reference_sha256=digest(REVIEW),external_lookalikes_sha256=digest(NEGATIVES),
                             scope="Fresh photographer-disjoint consensus-audit photos plus a previously scored "
                                   "negative-audit cohort used for regression. Report the two roles separately."))
    print(dict(tally),sum(len(r["objects"]) for r in rows),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("find","freeze"))
    p.add_argument("--limit",type=int,default=52)
    a = p.parse_args()
    find(a.limit) if a.action=="find" else freeze()
