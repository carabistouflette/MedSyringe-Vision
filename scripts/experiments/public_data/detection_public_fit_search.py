"""Discover additional training-only syringe photographs with file provenance."""
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from medsyringe.data import read_json, write_json
from scripts.experiments.public_data.detection_public_benchmark import (API, MANIFEST, POOL, WORK, get_asset, lexical_key, metadata,
                                                near_any, order, prior_exposure, request, sheet, url_key,
                                                withheld)

QUERIES = (
    'intitle:syringe disposable filetype:bitmap',
    'intitle:syringe insulin filetype:bitmap',
    'intitle:syringe graduated filetype:bitmap',
    'intitle:syringe medicine filetype:bitmap',
    'intitle:syringe glass filetype:bitmap',
    'intitle:syringe injection filetype:bitmap',
    'intitle:syringe filetype:bitmap',
    'intitle:syringes filetype:bitmap',
)
CATALOG = WORK / "additional-fit-search.json"
PICKED = WORK / "additional-fit-search-images.json"


def search():
    result = read_json(CATALOG) if CATALOG.exists() else {}
    for query in QUERIES:
        if query in result:
            continue
        parameters = dict(action="query", generator="search", gsrlimit="38", gsrnamespace="6", gsrsearch=query,
                          prop="imageinfo", iiprop="url|size|mime|extmetadata|timestamp", iiurlwidth="960", format="json")
        try:
            body = json.loads(request(API+"?"+urlencode(parameters),8*1024*1024))
        except (ValueError, OSError, HTTPError, URLError) as exc:
            print(query,"unavailable",exc,flush=True)
            continue
        pages = body.get("query",{}).get("pages",{})
        result[query] = [row for item in pages.values() if (row:=metadata(item,query,"syringe"))]
        write_json(CATALOG,result)
        print(query,len(result[query]),flush=True)
    return result


def collect(limit):
    catalog = search()
    exposure = prior_exposure()
    prior = [*read_json(MANIFEST)["rows"],*read_json(POOL)["rows"]]
    for row in prior:
        exposure["ids"].add(row["id"])
        exposure["authors"].update({lexical_key(row["author"]),*row.get("profile_keys",[])})
        exposure["titles"].add(lexical_key(row["title"]))
        exposure["urls"].update(url_key(row[k]) for k in ("source_url","original_url"))
        exposure["difference"].add(int(row["dhash"],16))
        exposure["hashes"].add(row["sha256"])
    stash = read_json(PICKED) if PICKED.exists() else {"rows":[],"screened":[]}
    selected = stash["rows"]
    visited = set(stash["screened"])
    for row in selected:
        exposure["authors"].update({lexical_key(row["author"]),*row.get("profile_keys",[])})
        exposure["difference"].add(int(row["dhash"],16))
        exposure["hashes"].add(row["sha256"])
    counts = Counter(r["category"] for r in selected)
    iterators = [iter(sorted(catalog.get(query,[]),key=lambda r:order(r["id"]))) for query in QUERIES]
    while len(selected) < limit:
        progress = False
        for candidates in iterators:
            if len(selected) >= limit:
                break
            current = next(candidates,None)
            if not current:
                continue
            progress = True
            key = current["id"]
            if key in visited or counts[current["category"]]>=12:
                continue
            visited.add(key)
            asset = None if withheld(current,exposure) else get_asset(current)
            if asset:
                row,image = asset
                if row["sha256"] not in exposure["hashes"] and not near_any(image,exposure["difference"]):
                    selected.append(row)
                    exposure["authors"].update({lexical_key(row["author"]),*row.get("profile_keys",[])})
                    exposure["difference"].add(int(row["dhash"],16))
                    exposure["hashes"].add(row["sha256"])
                    counts[row["category"]]+=1
                    print(len(selected),row["title"],flush=True)
            if len(visited)%8==0:
                write_json(PICKED,dict(rows=selected,screened=sorted(visited)))
        if not progress:
            break
    write_json(PICKED,dict(rows=selected,screened=sorted(visited)))
    sheet(selected,"additional-fit-search",4,3)


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--limit",type=int,default=36)
    collect(arg.parse_args().limit)
