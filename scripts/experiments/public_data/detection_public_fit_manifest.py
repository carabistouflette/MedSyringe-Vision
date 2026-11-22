"""Materialize the manually reviewed, training-only supplemental photographs."""
from collections import Counter
from pathlib import Path

from medsyringe.data import digest, leakage, read_image, read_json, valid_box, write_json
from scripts.experiments.detector_training.detection_hillclimb_corpus import ART
from scripts.experiments.public_data.detection_public_benchmark import MANIFEST, POOL, ROOT, WORK, lexical_key, near_any, prior_exposure, url_key, withheld

# Endpoints were transcribed from the coordinate-labelled local contact sheets.
PIXEL_BOXES = {
    "200299685": [[223,7,259,217]], "150790461": [[321,36,551,84],[336,137,548,198]],
    "4487086": [[675,123,819,160]], "72197330": [[59,334,280,392],[40,380,280,432]],
    "11195784": [[318,334,539,379]], "108248905": [[113,530,164,706]],
    "1783247": [[321,669,552,713]], "2779427": [[617,637,789,721]],
    "145244956": [[643,72,808,111]], "26379221": [[97,328,205,391]],
    "155856137": [[327,365,519,407]], "131793963": [[746,339,769,453]],
    "44352059": [[64,576,103,650]], "30051046": [[24,38,244,178]],
}
ZOOM_BOXES = {
    "191240619": [[.243,.237,.535,.792],[.375,.173,.707,.711]],
    "29035207": [[0.,.218,.771,.613]], "25524601": [[.789,.357,.902,.499]],
    "8203134": [[.478,.207,.500,.316]],
}
NEGATIVES = {
    "17328716": "loose needle", "10621296": "detached needles", "148368823": "pump display",
    "17283750": "epoxy cartridge", "31351996": "catheter and blood collection tubing",
    "3323591": "macro of loose needle", "993459": "thermal paste applicator", "115697835": "micropipette",
    "59284146": "transfer pipettes", "97257014": "Pasteur pipettes", "18864470": "adjustable pipette",
    "127790110": "pen needle packs", "17714440": "mechanical pencil tip", "116616738": "mechanical pencil set",
    "1576673": "pencil detail",
}
LOOKALIKE_BOXES = {
    "115697835": [.095,.161,.874,.849], "59284146": [.093,.211,.962,.979],
    "18864470": [.075,.350,.962,.635], "127790110": [.066,.258,.892,.770],
    "17714440": [.192,.002,1.,.530], "1576673": [.003,.004,.625,.636],
    "3323591": [.376,.003,.658,.791], "993459": [.054,.180,.753,.799],
    "17283750": [.027,.332,.968,.929],
}
OUT = ART / "public_additional_fit.json"


def materialize():
    if OUT.exists():
        raise ValueError("The additional crop fitting data have already been sealed.")
    catalog = WORK / "additional-fit-search-images.json"
    coordinates = WORK / "sheets" / "fit-coords-positive-coordinates.json"
    positions = read_json(coordinates)
    sources = {r["id"]:r for p in (catalog,POOL) for r in read_json(p)["rows"]}
    reviewed = ZOOM_BOXES.copy()
    for key, boxes in PIXEL_BOXES.items():
        _,x,y,w,h = positions[key]
        reviewed[key] = [[round((l-x)/w,6),round((t-y)/h,6),round((r-x)/w,6),round((b-y)/h,6)]
                         for l,t,r,b in boxes]
    prior = prior_exposure()
    protected = [*read_json(MANIFEST)["rows"],*read_json(ART/"dental_manifest.json")["rows"],
                 *[r for r in read_json(ART/"phlebotomy.json")["rows"] if r["split"]!="fit"]]
    for row in protected:
        prior["ids"].add(row["id"])
        prior["hashes"].add(row["sha256"])
        prior["difference"].add(int(row["dhash"],16))
        prior["authors"].update(row.get("profile_keys",[]))
        if row.get("author"):
            prior["authors"].add(lexical_key(row["author"]))
        prior["urls"].update(url_key(row[k]) for k in ("source_url","original_url","image_url") if row.get(k))
    result = []
    excluded = Counter()
    for key in (*reviewed,*NEGATIVES):
        row = sources[key]
        path = ROOT/row["image"]
        if digest(path)!=row["sha256"]:
            raise ValueError(f"The reviewed source bytes changed for {key}.")
        image = read_image(path)
        if withheld(row,prior) or row["sha256"] in prior["hashes"] or near_any(image,prior["difference"]):
            excluded["source, author or visual overlap"] += 1
            continue
        objects = [dict(bbox=b) for b in reviewed.get(key,[])]
        lookalikes = [dict(bbox=LOOKALIKE_BOXES[key],kind=NEGATIVES[key])] if key in LOOKALIKE_BOXES else []
        if any(not valid_box(obj["bbox"]) for obj in (*objects,*lookalikes)):
            raise ValueError(f"The human coordinates are invalid for {key}.")
        result.append({**row,"split":"fit","objects":objects,"lookalikes":lookalikes,
                       "scene_note":NEGATIVES.get(key,"Manual whole-visible-syringe box")})
        prior["authors"].update({lexical_key(row["author"]),*row.get("profile_keys",[])})
        prior["difference"].add(int(row["dhash"],16))
        prior["hashes"].add(row["sha256"])
    if leakage([*result,*[{**r,"split":"protected"} for r in protected]]) or len({r["group"] for r in result})!=len(result):
        raise ValueError("Supplemental fit contains a protected source or repeated photographer.")
    counts = dict(images=len(result),positive=sum(bool(r["objects"]) for r in result),
                  objects=sum(len(r["objects"]) for r in result),negative=sum(not r["objects"] for r in result))
    write_json(OUT,dict(rows=result,counts=counts,excluded=dict(excluded),code_sha256=digest(__file__),
                        inputs={str(p):digest(p) for p in (catalog,coordinates,MANIFEST,POOL)},
                        scope="Single-reviewer labels from photos with no model overlays. Training use only. "
                              "No previously used or protected authors, known source files or transformed close hashes."))
    print(counts, dict(excluded), flush=True)


if __name__ == "__main__":
    materialize()
