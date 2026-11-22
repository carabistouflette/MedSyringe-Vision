"""Freeze photographer-disjoint syringe and lookalike images before inference."""
from collections import Counter
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, records, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision, protection
from scripts.experiments.crowds.native_candidate_bank import cohorts
from scripts.experiments.occlusion.occlusion_detector import ART, path_for, raster
from scripts.experiments.public_data.public_training_intake import raster_hashes

WORK = Path("data/syringe-object-verifier")
REFERENCE = ART / "syringe_object_verifier_references.json"
PD = "https://www.dvidshub.net/about/copyright"
CC = "https://creativecommons.org/licenses/by-sa/4.0/"
PEXELS = "https://www.pexels.com/license/"
COMMONS = "https://commons.wikimedia.org/wiki/File:"
WIKI = "https://upload.wikimedia.org/wikipedia/commons/"
DVIDS = "https://www.dvidshub.net/image/"
SOURCES = (
    ("fit", "9090709", DVIDS + "9090709/mdg-cares-offutt-patients", "Daniel Martinez", PD,
     (780, 557), (), (("pipette", (163, 244, 382, 379)),)),
    ("fit", "9242074", DVIDS + "9242074/nreip-orise-interns-present-research-namru-san-antonio", "Burrell Parmer", PD,
     (780, 557), (), (("pipette", (351, 200, 468, 395)),)),
    ("fit", "tube", COMMONS + "Vacutainer_01.jpg", "Every1blowz", "Public domain per Commons",
     (640, 480), (), (("blood tube", (378, 26, 464, 465)),)),
    ("fit", "9500907", DVIDS + "9500907/forensic-toxicology-reinforces-joint-standards", "Noah Coger", PD,
     (780, 488), (), (("pipette", (361, 216, 385, 336)), ("pipette", (654, 33, 734, 315)))),
    ("fresh", "35609687", "https://www.pexels.com/photo/medical-professional-preparing-syringe-for-injection-35609687/",
     "DΛVΞ GΛRCIΛ", PEXELS, (453, 680), ((142, 8, 247, 355),), ()),
    ("fresh", "27925592", "https://www.pexels.com/photo/a-person-in-gloves-holding-a-syringe-in-their-hand-27925592/",
     "Hannah Barata", PEXELS, (780, 520), ((268, 84, 383, 244),), ()),
    ("fresh", "5355836", "https://www.pexels.com/photo/close-up-photo-of-a-syringe-held-by-a-medical-professional-5355836/",
     "Tima Miroshnichenko", PEXELS, (453, 680), ((142, 96, 297, 602),), ()),
    ("fresh", "10052835", DVIDS + "10052835/two-paths-one-mission-bjach-graduates-army-laboratory-specialists",
     "Jean Graves", PD, (510, 680), (), (("pipette", (327, 69, 364, 165)),)),
    ("fresh", "vacutainer", COMMONS + "Vacutainer_with_blood_collection_tube.jpg",
     "Whispyhistory", "https://creativecommons.org/licenses/by/3.0/us/",
     (510, 680), (), (("vacuum blood tube and holder", (101, 147, 416, 312)), ("loose needle assembly", (241, 322, 351, 470)))),
    ("fresh", "micro", COMMONS + "Micropipettes_on_a_pipette_stand_01.jpg", "Gannu03", CC, (780, 520),
     (), (("pipette", (12, 109, 159, 317)), ("pipette", (169, 119, 274, 466)),
          ("pipette", (281, 139, 352, 455)), ("pipette", (337, 148, 419, 496)),
          ("pipette", (476, 140, 558, 515)), ("pipette", (538, 116, 598, 415)))),
    ("fresh", "10067869", DVIDS + "10067869/uss-john-c-stennis-sailors-cvn-74-receive-flu-shots",
     "Scarlett Goodlander", PD, (780, 520), ((444, 291, 489, 326),), ()),
    ("fresh", "10015275", DVIDS + "10015275/ensuring-lethality-dha-bumed-staff-receive-annual-influenza-vaccinations",
     "Sasha Ambrose", PD, (486, 680), ((337, 305, 356, 337),), ()),
    ("fresh", "9918854", DVIDS + "9918854/95th-medical-detachment-blood-support-conducts-deglycerolization-blood-stress-test",
     "Martin Nalls", PD, (780, 521), (), (("pipette", (208, 60, 237, 263)),)),
    ("fresh", "9364294", DVIDS + "9364294/walters-way-behind-scenes-lab-professionals-walter-reed-national-military-medical-center",
     "Ricardo Reyes-Guevara", PD, (780, 537), (), (("pipette", (243, 255, 333, 388)),)),
    ("fresh", "9922212", DVIDS + "9922212/biogeochemical-sample-analysis-fort-collins-laboratory",
     "Forest Service OC", PD, (780, 565), (), (("pipette", (337, 244, 383, 366)),)),
    ("fresh", "7000883", DVIDS + "7000883/enlisted-us-air-force-medics-plays-vital-role-michigan-hospital-covid-fight",
     "Timothy Hughes", PD, (780, 439), (), ()),
    ("fresh", "8233858", DVIDS + "8233858/day-life-laboratory-technician",
     "Alysa Calvarese", PD, (780, 520), (), (("specimen tube", (201, 185, 218, 252)),)),
    ("fresh", "8626526", DVIDS + "8626526/nmrc-combat-wounds-lab-prepares-skin-cells-early-stage-testing",
     "Michael Wilson", PD, (780, 520), (), (("pipette", (532, 226, 682, 304)),)),
    ("fresh", "8211799", DVIDS + "8211799/nmrcs-naval-infectious-disease-diagnostic-laboratory-conducts-lyme-disease-testing",
     "Alejandra Ramirez Alarcon", PD, (453, 680), (), (("pipette", (0, 516, 171, 578)),)),
)
DVIDS_PATHS = {
    "9090709": "2506", "9242074": "2508", "10052835": "2610", "9500907": "2601",
    "10067869": "2611", "10015275": "2610", "9918854": "2609", "9364294": "2510", "9922212": "2609",
    "7000883": "2201", "8233858": "2402", "8626526": "2409", "8211799": "2401",
}
WIKI_ASSETS = {
    "tube": "9/95/Vacutainer_01.jpg",
    "vacutainer": "5/58/Vacutainer_with_blood_collection_tube.jpg",
    "micro": "5/59/Micropipettes_on_a_pipette_stand_01.jpg",
}
EXCLUSIONS = {
    "8711851": "Several cropped or softly focused background syringes are hard to assign precise extents.",
    "7230835": "Previously visually screened. Additional source coverage was available.",
    "10098263": "Wide event image with no identifiable syringe or useful lookalike.",
    "10069741": "Fingers obscure the syringe barrel and plunger extents.",
    "10021482": "The syringe is largely hidden between fingers and the patient.",
    "10048453": "The injector is obscured by the operator's hand.",
}


def asset(key, license):
    if license == PEXELS:
        return f"https://images.pexels.com/photos/{key}/pexels-photo-{key}.jpeg?auto=compress&cs=tinysrgb&w=1000"
    if key in DVIDS_PATHS:
        return f"https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/{DVIDS_PATHS[key]}/{key}/1000w_q95.jpg"
    return WIKI + WIKI_ASSETS[key]


def historical():
    h, private, private_pixels = protection()
    past = [r for role in ("fit", "development", "evaluation") for r in cohorts(role)]
    for p in sorted(ART.glob("*reserve.json")):
        past.extend(records(read_json(p)))
    pixels = h["dhash"] | private_pixels
    for path in sorted({path_for(r) for r in past if r.get("image")}):
        if path.is_file():
            pixels.update(raster_hashes(path))
    return dict(
        pixels=pixels, hashes=h["sha256"] | private | {r["sha256"] for r in past if r.get("sha256")},
        authors=h["authors"] | {lexical_key(r["author"]) for r in past if r.get("author")},
        urls=h["urls"] | {url_key(r["source_url"]).rstrip("/") for r in past if r.get("source_url")},
    )


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    known = historical()
    roles, rejected = {"fit": [], "fresh": []}, dict(EXCLUSIONS)
    for role, key, page, author, lic, preview, objects, lookalikes in SOURCES:
        file = WORK / f"{key}.jpg"
        who, url, content = lexical_key(author), url_key(page).rstrip("/"), digest(file)
        pixel = raster_hashes(file)
        if (url in known["urls"] or content in known["hashes"] or
                any(who in name for name in known["authors"]) or
                fingerprint_collision(pixel, known["pixels"])):
            rejected[key] = "Historical source, contributor, byte hash or visual match."
            continue

        def normalized(b):
            coords = [v/n for v,n in zip(b,preview*2,strict=True)]
            if not valid_box(coords):
                raise ValueError(f"Invalid annotation for {key}.")
            return coords

        row = dict(id=f"object-{key}", image=str(file), sha256=content,
                   source_url=page, original_url=asset(key,lic), author=author, license=lic,
                   group=f"object:{who}", objects=[dict(bbox=normalized(b)) for b in objects],
                   lookalikes=[dict(kind=k,bbox=normalized(b)) for k,b in lookalikes])
        im = raster(row)
        if abs(im.width/im.height - preview[0]/preview[1]) > .004:
            raise ValueError(f"Wrong review raster for {key}.")
        roles[role].append(row)
        known["hashes"].add(content)
        known["pixels"].update(pixel)
        known["urls"].add(url)
        known["authors"].add(who)
    fresh = Counter(bool(r["objects"]) for r in roles["fresh"])
    if min(fresh[True], fresh[False]) < 2:
        raise ValueError(f"Need independent positive and lookalike scenes after screening: {rejected}")
    write_json(REFERENCE, dict(code=digest(__file__), roles=roles, excluded=rejected,
                               notes="One visual reviewer. True object and lookalike labels preceded inference."))
    print({k:[r["id"] for r in v] for k,v in roles.items()}, rejected, flush=True)


def references(role):
    state = read_json(REFERENCE)
    if state["code"] != digest(__file__):
        raise ValueError("Photo annotation protocol changed.")
    return state["roles"][role]


if __name__ == "__main__":
    freeze()
