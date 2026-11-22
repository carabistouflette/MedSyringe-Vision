"""Score frozen native-crop policies on independently annotated public acquisitions."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.grounding_backend import analyze
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from scripts.detection_public_benchmark import lexical_key, url_key
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.native_candidate_bank import WORK, archive, cohorts, collect
from scripts.native_candidate_verifier import POLICY, RESULT as HISTORY, run_evaluation, sealed
from scripts.occlusion_detector import ART, path_for, raster
from scripts.public_training_intake import raster_hashes
from scripts.sam3_study import configure_cpu

FOLDER = WORK / "prospective"
REFERENCE = ART / "native_candidate_verifier_reserve.json"
RESULT = ART / "native_candidate_verifier_prospective.json"

# One row per photographed acquisition. Pixel annotation coordinates were reviewed
# on the source raster before either detector was run on these photographs.
DVIDS = (
    ("6892731", "2110", "Matthew Damon", "Blue Grass Army Depot 2021-10-16", "operation-fortified-guardian",
     (1000, 667), [(62, 351, 887, 526), (87, 376, 995, 559)],
     "Two overlapping defocused syringes with separate plungers and protective covers. The vial is separate."),
    ("9190822", "2507", "Alexander Casco", "USS Gerald R. Ford 2025-07-14",
     "uss-gerald-r-ford-cvn-78-conducts-deployment-shot-exercise", (1000, 667), [(532, 248, 615, 509)],
     "One held syringe with attached visible needle and a separate vial."),
    ("10052835", "2610", "Jean Graves", "Fort Polk 2026-10-27",
     "two-paths-one-mission-bjach-graduates-army-laboratory-specialists", (1000, 1333), [],
     "Laboratory pipette, small vessel, and testing equipment. No syringe visible."),
    ("9918854", "2609", "Martin Nalls", "Camp Humphreys 2026-08-25",
     "95th-medical-detachment-blood-support-conducts-deglycerolization-blood-stress-test",
     (1000, 668), [], "A transfer pipette above a test fixture. No syringe visible."),
    ("9468367", "2601", "Devyn Waits", "Keesler 2025-10-29",
     "medical-genetics-center-empowering-patients-one-test-time", (1000, 667), [],
     "Micropipettes, tubes, and racks. No syringe assembly visible."),
)
COMMONS = (
    ("chemyx", "Chemyx_SKE10.jpg", "e/e7", "SabrielxD", "Chemyx SKE10 2022-03-30",
     "https://creativecommons.org/licenses/by-sa/4.0/", (1174, 880), (1174, 880),
     [(305, 364, 558, 389), (305, 386, 558, 412)],
     "Two thin glass syringe assemblies in a pump. External ducts and the pump chassis are excluded."),
    ("capa", "Capa_Valve_Multi-chamber_syringe.png", "c/c2", "Philcapawiki",
     "Multichamber syringe bench 2023-12-23", "https://creativecommons.org/licenses/by/4.0/",
     (3024, 3024), (680, 680), [(110, 78, 588, 630)],
     "One unmarked multichamber syringe. A detached fitting and a glass container are separate."),
)


def source_records():
    for key, month, author, group, slug, size, boxes, note in DVIDS:
        page = f"https://www.dvidshub.net/image/{key}/{slug}"
        preview = f"https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/{month}/{key}/1000w_q95.jpg"
        yield key, FOLDER/f"{key}.jpg", author, group, page, preview, "Public domain (DVIDS)", size, size, boxes, note
    for key, filename, part, author, group, license, size, units, boxes, note in COMMONS:
        path = FOLDER / f"{key}{Path(filename).suffix}"
        yield (key, path, author, group, f"https://commons.wikimedia.org/wiki/File:{filename}",
               f"https://upload.wikimedia.org/wikipedia/commons/{part}/{filename}",
               license, size, units, boxes, note)


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    history, private_hashes, private_visual = protection()
    prior = [r for role in ("fit", "development", "evaluation") for r in cohorts(role)]
    protected = private_hashes | history["sha256"] | {r["sha256"] for r in prior}
    visual = private_visual | history["dhash"]
    for row in prior:
        visual.update(raster_hashes(path_for(row)))
    authors = history["authors"] | {lexical_key(r["author"]) for r in prior if r.get("author")}
    urls = history["urls"] | {url_key(r["source_url"]).rstrip("/") for r in prior if r.get("source_url")}
    rows = []
    for key, path, author, group, page, original, license, size, units, boxes, note in source_records():
        sha256 = digest(path)
        pixels = raster_hashes(path)
        author_key = lexical_key(author)
        source_key = url_key(page).rstrip("/")
        if (key in history["ids"] or source_key in urls or
                any(author_key and author_key in known for known in authors) or
                sha256 in protected or fingerprint_collision(pixels, visual)):
            raise ValueError(f"Known source, contributor, or visual overlap for {key}")
        w, h = units
        objects = [dict(bbox=[x/w, y/h, R/w, B/h]) for x, y, R, B in boxes]
        row = dict(id=f"crop-reserve-{key}", image=str(path), sha256=sha256, author=author,
                   group=group, source_url=page, original_url=original, license=license, objects=objects, review=note)
        if raster(row).size != size or any(not valid_box(p["bbox"]) for p in objects):
            raise ValueError(f"Raster or box verification failed for {key}")
        rows.append(row)
        protected.add(sha256)
        visual.update(pixels)
        authors.add(author_key)
        urls.add(source_key)
    write_json(REFERENCE, dict(code_sha256=digest(__file__), rows=rows,
                              release_condition="Require the dev-selected policy, new prospective matches, no "
                                                "per-image false-positive increase or loss of existing matches "
                                                "or count quality in prospective and archived cohorts.",
                              screening="Source, contributor, exact hashes, and rotated/mirrored difference hashes. "
                                        "No Fil Rouge raster or comparison crop used."))
    print(len(rows), "new acquisitions with", sum(len(r["objects"]) for r in rows), "reviewed syringes", flush=True)


def references():
    record = read_json(REFERENCE)
    if record["code_sha256"] != digest(__file__):
        raise ValueError("The prospective study changed after freezing the references.")
    return record["rows"]


def baseline():
    binding = dict(reference=digest(REFERENCE), policy=digest(POLICY))
    sealed()
    configure_cpu()
    model = MedSyringeEngine(Path.cwd())
    configure_device(model, Path.cwd(), "xpu")
    target = FOLDER / "released.json"
    state = read_json(target) if target.exists() else dict(binding=binding, outputs={}, seconds={})
    if state["binding"] != binding:
        raise ValueError("The released output cache belongs to a different reserve or policy.")
    try:
        for row in references():
            key = row["id"]
            if key in state["outputs"]:
                continue
            started = perf_counter()
            result = analyze(model, raster(row), False)
            checks = result["detection_checks"]
            if checks["status"] != "available" or checks["primary_backend"] != "xpu":
                raise ValueError(f"Released detector unavailable for {key}")
            state["outputs"][key] = result["objects"]
            state["seconds"][key] = perf_counter()-started
            write_json(target, state)
            print(key, "done", flush=True)
    finally:
        model.close()


def proposals():
    sealed()
    collect(references(), "prospective")


def measure():
    state = sealed()
    rows = references()
    release = read_json(FOLDER / "released.json")
    if release["binding"] != dict(reference=digest(REFERENCE), policy=digest(POLICY)):
        raise ValueError("The prospective release output cache changed.")
    historical = read_json(HISTORY)
    if historical["policy"] != digest(POLICY):
        raise ValueError("Retrospective score belongs to an earlier policy.")
    before = release["outputs"]
    outcomes = run_evaluation({"prospective": rows}, before, archive(rows, "prospective"))["prospective"]
    selected = state["decision"]["selected"]
    key = (f'{selected["rule"]["appearance"]}-'
           f'{"box" if selected["rule"]["distinct"] is None else "identity"}') if selected else None
    gate = False
    if key:
        measured = outcomes["arms"][key]["baseline_delta"]
        gate = (measured["safe"] and measured["gain"] > 0 and
                all(item["arms"][key]["baseline_delta"]["safe"] for item in historical["evidence"].values()))
    write_json(RESULT, dict(reference=digest(REFERENCE), policy=digest(POLICY), retrospective=digest(HISTORY),
                            release_sha256=digest(FOLDER / "released.json"),
                            rf_sha256=digest(WORK / "rf-prospective.json"),
                            selected_arm=key, release_gate=gate, prospective=outcomes))
    print(key, gate, {name: (v["baseline_delta"]["gain"], v["baseline_delta"]["extra_fp"])
                      for name, v in outcomes["arms"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "baseline", "proposals", "measure"))
    {"freeze": freeze, "baseline": baseline, "proposals": proposals, "measure": measure}[parser.parse_args().stage]()
