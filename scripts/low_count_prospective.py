"""Run a small independent low-count recovery pilot with frozen references."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from medsyringe.grounding_backend import analyze
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.recall_guard import compare
from scripts.detection_public_benchmark import lexical_key, url_key
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.low_count_recovery import DATASET, FREEZE as VERIFIER, CrowdedMasks, brief, configure_cpu, crowd, load_policy
from scripts.low_count_review import propose
from scripts.public_training_intake import raster_hashes

REFERENCE = crowd.ART / "low_count_prospective_reference.json"
OUTPUT = crowd.ART / "low_count_prospective_results.json"
WORK = Path("data/low-count-prospective")
CATALOG = Path("data/public-web-collection/manifest.json")
POSITIVE = WORK / "Influenza_vaccine_syringes.jpg"
SOURCE = "https://commons.wikimedia.org/wiki/File:Influenza_vaccine_syringes.jpg"
NEGATIVES = ("openverse-ed1b263b-9b98-4ddf-8c5e-6d697a688b48", "openverse-74f4ecd6-6d3a-4f4f-8df4-fd9abd0a2fbc")
PIXELS = [
    (154, 74, 767, 220), (135, 247, 772, 387), (115, 446, 785, 571), (103, 637, 793, 775),
    (63, 847, 787, 1005), (641, 66, 1189, 185), (623, 202, 1180, 337), (646, 373, 1234, 484),
    (631, 546, 1239, 674), (623, 739, 1235, 859),
]


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    if read_image(POSITIVE).size != (1280, 1043):
        raise ValueError("Unexpected photo dimensions.")
    gallery = {r["id"]: r for r in read_json(CATALOG)["rows"]}
    rows = [dict(id="epolk-influenza-tray", source_url=SOURCE, author="Epolk",
                 group="Epolk influenza tray 2024-10-16", license="https://creativecommons.org/licenses/by-sa/4.0/",
                 sha256=digest(POSITIVE), image=str(POSITIVE),
                 objects=[dict(bbox=[a/1280, b/1043, c/1280, d/1043]) for a, b, c, d in PIXELS])]
    for key in NEGATIVES:
        r = gallery[key]
        rows.append({**{name: r[name] for name in ("id", "author", "group", "source_url", "sha256", "license")},
                     "image": str(Path("data") / r["image"]), "objects": []})
    history, private, hashes = protection()
    current = read_json(DATASET)["rows"]
    old_groups = {lexical_key(r["author"]) for r in current}
    used_sha = private | history["sha256"] | {r["sha256"] for r in current}
    used_dhash = hashes | history["dhash"]
    for row in rows:
        if (row["sha256"] in used_sha or digest(row["image"]) != row["sha256"]
                or url_key(row["source_url"]).rstrip("/") in history["urls"]
                or lexical_key(row["author"]) in history["authors"] | old_groups
                or fingerprint_collision(raster_hashes(row["image"]), used_dhash)
                or not all(valid_box(obj["bbox"]) for obj in row["objects"])):
            raise ValueError(f"Protected exposure or invalid local reference: {row['id']}")
    write_json(REFERENCE, dict(verifier=digest(VERIFIER), association=digest(Path("scripts/low_count_review.py")),
                               rows=rows, scope="One crowded acquisition and two pipette hard negatives. "
                                               "Counts and boxes reviewed from pixels before new inference."))


def inputs():
    state = read_json(REFERENCE)
    if state["verifier"] != digest(VERIFIER) or state["association"] != digest(Path("scripts/low_count_review.py")):
        raise ValueError("Prospective policy changed.")
    if any(digest(r["image"]) != r["sha256"] for r in state["rows"]):
        raise ValueError("An evaluation photo changed.")
    return state["rows"]


def released():
    configure_cpu()
    root = Path.cwd()
    engine = MedSyringeEngine(root)
    configure_device(engine, root, "xpu")
    outputs = {}
    try:
        for row in inputs():
            result = analyze(engine, read_image(row["image"]), False)
            checks = result["detection_checks"]
            if checks["primary_backend"] != "xpu" or checks["status"] != "available":
                raise ValueError("Released Intel inference was unavailable.")
            outputs[row["id"]] = result["objects"]
            print(row["id"], len(outputs[row["id"]]), flush=True)
    finally:
        engine.close()
    write_json(WORK / "released.json", dict(reference=digest(REFERENCE), outputs=outputs))


def measure():
    configure_cpu()
    rows = inputs()
    fixed = read_json(WORK / "released.json")
    if fixed["reference"] != digest(REFERENCE):
        raise ValueError("Released predictions refer to a different freeze.")
    before = fixed["outputs"]
    head = load_policy()
    model = CrowdedMasks(756, "xpu")
    after = {}
    for row in rows:
        key = row["id"]
        bank = dict(model.candidates(read_image(row["image"])))
        after[key] = [*before[key], *propose(before[key], bank, head)]
        print(key, len(row["objects"]), len(before[key]), len(after[key]), flush=True)
    old, new = (brief(rows, branch) for branch in (before, after))
    write_json(OUTPUT, dict(reference=digest(REFERENCE), baseline=old, additive=new, retention=compare(old, new),
                           photos={r["id"]: dict(expected=len(r["objects"]), baseline=len(before[r["id"]]),
                                                  additive=len(after[r["id"]])) for r in rows},
                           release_evidence="One positive source and two hard negatives constitute a limited pilot."))
    print({k: (old[k], new[k]) for k in ("tp", "fp", "exact_count")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("freeze", "released", "measure"))
    {"freeze": freeze, "released": released, "measure": measure}[parser.parse_args().job]()
