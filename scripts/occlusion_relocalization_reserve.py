"""Freeze new public photographs and score the locked box-refinement rule locally."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.evaluate import evaluate, match
from medsyringe.grounding_backend import analyze
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.occlusion_detector import ART, FREEZE, TRAIN, frozen_frames, inference, partitions, path_for, raster
from scripts.occlusion_relocalization import POLICY, diagnostics, hypotheses
from scripts.part_verifier_study import configure_cpu
from scripts.public_training_intake import raster_hashes

DATA = Path("data/occlusion-relocalization")
REFERENCE = ART / "occlusion_relocalization_reserve.json"
RESULT = ART / "occlusion_relocalization_prospective.json"

# Reviewed pixel bounds on the untouched DVIDS preview, without model outputs.
SOURCES = (
    ("8711851", "2410", "Belinda Guachun-Chichay", "Luke AFB 2024-10-09",
     "mission-ready-56th-fw-rapid-flu-shot-pod", (1000, 665),
     [(454, 82, 516, 403), (40, 207, 116, 425), (68, 194, 133, 435), (89, 185, 143, 412),
      (130, 190, 163, 417), (670, 548, 787, 665), (717, 558, 850, 665),
      (778, 571, 906, 665), (842, 596, 929, 665)],
     "One held syringe, four out-of-focus syringes on the left and four clipped assemblies at the lower right. "
     "Blur and clipping increase uncertainty in the visible extents."),
    ("8917760", "2503", "Adriana Barrientos", "Kingstown 2025-03-11",
     "lamat-2025-medical-partnership-st-vincent-and-grenadines", (999, 665), [(366, 19, 515, 339)],
     "One syringe under a hand, with exposed plunger and hub; separate vial excluded."),
    ("8948275", "2504", "Carlos Chavez-Flores", "MCB Hawaii 2025-03-27",
     "casualty-care-us-service-members-conduct-valkyrie-emergency-fresh-whole-blood-transfusion-training-mcbh",
     (1000, 1500), [(252, 465, 554, 1008)],
     "One visible syringe with needle inserted into a distinct vial."),
    ("9090709", "2506", "Daniel Martinez", "Offutt 2025-06-02",
     "mdg-cares-offutt-patients", (1000, 714), [],
     "Precision pipette and vials. No syringe assembly visible."),
    ("9364294", "2510", "Ricardo Reyes-Guevara", "Bethesda 2025-10-16",
     "walters-way-behind-scenes-lab-professionals-at-walter-reed-national-military-medical-center",
     (1000, 688), [], "Pipette, tubes, and centrifuge. No syringe assembly visible."),
)


def protocol():
    return dict(selection=digest(POLICY), implementation=digest(__file__), detector_training=digest(TRAIN),
                operating_points=digest(FREEZE))


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    if not read_json(POLICY)["selected"]:
        raise ValueError("The selection step did not find a qualified rule.")
    history, private, private_visual = protection()
    fit, dev, _ = partitions()
    past, _ = frozen_frames()
    archived = [*fit, *dev, *(r for rows in past.values() for r in rows)]
    hashes = history["sha256"] | private | {row["sha256"] for row in archived}
    visual = history["dhash"] | private_visual
    for row in archived:
        visual.update(raster_hashes(path_for(row)))
    rows = []
    for key, month, author, group, title, (w, h), boxes, note in SOURCES:
        path = DATA / f"{key}.jpg"
        sha256 = digest(path)
        original_url = f"https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/{month}/{key}/1000w_q95.jpg"
        source_url = f"https://www.dvidshub.net/image/{key}/{title}"
        aliases = {a.replace("-", " ") for a in history["authors"]}
        author_key = author.casefold().replace("-", " ")
        if (key in history["ids"] or author_key in aliases or any(key in u for u in history["urls"])
                or sha256 in hashes or fingerprint_collision(raster_hashes(path), visual)):
            raise ValueError(f"Previously indexed source or near-duplicate: {key}")
        row = dict(id=f"dvids-{key}", image=str(path), sha256=sha256, author=author, group=group,
                   source_url=source_url, original_url=original_url, license="Public domain (DVIDS)",
                   objects=[dict(bbox=[l/w, t/h, r/w, b/h]) for l, t, r, b in boxes], review=note)
        if raster(row).size != (w, h) or any(not valid_box(p["bbox"]) for p in row["objects"]):
            raise ValueError(f"Invalid annotation or source raster: {key}")
        rows.append(row)
        hashes.add(sha256)
        visual.update(raster_hashes(path))
    write_json(REFERENCE, dict(protocol=protocol(), rows=rows, grouped=len({r["group"] for r in rows}),
                              exclusions=dict(dvids_9340582="Dense defocused tray; complete count and individual "
                                                              "box assignment too ambiguous for this reserve."),
                              filter="Known source metadata, SHA-256, rotated/mirrored dHash within Hamming radius 4.",
                              gate="At least one new match at IoU 0.50, zero lost matches at IoU 0.50 or 0.75, "
                                   "and zero additional false-positive boxes and count regressions."))


def refs():
    selection = read_json(REFERENCE)
    if selection["protocol"] != protocol():
        raise ValueError("The locked prospective protocol changed.")
    return selection["rows"]


def baseline():
    from time import perf_counter

    configure_cpu()
    rows = refs()
    output = DATA / "released.json"
    engine = MedSyringeEngine(Path.cwd())
    configure_device(engine, Path.cwd(), "xpu")
    try:
        predictions, seconds = {}, {}
        for row in rows:
            started = perf_counter()
            result = analyze(engine, raster(row), False)
            checks = result["detection_checks"]
            if checks["status"] != "available" or checks["primary_backend"] != "xpu":
                raise ValueError(f"Released detector unavailable for {row['id']}.")
            predictions[row["id"]] = result["objects"]
            seconds[row["id"]] = perf_counter()-started
            print(row["id"], "done", flush=True)
        write_json(output, dict(reference=digest(REFERENCE), predictions=predictions, seconds=seconds))
    finally:
        engine.close()


def collect():
    predictions, latency = inference(refs())
    write_json(DATA / "challenger.json", dict(reference=digest(REFERENCE), predictions=predictions,
                                            median_latency_ms=latency))


def annotate_matches(rows, predictions):
    assignments, _ = match(rows, predictions, .50)
    result = {}
    for row in rows:
        key = row["id"]
        result[key] = dict(expected=len(row["objects"]), count=len(predictions[key]),
                           boxes=[p["bbox"] for p in predictions[key]],
                           matched=sum(item[2] is not None for item in assignments if item[0] == key))
    return result


def measure():
    rows = refs()
    released_path, model_path = DATA / "released.json", DATA / "challenger.json"
    release, model = read_json(released_path), read_json(model_path)
    if any(record["reference"] != digest(REFERENCE) for record in (release, model)):
        raise ValueError("The prospective outputs refer to a different validation cohort.")
    original = release["predictions"]
    original_values = evaluate(rows, original, 0.)
    original_cases = annotate_matches(rows, original)
    selection = read_json(POLICY)["selected"]["policy"]
    trials = hypotheses(rows, original, model["predictions"], selection, read_json(FREEZE)["floors"])
    comparison = {}
    for name, outputs in trials.items():
        summary = diagnostics(rows, original, outputs, detailed=True)
        cases = annotate_matches(rows, outputs)
        summary["per_image"] = cases
        new_count_error = any(abs(r["count"]-r["expected"]) >
                              abs(original_cases[key]["count"]-r["expected"]) for key, r in cases.items())
        new_false_positive = any(r["count"]-r["matched"] >
                                 original_cases[key]["count"]-original_cases[key]["matched"]
                                 for key, r in cases.items())
        summary["passes_prospective_gate"] = bool(
            summary["iou50"]["gained"] and not summary["iou50"]["lost"] and not summary["iou75"]["lost"]
            and not new_count_error and not new_false_positive
        )
        comparison[name] = summary
    write_json(RESULT, dict(reference=digest(REFERENCE), caches={str(p): digest(p) for p in (released_path, model_path)},
                            base=original_values, baseline_cases=original_cases, trials=comparison,
                            model_median_latency_ms=model["median_latency_ms"]))
    print({key: (item["metrics"]["tp"], item["metrics"]["fp"], item["metrics"]["exact_count"],
                 item["passes_prospective_gate"]) for key, item in comparison.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "baseline", "collect", "measure"))
    {"freeze": freeze, "baseline": baseline, "collect": collect, "measure": measure}[parser.parse_args().action]()
