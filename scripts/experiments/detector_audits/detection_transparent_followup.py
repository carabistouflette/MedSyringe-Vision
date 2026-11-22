"""Qualify a fixed appearance-veto exception on separate public acquisitions."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.runtime.appearance_runtime import RELEASE as APPEARANCE
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.runtime.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_context_data import RELEASE as CONTEXT
from scripts.experiments.detector_training.detection_context_study import CACHE as CONTEXT_ROUTES, inputs as transparent
from scripts.experiments.crowds.detection_crowd_source_data import MANIFEST as CROWD
from scripts.experiments.detector_audits.detection_deskew_additive import CACHE as DESKEW_CACHE, OUTPUT as DESKEW_DEVELOPMENT
from scripts.experiments.detector_audits.detection_deskew_study import aligned_detection, orientation
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.detector_audits.detection_medbin_data import MANIFEST as MEDBIN
from scripts.experiments.detector_audits.detection_transparent_recalls import ExplicitAppearance, OUTPUT as DEVELOPMENT, on_image, originals, promote
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.datasets.package_detection_evidence import CAPTURE, MANIFEST, NATIVE, PUBLIC, confined
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

POLICY = ART / "transparent_verifier_audit_freeze.json"
OUTPUT = ART / "transparent_verifier_audit.json"
CACHE = DATA / "proposals/transparent-reserve-current.json"
ROTATION_IOU = .50


def live_groups():
    return {
        "bench-confirmation": [row for row in read_json(CONTEXT)["rows"]
                               if row["split"] == "device-context-reserve"],
        "medical-waste": read_json(MEDBIN)["rows"],
        "crowded-commons-confirmation": read_json(CROWD)["rows"],
    }


def freeze(bundle):
    if POLICY.exists() or CACHE.exists() or OUTPUT.exists():
        raise ValueError("The transparent-verifier comparison has already been opened.")
    qualified = [r for r in read_json(DEVELOPMENT)["leading"]
                 if r["development"]["transparent"]["guard"]["recovered_matches"]]
    deskew = read_json(DESKEW_DEVELOPMENT)
    if deskew["cache_sha256"] != digest(DESKEW_CACHE):
        raise ValueError("The orientation development observations changed.")
    original = read_json(CONTEXT_ROUTES)["frames"]
    rescues = read_json(DESKEW_CACHE)["frames"]
    transparent_rows = transparent()
    before = {row["id"]: rescues[row["id"]]["baseline"] for row in transparent_rows}
    gated = {name: accepted or corroborated(rescues[name]["supplemental"], originals(original[name], True))
             for name, accepted in before.items()}
    a, b = (evaluate(transparent_rows, value, 0.) for value in (before, gated))
    rotation = compare(a, b)
    rotation_qualified = (deskew["development_gate"] and rotation["passed"]
                          and rotation["recovered_matches"] > 0 and b["fp"] <= a["fp"])
    if not qualified and not rotation_qualified:
        raise ValueError("No development candidate improved transparent localization without additional errors.")
    choice = qualified[0]["options"] if qualified else None
    archive = read_json(bundle / MANIFEST)
    for cohort in (*PUBLIC, *CAPTURE, NATIVE):
        item = archive["cohorts"][cohort]
        if digest(bundle / confined(item["path"])) != item["sha256"]:
            raise ValueError(f"Invalid regression archive for {cohort}.")
    write_json(POLICY, dict(options=choice, orientation_fallback=rotation_qualified,
                            orientation_minimum_original_iou=ROTATION_IOU,
                            orientation_development=dict(previous=short(a), proposed=short(b), guard=rotation),
                            development_sha256=digest(DEVELOPMENT),
                            archive_sha256=digest(bundle / MANIFEST),
                            sources={str(p): digest(p) for p in (
                                APPEARANCE, CONTEXT, MEDBIN, CROWD, DESKEW_DEVELOPMENT, __file__,
                                "scripts/experiments/detector_audits/detection_transparent_recalls.py", "scripts/experiments/detector_audits/detection_deskew_study.py")},
                            precedence="Keep accepted original boxes. When the original is empty, prefer a restored "
                                       "original vetoed box. A single accepted rotated box can be used only where it "
                                       "agrees with a single originally vetoed box by at least 0.50 IoU.",
                            acceptance="Increase the matched-instance count on at least one of two sealed sources: "
                                       "the bench project or four independently sourced Commons scenes. "
                                       "Retain every already matched instance and every correct scene count "
                                       "and add zero false boxes on any scope. The archived sources are regression "
                                       "controls. Neither sealed source is used for tuning."))
    print("Conditional recovery sealed", choice, rotation_qualified, rotation["recovered_matches"], flush=True)


def corroborated(rotated, demoted):
    if len(rotated) == len(demoted) == 1 and box_iou(rotated[0]["bbox"], demoted[0]["bbox"]) >= ROTATION_IOU:
        return rotated
    return []


def registered(bundle):
    lock = read_json(POLICY)
    if (any(digest(p) != expected for p, expected in lock["sources"].items()) or
            lock["archive_sha256"] != digest(bundle / MANIFEST) or
            lock["development_sha256"] != digest(DEVELOPMENT)):
        raise ValueError("The release, evaluation inputs or policy have changed.")
    return lock


def collect(bundle):
    policy = registered(bundle)
    configure_cpu()
    stamp = digest(POLICY)
    stored = read_json(CACHE) if CACHE.exists() else dict(policy=stamp, frames={}, regression={})
    if stored["policy"] != stamp:
        raise ValueError("The saved bench detections use another release seal.")
    root = Path.cwd()
    engine = MedSyringeEngine(root)
    try:
        configure_device(engine, root, "xpu")
        if engine.identity.get("primary_backend") != "xpu":
            raise ValueError("Bench confirmation requires the pinned Intel runtime.")
        for cohort, rows in live_groups().items():
            observations = stored["frames"].setdefault(cohort, {})
            for row in rows:
                name = row["id"]
                if name in observations:
                    continue
                picture = source(row)
                started = perf_counter()
                engine.detection_checks.begin()
                ranked, floor, sam = engine.expanded.detect(picture)
                health = dict(model=engine.identity, sam3_status=sam)
                engine.detection_checks.attach(health)
                if health["detection_checks"]["status"] != "available":
                    raise ValueError(f"The detector was degraded for reserved image {name}.")
                accept = [p for p in ranked if p["score"] >= floor]
                demoted = originals(dict(ranked=ranked), True)
                angle = orientation(picture) if not accept and len(demoted) == 1 and policy["orientation_fallback"] else None
                view = aligned_detection(picture, angle, engine) if angle is not None else []
                observations[name] = dict(previous=accept, demoted=demoted,
                                          rotated=view, angle=angle, seconds=perf_counter() - started)
                write_json(CACHE, stored)
                print("Sealed real source", cohort, len(observations), "/", len(rows), flush=True)
        if policy["orientation_fallback"]:
            for cohort in PUBLIC:
                observed = stored["regression"].setdefault(cohort, {})
                rows, frames = archived(bundle, cohort)
                for row in rows:
                    identifier = row["id"]
                    if identifier in observed or frames[identifier]["previous"]:
                        continue
                    if len(frames[identifier]["demoted"]) != 1:
                        continue
                    image = archive_image(bundle, row)
                    orientation_ = orientation(image)
                    view = aligned_detection(image, orientation_, engine) if orientation_ is not None else []
                    observed[identifier] = dict(rotated=view, angle=orientation_)
                    write_json(CACHE, stored)
                    print("Archived rotation control", cohort, len(observed), flush=True)
    finally:
        engine.close()


def archived(bundle, rowset):
    item = read_json(bundle / MANIFEST)["cohorts"][rowset]
    path = bundle / confined(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError(f"The archived data changed for {rowset}.")
    record = read_json(path)
    rows = record["rows"]
    if rowset == NATIVE:
        candidate = {key: obs["actual"] for key, obs in record["observations"].items()}
        underlying = {key: obs["baseline"] for key, obs in record["observations"].items()}
    else:
        floor = record["floor"]
        candidate, underlying = ({k: [p for p in ranked if p["score"] >= floor]
                                  for k, ranked in record[field].items()} for field in ("candidate", "previous"))
    frames = {}
    for key in candidate:
        accepted = candidate[key]
        same_boxes = {tuple(p["bbox"]) for p in accepted}
        demoted = [p for p in underlying[key] if tuple(p["bbox"]) not in same_boxes]
        frames[key] = dict(previous=accepted, demoted=demoted)
    return rows, frames


def archive_image(bundle, row):
    path = bundle / "data" / confined(row["image"])
    if digest(path) != row.get("derived_sha256", row["sha256"]):
        raise ValueError(f"Archived reference raster changed for {row['id']}.")
    return read_image(path)


def measure(bundle):
    if OUTPUT.exists():
        raise ValueError("The separate datasets have already been scored.")
    policy = registered(bundle)
    saved = read_json(CACHE)
    independent = live_groups()
    if saved["policy"] != digest(POLICY) or any(set(saved["frames"].get(k, {})) != {r["id"] for r in rows}
                                              for k, rows in independent.items()):
        raise ValueError("Every registered real scene must be analyzed under the chosen release.")
    configure_cpu()
    groups = {key: archived(bundle, key) for key in (*PUBLIC, *CAPTURE, NATIVE)}
    groups.update({key: (rows, saved["frames"][key]) for key, rows in independent.items()})
    encoder = None
    combined = {}
    for key, (rows, frames) in groups.items():
        before = {name: f["previous"] for name, f in frames.items()}
        after = before.copy()
        triggers = {}
        for row in rows:
            name = row["id"]
            frame = frames[name]
            if before[name]:
                continue
            if key in (*CAPTURE, NATIVE):
                raise ValueError(f"An earlier positive control is empty for {name}.")
            live = key in independent
            if policy["options"] is not None and len(frame["demoted"]) == 1:
                image = source(row) if live else archive_image(bundle, row)
                if encoder is None:
                    encoder = ExplicitAppearance()
                demoted = frame["demoted"]
                _, margins = on_image(encoder, image, demoted)
                evidence = {**frame, "demoted": [{**p, **m} for p, m in zip(demoted, margins, strict=True)]}
                after[name] = promote(evidence, policy["options"])
                if after[name]:
                    triggers[name] = "appearance"
            if not after[name] and len(frame["demoted"]) == 1 and policy["orientation_fallback"]:
                alternate = frame if live else saved["regression"].get(key, {}).get(name)
                if alternate is None:
                    raise ValueError(f"A registered rotation control is missing for {name}.")
                after[name] = corroborated(alternate["rotated"], frame["demoted"])
                if after[name]:
                    triggers[name] = "orientation"
        a, b = (evaluate(rows, predictions, 0.) for predictions in (before, after))
        guard = compare(a, b)
        spoiled = [r["id"] for r in rows if len(before[r["id"]]) == len(r["objects"])
                   and len(after[r["id"]]) != len(r["objects"])]
        combined[key] = dict(current=short(a), prospective=short(b), comparison=guard,
                             changed=[name for name in after if before[name] != after[name]],
                             restored_by=triggers,
                             spoiled_counts=spoiled, safe=guard["passed"] and not spoiled and b["fp"] <= a["fp"])
    passed = all(v["safe"] for v in combined.values()) and any(
        combined[k]["comparison"]["recovered_matches"] > 0
        for k in ("bench-confirmation", "crowded-commons-confirmation"))
    write_json(OUTPUT, dict(passed=passed, policy_sha256=digest(POLICY), base_bench_sha256=digest(CACHE),
                            outcomes=combined))
    print("Independent recovery qualification", passed,
          {k: combined[k]["comparison"] for k in independent}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "collect", "measure"))
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "collect": collect, "measure": measure}[args.action](args.archive)
