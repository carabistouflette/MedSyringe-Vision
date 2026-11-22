"""Freeze and evaluate the graduated-fill estimator on local image references."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median

from PIL import Image, ImageOps

from medsyringe.measurement.attributes import crop
from medsyringe.measurement.capacity_ocr import resolve
from medsyringe.data import digest, read_json, write_json
from medsyringe.measurement.fill import estimate
from medsyringe.measurement.marking_views import agreement, supplement
from medsyringe.measurement.markings_guard import confirm_untyped_capacity
from scripts.experiments.expanded.expanded_proposals import image_for

ART = Path("artifacts/fill")
REFS = ART / "visual_reference.tsv"
LOCK = ART / "selection.json"
SOURCE_ROWS = Path("artifacts/expanded/annotations.json")
HOLDOUT = "orange-20-dark-wall"
HOLDOUT_60 = "orange-60-dark-wall"
LEGACY_ROWS = Path("artifacts/marking_detected_manifest.json")
LEGACY_PHOTOMETRY = Path("artifacts/liquid_detected_results.json")
LEGACY_TOKENS = tuple(Path(f"data/ocr-v6/markings-detected{k}.json") for k in ("", "-guided"))


def references():
    rows = {r["id"]: r for r in read_json(SOURCE_ROWS)["rows"]}
    labels = {}
    for line in REFS.read_text().splitlines():
        if line and not line.startswith("#"):
            identifier, low, high = line.split()
            key = f"IMG_{identifier}"
            if key in labels or float(low) > float(high):
                raise ValueError(f"Invalid repeat reference for {key}.")
            labels[key] = (float(low), float(high))
    return rows, labels


def joined_reading(row):
    if "clahe" not in row:
        return row["baseline"]
    joint = agreement(row["clahe"], row["enhanced"])
    return confirm_untyped_capacity(supplement(row["baseline"], joint), joint)


def older_crops(ids, labels):
    images = {r["id"]: r for r in read_json(LEGACY_ROWS)["files"]}
    raw, guided = (read_json(p)["frames"] for p in LEGACY_TOKENS)
    liquid = read_json(LEGACY_PHOTOMETRY)["frames"]
    for identifier in ids:
        row = images[identifier]
        if digest(row["image"]) != row["sha256"]:
            raise ValueError(f"The detector crop changed for {identifier}.")
        tokens = raw[identifier]["tokens"] + guided[identifier]["tokens"]
        texts = confirm_untyped_capacity(resolve(tokens, cutoff=.74, scale_cutoff=.80), {})
        with Image.open(row["image"]) as region:
            yield identifier, ImageOps.contain(region.convert("RGB"), (600, 600)), texts, liquid[identifier], labels[identifier]


def prepared(source, split):
    rows, labels = references()
    if split == "holdout-60":
        yield from older_crops([k for k in labels if rows[k]["group"] == HOLDOUT_60], labels)
        return
    if split == "clear":
        path = "test"
        appearance = read_json("artifacts/expanded/liquid_test.json")["legacy_frames"]
        ids = [k for k, r in rows.items() if r["split"] == "test"]
    else:
        path = "calibration"
        appearance = read_json("artifacts/expanded/liquid_calibration.json")["legacy_frames"]
        ids = [k for k in labels if (rows[k]["group"] == HOLDOUT) == (split == "holdout")
               and rows[k]["split"] == "calibration"]
    readings = read_json(f"data/expanded/markings-{path}.json")["frames"]
    for identifier in ids:
        row = rows[identifier]
        region = crop(image_for(row, source), row["objects"][0]["bbox"])
        result = joined_reading(readings[identifier]["readings"])
        yield identifier, ImageOps.contain(region, (600, 600)), result, appearance[identifier]["measurement"], labels.get(identifier)
    if split == "development":
        names = [k for k in labels if rows[k]["group"] == "amber-20-low-fill-dark-wall"]
        yield from older_crops(names, labels)


def score(items, settings):
    cases = {}
    for identifier, image, texts, liquid, expected in items:
        result = estimate(image, texts, liquid, settings)
        record = dict(result=result, reference_ml=expected)
        if result["status"] == "visual estimate" and expected is not None:
            low, high = expected
            record["error_beyond_bracket_ml"] = max(low-result["ml"], result["ml"]-high, 0.)
            interval = result["interval_ml"]
            record["interval_overlaps"] = max(low, interval[0]) <= min(high, interval[1])
        cases[identifier] = record
    answered = [v for v in cases.values() if v["result"]["status"] == "visual estimate"]
    errors = [p["error_beyond_bracket_ml"] for p in answered if "error_beyond_bracket_ml" in p]
    comparable = [p for p in answered if "interval_overlaps" in p]
    summary = dict(images=len(cases), answered=len(answered), median_error_ml=median(errors) if errors else None,
                   worst_error_ml=max(errors, default=None), within_visible_bracket=sum(e == 0. for e in errors),
                   within_one_minor_mark=sum(p["error_beyond_bracket_ml"] <= (1 if p["result"]["capacity_ml"] == 20 else 2)
                                             for p in comparable),
                   intervals_overlap=sum(p["interval_overlaps"] for p in comparable),
                   false_clear_answers=sum(p["reference_ml"] is None for p in answered))
    return summary, cases


def inputs():
    items = (REFS, SOURCE_ROWS, Path("medsyringe/measurement/fill.py"), Path(__file__),
             Path("artifacts/expanded/liquid_calibration.json"), Path("data/expanded/markings-calibration.json"),
             LEGACY_ROWS, LEGACY_PHOTOMETRY, *LEGACY_TOKENS)
    return {str(p): digest(p) for p in items}


def select(source):
    if LOCK.exists():
        raise ValueError("The older selection is already frozen.")
    items = list(prepared(source, "development"))
    trials = []
    for minimum, drop, continuity, ticks in product((.57, .68, .77), (8., 11., 15.), (.66, .80), (False, True)):
        cfg = dict(ocr_min=minimum, min_drop=drop, continuity=continuity, tick_check=ticks)
        summary, _ = score(items, cfg)
        trials.append(dict(settings=cfg, metrics=summary))
    def order(trial):
        m = trial["metrics"]
        outliers = m["answered"]-m["within_one_minor_mark"]
        return (-outliers, m["intervals_overlap"], m["within_visible_bracket"], m["answered"],
                trial["settings"]["continuity"], trial["settings"]["min_drop"])

    trials.sort(key=order, reverse=True)
    selected = trials[0]
    settings = selected["settings"]
    write_json(LOCK, dict(inputs=inputs(), heldout_groups=[HOLDOUT, HOLDOUT_60], selected=selected, trials=trials))
    _, cases = score(items, settings)
    write_json(ART / "development.json", dict(metrics=selected["metrics"], cases=cases))
    print(selected, flush=True)


def audit(source):
    frozen = read_json(LOCK)
    if frozen["inputs"] != inputs():
        raise ValueError("The frozen measurement inputs changed.")
    results = {}
    for cohort in ("holdout", "holdout-60", "clear"):
        summary, cases = score(list(prepared(source, cohort)), frozen["selected"]["settings"])
        results[cohort] = dict(metrics=summary, cases=cases)
        print(cohort, summary, flush=True)
    wrote = results["holdout"]["metrics"]
    large = results["holdout-60"]["metrics"]
    passed = (wrote["answered"] >= 3 and wrote["within_one_minor_mark"] == wrote["answered"] and
              wrote["intervals_overlap"] == wrote["answered"] and large["answered"] >= 2 and
              large["within_one_minor_mark"] == large["answered"] and
              large["intervals_overlap"] == large["answered"] and results["clear"]["metrics"]["false_clear_answers"] == 0)
    write_json(ART / "audit.json", dict(passed=passed, cohorts=results, selection_sha256=digest(LOCK),
               scope="Graduation-bracket references from one photo session. No calibrated liquid-volume readings."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "audit"))
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    (select if args.stage == "select" else audit)(args.source)


if __name__ == "__main__":
    main()
