"""Compare contrast preprocessing for local OCR on reviewed syringe crops."""
import argparse
from pathlib import Path
from time import perf_counter

from PIL import ImageOps

from medsyringe.attributes import crop
from medsyringe.capacity_ocr import SceneOCR, resolve
from medsyringe.data import digest, read_json, write_json
from medsyringe.marking_views import agreement, contrast_view, supplement
from medsyringe.ocr_refine import guided_read
from scripts.expanded_annotations import ANNOTATIONS, ART
from scripts.expanded_proposals import CACHE, image_for

RELEASE = Path("artifacts/capacity_ocr_release.json")
LOCK = ART / "markings_freeze.json"


def read(reader, image, settings):
    tokens = reader.read(image)
    if settings["guided"]:
        tokens += guided_read(reader, image, tokens)
    return resolve(tokens, cutoff=settings["cutoff"], repeats=settings["repeats"],
                   scale_cutoff=settings["scale_cutoff"])


def batch(source, cohort):
    if cohort == "test" and not LOCK.is_file():
        raise ValueError("Freeze the marking candidate before reading held-out crops.")
    rows = [r for r in read_json(ANNOTATIONS)["rows"] if r["split"] == cohort]
    path = CACHE / f"markings-{cohort}.json"
    key = {str(p): digest(p) for p in (RELEASE, ANNOTATIONS, Path(__file__), Path("medsyringe/marking_views.py"))}
    previous = read_json(path) if path.is_file() else {}
    frames = previous["frames"] if previous.get("key") == key else {}
    options = read_json(RELEASE)["options"]
    reader = None
    for position, row in enumerate(rows, 1):
        key_id = row["id"]
        if key_id in frames:
            continue
        if reader is None:
            reader = SceneOCR()
        region = ImageOps.contain(crop(image_for(row, source), row["objects"][0]["bbox"]), (1900, 1900))
        tick = perf_counter()
        baseline = read(reader, region, options)
        seconds = dict(baseline=perf_counter()-tick)
        readings = dict(baseline=baseline)
        if not baseline.get("capacity") or not baseline.get("scale"):
            for method in ("clahe", "enhanced"):
                tick = perf_counter()
                readings[method] = read(reader, contrast_view(region, method), options)
                seconds[method] = perf_counter()-tick
        frames[key_id] = dict(readings=readings, group_scale=row["group_scale"], group=row["group"], seconds=seconds)
        write_json(path, dict(key=key, frames=frames))
        print(cohort, position, key_id, "seconds", round(sum(seconds.values()), 2), flush=True)


def variants(record):
    original = record["baseline"]
    if "clahe" not in record:
        return {k: original for k in ("baseline", "clahe", "enhanced", "both")}
    return dict(baseline=original, clahe=supplement(original, record["clahe"]),
                enhanced=supplement(original, record["enhanced"]),
                both=supplement(original, agreement(record["clahe"], record["enhanced"])))


def evaluate(cohort):
    from statistics import median

    input_file = CACHE / f"markings-{cohort}.json"
    frames = read_json(input_file)["frames"]
    expected = sum(r["split"] == cohort for r in read_json(ANNOTATIONS)["rows"])
    if len(frames) != expected:
        raise ValueError(f"Expected {expected} frames, got {len(frames)}.")
    totals, evidence = {}, {}
    for name in ("baseline", "clahe", "enhanced", "both"):
        totals[name] = {field: dict(correct=0, wrong=0, abstained=0) for field in ("capacity", "scale")}
        evidence[name] = {}
        elapsed = []
        for identifier, row in frames.items():
            result = variants(row["readings"])[name]
            refs = dict(scale=row["group_scale"], capacity=float(row["group_scale"].split("-")[0]))
            evidence[name][identifier] = {key: result.get(key) for key in refs}
            for field, truth in refs.items():
                prediction = result.get(field)
                outcome = "abstained" if not prediction else "correct" if prediction["value"] == truth else "wrong"
                totals[name][field][outcome] += 1
            additions = ("clahe", "enhanced") if name == "both" else (name,) if name != "baseline" else ()
            elapsed.append(row["seconds"]["baseline"] + sum(row["seconds"].get(key, 0.) for key in additions))
        totals[name]["median_seconds"] = median(elapsed)
    write_json(ART / f"markings_{cohort}.json", dict(images=len(frames), totals=totals, evidence=evidence,
               scope="Reference crop. Physical scale uses reviewed sequence context. Per-frame legibility is separate."))
    if cohort == "calibration":
        base = totals["baseline"]
        viable = [k for k, v in totals.items() if all(v[f]["wrong"] <= base[f]["wrong"] for f in ("capacity", "scale"))]
        # An extra reading must agree between both enhanced views to qualify for release.
        eligible = [k for k in ("baseline", "both") if k in viable]
        selected = max(eligible, key=lambda k: (sum(totals[k][f]["correct"] for f in ("capacity", "scale")),
                                                  -totals[k]["median_seconds"]))
        write_json(LOCK, dict(selected=selected, evidence_sha256=digest(ART / "markings_calibration.json"),
                              annotations_sha256=digest(ANNOTATIONS)))
    print(cohort, totals, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cohort", choices=("calibration", "test"))
    parser.add_argument("stage", choices=("batch", "evaluate"))
    args = parser.parse_args()
    batch(args.source, args.cohort) if args.stage == "batch" else evaluate(args.cohort)


if __name__ == "__main__":
    main()
