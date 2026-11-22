"""Cache local detector branches without exposing the held-out images to selection."""
import argparse
from pathlib import Path
from time import perf_counter

from PIL import ImageOps

from medsyringe.data import digest, read_image, read_json, write_json
from scripts.experiments.expanded.expanded_dataset import ART

CACHE = Path("data/expanded")
LOCK = ART / "detector_freeze.json"


def public_rows(cohort):
    reserve = read_json("artifacts/sam3_reserve.json")["rows"]
    pens = read_json("artifacts/rfdetr_pen_manifest.json")["rows"]
    if cohort == "reserve":
        return reserve + [r for r in pens if r["split"] == "pen_reserve"]
    blocked_groups = {r["group"] for r in public_rows("reserve")}
    blocked_hashes = {r["sha256"] for r in public_rows("reserve")}
    if cohort == "fit":
        data = read_json("artifacts/precision_study_manifest.json")["train"]
        data += [r for r in pens if r["split"] == "pen_training"]
    else:
        data = read_json("artifacts/precision_study_manifest.json")["test"]
        data += [r for r in pens if r["split"] == "pen_validation"]
        blocked_groups.update(r["group"] for r in public_rows("fit"))
        blocked_hashes.update(r["sha256"] for r in public_rows("fit"))
    unique = {r["sha256"]: r for r in data if r["group"] not in blocked_groups
              and r["sha256"] not in blocked_hashes}
    return sorted(unique.values(), key=lambda r: r["id"])


def rows_for(cohort):
    inventory = read_json(ART / "inventory.json")
    if not inventory["leakage_check"]["passed"]:
        raise ValueError("The capture inventory failed the overlap checks.")
    capture = [r for r in inventory["rows"] if r["split"] == cohort]
    return capture + (public_rows(cohort) if cohort != "test" else [])


def image_for(row, source):
    path = source / row["file"] if "file" in row else Path("data") / row["image"]
    if digest(path) != row.get("derived_sha256", row["sha256"]):
        raise ValueError(f"Source checksum mismatch for {row['id']}.")
    return read_image(path)


def cache_file(cohort, branch):
    return CACHE / f"{cohort}-{branch}.json"


def cache(cohort, branch):
    return read_json(cache_file(cohort, branch))["frames"]


def collect(source, cohort, branch):
    from scripts.experiments.expanded.sam3_study import configure_cpu

    configure_cpu()
    if cohort in {"test", "reserve"} and not LOCK.is_file():
        raise ValueError("Freeze the detector candidate before running holdout inference.")
    rows = rows_for(cohort)
    locked_paths = [ART / "inventory.json", Path(__file__), *map(Path, (
        "artifacts/detector_release.json", "artifacts/rfdetr_verified_release.json", "artifacts/sam3_release.json",
    ))]
    key = dict(files={str(p): digest(p) for p in locked_paths}, rows={r["id"]: r["sha256"] for r in rows})
    target = cache_file(cohort, branch)
    previous = read_json(target) if target.is_file() else {}
    frames = previous["frames"] if previous.get("key") == key else {}
    if len(frames) == len(rows):
        print(f"{cohort} {branch}: {len(rows)} cached", flush=True)
        return
    if branch in {"cpu", "xpu"}:
        from medsyringe.analysis.inference import Engine

        engine = Engine(Path("."))
        floor = engine.config["cutoff"]
        if branch == "xpu":
            from medsyringe.runtime.xpu_grounding import AcceleratedGrounder

            engine.detector = AcceleratedGrounder(engine.detector)
    else:
        from itertools import chain

        from medsyringe.detection.additive import overlapping_tiles
        from medsyringe.detection.detectors import suppress
        from scripts.experiments.expanded.sam3_study import Segmenter, crop_predictions

        cpu = cache(cohort, "xpu" if branch == "fallback-xpu" else "cpu") if branch.startswith("fallback") else {}
        floor = read_json("artifacts/detector_release.json")["cutoff"]
        segmenter = Segmenter(504, "xpu")
    for index, row in enumerate(rows, 1):
        identifier = row["id"]
        if identifier in frames:
            continue
        if branch.startswith("fallback") and any(p["score"] >= floor for p in cpu[identifier]["prior"]):
            frames[identifier] = dict(tiles=[], seconds=0.)
            continue
        image = image_for(row, source)
        started = perf_counter()
        if branch in {"cpu", "xpu"}:
            primary = engine.rank_detections(image)
            elapsed = perf_counter() - started
            result = dict(primary=primary, prior=engine.recovery.apply(image, primary, floor),
                          primary_seconds=elapsed)
            result["cascade_seconds"] = perf_counter() - started
            started_rf = perf_counter()
            result["rf"] = engine.recovery._region(image)
            result["rf_seconds"] = perf_counter() - started_rf
        elif branch == "sam":
            result = dict(full=segmenter.predict(ImageOps.contain(image, (1500, 1500))))
        else:
            windows = chain([(0, 0, *image.size)], overlapping_tiles(image.size, .66))
            result = dict(tiles=suppress(list(crop_predictions(segmenter.predict, image, windows)), .45))
        result["seconds"] = perf_counter() - started
        frames[identifier] = result
        write_json(target, dict(key=key, frames=frames))
        if index % 8 == 0 or index == 1:
            print(cohort, branch, index, identifier, round(result["seconds"], 3), flush=True)
    write_json(target, dict(key=key, frames=frames))
    print(f"{cohort} {branch}: {len(frames)} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cohort", choices=("fit", "calibration", "test", "reserve"))
    parser.add_argument("branch", choices=("cpu", "xpu", "sam", "fallback", "fallback-xpu"))
    args = parser.parse_args()
    collect(args.source, args.cohort, args.branch)


if __name__ == "__main__":
    main()
