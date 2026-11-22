"""Run live inference on the supplied collection and score its draft boxes."""
import argparse
from collections import Counter
from pathlib import Path

from medsyringe.data import digest, read_image, read_json, train_rows, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.analysis.inference import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    artifacts = args.root / "artifacts"
    inventory = read_json(artifacts / "source_inventory.json")
    expected_files = {row["source"] for row in inventory}
    source_files = {path.relative_to(args.source).as_posix()
                    for path in args.source.rglob("*") if path.is_file()}
    if source_files != expected_files:
        raise ValueError("Source file list differs from the original inventory.")
    for row in inventory:
        if digest(args.source / row["source"]) != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {row['source']}")
    rows = train_rows(args.root)
    engine = Engine(args.root)
    predictions, frames = {}, []
    attributes = {field: Counter() for field in ("scale", "color")}
    for i, row in enumerate(rows, 1):
        result = engine.analyze(read_image(args.source / row["source"]))
        predictions[row["id"]] = result["objects"]
        observed = result["objects"]
        expected = row["objects"][0]
        overlap = max((box_iou(o["bbox"], expected["bbox"]) for o in observed), default=0)
        if len(observed) == 1 and overlap >= .5:
            for field, counts in attributes.items():
                if expected[field] is not None:
                    counts["eligible"] += 1
                    counts["answered"] += observed[0][field] is not None
                    counts["correct"] += observed[0][field] == expected[field]
        frames.append(dict(id=row["id"], count=len(observed), best_iou=overlap,
                           detections=observed, elapsed_ms=result["elapsed_ms"]))
        if i % 10 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} original images processed", flush=True)
    keys = ("tp", "fp", "fn", "exact_count")
    metrics = evaluate(rows, predictions, result["threshold"])
    scores = {key: metrics[key] for key in keys}
    write_json(artifacts / "optimized_fil_rouge_live.json", dict(
        scope="Live inference on development originals against draft references",
        original_files_verified=len(inventory), unique_images=len(rows),
        duplicates=[dict(source=r["source"], canonical=r["duplicate_of"])
                    for r in inventory if r.get("duplicate_of")],
        release_sha256=digest(artifacts / "detector_release.json"),
        cutoff=result["threshold"], scores=scores,
        attribute_diagnostics=attributes, frames=frames,
    ))
    print(scores)


if __name__ == "__main__":
    main()
