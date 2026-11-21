"""Rescore post-hoc development label corrections without changing the frozen selection."""
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.expanded_detection_study import ART, BASE, LOCK, load, short
from scripts.expanded_recall_selection import GUARDED, score


def main():
    revised = ART / "annotations_adjudicated.json"
    annotations = read_json(revised)
    updated = {r["id"]: r for r in annotations["rows"]}
    settings = read_json(GUARDED)["selected"]
    if settings is None:
        return
    head = read_json(LOCK)["heads"]
    floor = read_json(BASE)["cutoff"]
    joined = {}
    for split in ("fit", "calibration"):
        original, frames = load(split)
        capture = [row for row in original if row["id"] in updated]
        capture = [updated[row["id"]] for row in capture]
        subset = {row["id"]: frames[row["id"]] for row in capture}
        baseline = evaluate(capture, {k: f["prior"] for k, f in subset.items()}, floor)
        result = score(capture, subset, settings["policy"], head, floor)
        joined[split] = dict(baseline=short(baseline), candidate=short(result), guard=compare(baseline, result))
    write_json(ART / "localization_sensitivity.json", dict(slices=joined, reference_sha256=digest(revised),
               scope="Post-score development annotation checks. These scores were not used for selection."))
    print(joined, flush=True)


if __name__ == "__main__":
    main()
