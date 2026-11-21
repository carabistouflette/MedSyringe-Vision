"""Select alternate fill evidence on development groups, then audit it once."""
import argparse
from itertools import product
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.fill import estimate
from medsyringe.graduation_geometry import estimate as alternate
from scripts.fill_study import ART, inputs, prepared

LOCK = ART / "geometry_selection.json"
AUDIT = ART / "geometry_regression.json"
RELEASE = ART / "release.json"


def locked_inputs():
    paths = (__file__, "medsyringe/graduation_geometry.py", RELEASE)
    return {**inputs(), **{str(p): digest(p) for p in paths}}


def evaluate(items, settings, strategy):
    cases = []
    counts = dict(images=len(items), baseline=0, added=0, correct=0, within_one_mark=0, false_answers=0)
    for name, crop, inscriptions, appearance, expected in items:
        baseline = estimate(crop, inscriptions, appearance, settings)
        answer = alternate(crop, inscriptions, appearance, settings, strategy) if "ml" not in baseline else None
        counts["baseline"] += "ml" in baseline
        if answer is None:
            continue
        counts["added"] += 1
        error = max(expected[0]-answer["ml"], answer["ml"]-expected[1], 0.) if expected else None
        counts["correct"] += error == 0.
        counts["within_one_mark"] += error is not None and error <= (1 if answer["capacity_ml"] == 20 else 2)
        counts["false_answers"] += expected is None
        cases.append(dict(id=name, reference_ml=expected, error_ml=error, baseline_reason=baseline["reason"],
                          candidate=answer))
    return dict(counts=counts, added_cases=cases)


def select(source):
    if LOCK.exists():
        raise ValueError("The geometry candidate has already been frozen.")
    items = list(prepared(source, "development"))
    settings = read_json(RELEASE)["settings"]
    trials = []
    for span, use_scale in product((.24, .34, .44), (False, True)):
        strategy = dict(min_span=span, ordered_scale=use_scale)
        result = evaluate(items, settings, strategy)
        trials.append(dict(strategy=strategy, **result))
    eligible = [trial for trial in trials if (trial["counts"]["added"] > 0 and
                trial["counts"]["added"] == trial["counts"]["correct"])]
    key = lambda t: (t["counts"]["correct"], t["strategy"]["min_span"], not t["strategy"]["ordered_scale"])
    winner = max(eligible, key=key, default=None)
    write_json(LOCK, dict(inputs=locked_inputs(), trials=trials, selected=winner,
                         shadow=winner or max(trials, key=key),
                         criterion="At least one additional prediction, all inside visual reference brackets."))
    print([(t["strategy"], t["counts"]) for t in trials])


def audit(source):
    freeze = read_json(LOCK)
    if freeze["inputs"] != locked_inputs():
        raise ValueError("Geometry study inputs changed since selection.")
    strategy = freeze["shadow"]["strategy"]
    settings = read_json(RELEASE)["settings"]
    cases = {split: evaluate(list(prepared(source, split)), settings, strategy)
             for split in ("holdout", "holdout-60", "clear")}
    counts = [v["counts"] for v in cases.values()]
    passed = bool(freeze["selected"] and sum(c["added"] for c in counts) and
                  all(c["correct"] == c["added"] for c in counts))
    write_json(AUDIT, dict(passed=passed, strategy=strategy, selection_sha256=digest(LOCK), cases=cases,
                           scope="Previously consulted groups. Regression only, not new independent evidence."))
    print(passed, {k: v["counts"] for k, v in cases.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "audit"))
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    (select if args.stage == "select" else audit)(args.source)


if __name__ == "__main__":
    main()
