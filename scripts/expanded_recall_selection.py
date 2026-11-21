"""Freeze a stricter detector candidate with no lost fit or calibration matches."""
import argparse
from itertools import product

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.hybrid_detection import retain_baseline
from medsyringe.recall_guard import compare
from scripts.expanded_detection_study import ART, BASE, LOCK, TRIALS, apply, load, short

GUARDED = ART / "detector_recall_freeze.json"


def hybrid(frame, policy, heads, baseline_cutoff):
    options = policy["options"]
    ranked = apply(frame, options, heads)
    guard = policy.get("guard")
    if guard is None:
        return ranked
    minimum, area_limit = guard
    protected = []
    for p in frame["prior"]:
        x, y, r, b = p["bbox"]
        if p["score"] >= minimum and (r-x)*(b-y) <= area_limit:
            protected.append(p)
    return retain_baseline(protected, ranked, options["cutoff"], baseline_cutoff)


def score(rows, frames, policy, heads, floor):
    return evaluate(rows, {k: hybrid(f, policy, heads, floor) for k, f in frames.items()}, policy["options"]["cutoff"])


def select():
    if GUARDED.exists():
        raise ValueError("This recall study is already fixed.")
    if (ART / "detector_reserve.json").exists():
        raise ValueError("Freeze this study before scoring the public reserve.")
    initial = read_json(LOCK)
    train, fit = load("fit")
    valid, calibration = load("calibration")
    floor = read_json(BASE)["cutoff"]
    before_fit = evaluate(train, {k: f["prior"] for k, f in fit.items()}, floor)
    before_valid = evaluate(valid, {k: f["prior"] for k, f in calibration.items()}, floor)
    heads = initial["heads"]
    considered = []
    regular = [dict(options=t["options"], guard=None) for t in read_json(TRIALS)["trials"]
               if t["eligible"] and not t["options"]["method"].startswith("gradient")]
    protected = [dict(options=initial["selected"]["options"], guard=[s, a]) for s, a in
                 product((floor, .72, .8, .85), (.045, .09, .16, .30, 1.))]
    for policy in [*regular, *protected]:
        outcome = {}
        for name, cohort, frames, before in (("fit", train, fit, before_fit),
                                             ("calibration", valid, calibration, before_valid)):
            result = score(cohort, frames, policy, heads, floor)
            outcome[name] = dict(result=short(result), guard=compare(before, result), max_fp=before["fp"])
        eligible = (all(o["guard"]["passed"] and o["result"]["fp"] <= o["max_fp"] for o in outcome.values())
                    and outcome["calibration"]["result"]["f1"] > before_valid["f1"])
        considered.append(dict(policy=policy, outcome=outcome, eligible=eligible))
    candidates = [t for t in considered if t["eligible"]]
    selected = max(candidates, key=lambda t: (t["outcome"]["calibration"]["result"]["f1"],
                   t["outcome"]["calibration"]["result"]["ap50_95"], t["policy"]["guard"] is None), default=None)
    write_json(GUARDED, dict(selected=selected, compared=considered, base_freeze=digest(LOCK),
                            selection_inputs=["fit", "calibration"],
                            new_clear_scope="The parent clear-block report is not consulted by selection."))
    print("Strict recall selection", len(candidates), "eligible", selected, flush=True)


def audit(cohort):
    study, initial = read_json(GUARDED), read_json(LOCK)
    if study["base_freeze"] != digest(LOCK):
        raise ValueError("The parent detector freeze changed.")
    selected = study["selected"]
    if not selected:
        write_json(ART / f"detector_guarded_{cohort}.json", dict(candidate=None, freeze_sha256=digest(GUARDED)))
        return
    floor = read_json(BASE)["cutoff"]
    rows, frames = load(cohort)
    baseline = evaluate(rows, {k: f["prior"] for k, f in frames.items()}, floor)
    result = score(rows, frames, selected["policy"], initial["heads"], floor)
    guard = compare(baseline, result)
    write_json(ART / f"detector_guarded_{cohort}.json", dict(baseline=baseline, candidate=result, guard=guard,
               freeze_sha256=digest(GUARDED)))
    print("Strict recall", cohort, short(baseline), short(result), guard, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test", "reserve"))
    stage = parser.parse_args().stage
    select() if stage == "select" else audit(stage)


if __name__ == "__main__":
    main()
