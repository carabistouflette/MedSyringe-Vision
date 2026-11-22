"""Score field coverage and selective error on matched, reviewed syringe observations."""
import argparse
from pathlib import Path
from statistics import mean
from math import dist, isfinite

from medsyringe.data import digest, read_json, write_json
from medsyringe.field_status import number


def ratio(a, b):
    return a / b if b else None


def score(records, field):
    exposures = {name: [] for name in ("observable", "indeterminate", "unreviewed")}
    for record in records:
        reference = record.get("reference", {}).get(field, {})
        prediction = record.get("prediction", {}).get(field) or {}
        exposures[reference.get("status", "unreviewed")].append((reference, prediction))
    assessable = exposures["observable"]
    impossible = exposures["indeterminate"]
    def emitted(prediction):
        statuses = ("present", "empty") if field == "liquid_presence" else ("estimated", "interval only", "supported")
        return prediction.get("status") in statuses
    result = dict(observable=len(assessable), indeterminate=len(impossible),
                  unreviewed=len(exposures["unreviewed"]),
                  coverage=ratio(sum(emitted(p) for _, p in assessable), len(assessable)),
                  emission_on_indeterminate=ratio(sum(emitted(p) for _, p in impossible), len(impossible)))
    if field == "syringe_type":
        values = [p["value"] != r["value"] for r, p in assessable if emitted(p) and r.get("value") is not None]
        return {**result, "scored_types": len(values), "error_on_emitted": ratio(sum(values), len(values))}
    if field == "nominal_capacity":
        values = [p["ml"] != r["ml"] for r, p in assessable if emitted(p) and number(r.get("ml"))]
        return {**result, "scored_capacities": len(values), "error_on_emitted": ratio(sum(values), len(values))}
    if field == "liquid_presence":
        outcomes = {value: [(r, p) for r, p in assessable if r.get("value") is expected]
                    for value, expected in (("present", True), ("empty", False))}
        scores = {value: dict(
            references=len(pairs), correct=sum(p.get("status") == value for _, p in pairs),
            wrong=sum(emitted(p) and p.get("status") != value for _, p in pairs),
            unresolved=sum(not emitted(p) for _, p in pairs)) for value, pairs in outcomes.items()}
        accepted = sum(case["correct"] + case["wrong"] for case in scores.values())
        return {**result, "by_reference": scores,
                "error_on_emitted": ratio(sum(case["wrong"] for case in scores.values()), accepted)}
    if field == "liquid_colour":
        pairs = [(r["lab"], p["lab"]) for r, p in assessable if emitted(p)
                 and valid_lab(r.get("lab")) and valid_lab(p.get("lab"))]
        return {**result, "scored_colours": len(pairs),
                "mean_delta_e76": mean(dist(a, b) for a, b in pairs) if pairs else None,
                "lightness_mae_lstar": mean(abs(a[0] - b[0]) for a, b in pairs) if pairs else None}
    if field == "liquid_opacity":
        differences = [abs(p["value"] - r["value"]) for r, p in assessable if emitted(p)
                       and number(p.get("value")) and number(r.get("value"))]
        return {**result, "scored_relative_opacity": len(differences),
                "mean_absolute_relative_error": mean(differences) if differences else None}
    numeric, ranges, points_in_bracket, range_on_bracket = [], [], [], []
    for reference, prediction in assessable:
        exact = reference.get("ml")
        bracket = reference.get("interval_ml")
        point = prediction.get("ml")
        interval = prediction.get("interval_ml")
        if prediction.get("status") == "estimated" and number(point):
            if number(exact):
                numeric.append(abs(point - exact))
            if bracket:
                points_in_bracket.append(bracket[0] <= point <= bracket[1])
        if emitted(prediction) and interval:
            if number(exact):
                ranges.append(interval[0] <= exact <= interval[1])
            if bracket:
                range_on_bracket.append(max(bracket[0], interval[0]) <= min(bracket[1], interval[1]))
    intervals = [p["interval_ml"] for _, p in assessable if p.get("interval_ml") and emitted(p)]
    return {**result,
            "point_coverage": ratio(sum(p.get("status") == "estimated" for _, p in assessable), len(assessable)),
            "interval_coverage": ratio(len(intervals), len(assessable)),
            "interval_only_coverage": ratio(sum(p.get("status") == "interval only" for _, p in assessable), len(assessable)),
            "point_mae_ml": mean(numeric) if numeric else None, "scored_exact_points": len(numeric),
            "point_in_visual_bracket": ratio(sum(points_in_bracket), len(points_in_bracket)),
            "interval_contains_exact": ratio(sum(ranges), len(ranges)),
            "interval_overlaps_visual_bracket": ratio(sum(range_on_bracket), len(range_on_bracket)),
            "mean_interval_width_ml": mean(b - a for a, b in intervals) if intervals else None}


def valid_lab(value):
    return (isinstance(value, (tuple, list)) and len(value) == 3 and
            all(type(component) in (int, float) and isfinite(component) for component in value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=Path, help="JSON rows of paired reference and prediction fields.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = read_json(args.references)["rows"]
    write_json(args.output, dict(input_sha256=digest(args.references),
                                 **{field: score(rows, field) for field in (
                                     "syringe_type", "nominal_capacity", "liquid_presence",
                                     "liquid_volume", "liquid_colour", "liquid_opacity")}))
