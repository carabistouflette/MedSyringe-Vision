"""Require instance-level non-regression at the fixed IoU 0.50 operating point."""
from collections import Counter

REQUIRED_SETS = ("fresh", "fil_rouge", "historical", "previous_external")


def _misses(result):
    misses = result["false_negatives"]
    if len(misses) != result["fn"] or result["tp"] + result["fn"] != result["objects"]:
        raise ValueError("Inconsistent reference counts in recall evidence.")
    return Counter((item["id"], tuple(item["box"])) for item in misses)


def compare(baseline, candidate):
    for key in ("images", "objects"):
        if baseline[key] != candidate[key]:
            raise ValueError("Recall comparisons require the same reference set.")
    before, after = _misses(baseline), _misses(candidate)
    lost = [{"id": identifier, "box": box} for identifier, box in (after-before).elements()]
    return dict(passed=not lost, lost_matches=lost, recovered_matches=sum((before-after).values()),
                baseline_matches=baseline["tp"], candidate_matches=candidate["tp"])


def release_allowed(study):
    if not all(name in study for name in REQUIRED_SETS):
        return False
    checks = [compare(study[name]["baseline"], study[name]["candidate"]) for name in REQUIRED_SETS]
    fresh = study["fresh"]
    return (all(check["passed"] for check in checks) and
            fresh["candidate"]["fp"] < fresh["baseline"]["fp"])
