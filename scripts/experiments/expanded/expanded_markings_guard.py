"""Measure the effect of requiring corroboration for an untyped OCR capacity."""
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.measurement.marking_views import agreement
from medsyringe.measurement.markings_guard import confirm_untyped_capacity
from scripts.experiments.expanded.expanded_markings import ART, CACHE, LOCK, variants


def main():
    selected = read_json(LOCK)["selected"]
    config = dict(base_freeze_sha256=digest(LOCK), code_sha256=digest("medsyringe/measurement/markings_guard.py"),
                  policy="Apply to all capacities that lack a corroborating visible scale.")
    groups = {}
    for cohort in ("calibration", "test"):
        frames = read_json(CACHE / f"markings-{cohort}.json")["frames"]
        summary = dict(correct=0, wrong=0, abstained=0)
        removals = []
        for identifier, row in frames.items():
            options = row["readings"]
            result = variants(options)[selected]
            confirm = agreement(options["clahe"], options["enhanced"]) if "clahe" in options else {}
            guarded = confirm_untyped_capacity(result, confirm)
            capacity = guarded.get("capacity")
            target = float(row["group_scale"].split("-")[0])
            outcome = "abstained" if capacity is None else "correct" if capacity["value"] == target else "wrong"
            summary[outcome] += 1
            if result.get("capacity") and not capacity:
                removals.append(dict(id=identifier, suppressed=result["capacity"], group_capacity=target))
        groups[cohort] = dict(metrics=summary, suppressions=removals)
    write_json(ART / "markings_guard.json", dict(config=config, groups=groups,
               scope="Additional evidence rule evaluated after a discovered error. Test results are regression evidence."))
    print("Untyped capacity guard", {k: v["metrics"] for k, v in groups.items()}, flush=True)


if __name__ == "__main__":
    main()
