"""Measure available replacements after evaluation without selecting a new rule."""
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from scripts.occlusion_proposals import oracle
from scripts.recrop_proposal_bank import candidates
from scripts.relocalization_reserve import evidence
from scripts.relocalization_selector import ART, EXPOSED, POLICY, RESULT, rank, read_bank, sealed

OUT = ART / "relocalization_selector_diagnostic.json"


def inspect(rows, baseline, proposals, lock):
    chosen = lock["selection"]["selected"]
    if chosen is None:
        return []
    rule = chosen["rule"]
    ranked = rank(rows, baseline, proposals, lock["heads"])
    descriptions = []
    for row in rows:
        key = row["id"]
        original = baseline[key]
        if len(original) != 1 or len(row["objects"]) != 1:
            continue
        target = row["objects"][0]["bbox"]
        original_iou = box_iou(target, original[0]["bbox"])
        if original_iou >= .75:
            continue
        choices = ranked[key][rule["budget"]]
        best = max(choices, key=lambda p: box_iou(target, p["bbox"]), default=None)
        highest = max(choices, key=lambda p: p["scores"][rule["arm"]], default=None)
        descriptions.append(dict(id=key, incumbent_iou=original_iou,
            best_available=dict(iou=box_iou(target, best["bbox"]), score=best["scores"][rule["arm"]]) if best else None,
            highest_scored=dict(iou=box_iou(target, highest["bbox"]), score=highest["scores"][rule["arm"]])
            if highest else None))
    return descriptions


def run():
    lock = sealed()
    result = read_json(RESULT)
    if result["policy"] != digest(POLICY) or any(digest(p) != sha for p, sha in result["evaluated_inputs"].items()):
        raise ValueError("The reported evaluation evidence changed.")
    fresh, previous, proposals, _ = evidence()
    exposed, bank = read_bank("reserve")
    crowded = {}
    for row in fresh:
        if len(row["objects"]) < 2:
            continue
        key = row["id"]
        frame = proposals[key]
        pools = {"released": previous[key], "first_pass": frame["seeds"],
                 "first_pass_and_recrops": [*frame["seeds"], *candidates(frame, 6)]}
        crowded[key] = dict(targets=len(row["objects"]), released_count=len(previous[key]),
                            oracle={name: oracle(row["objects"], items)["matched"] for name, items in pools.items()})
    write_json(OUT, dict(result=digest(RESULT), code=digest(__file__),
        fresh_single_object=inspect(fresh, previous, proposals, lock),
        exposed_single_object=inspect(exposed, read_json(EXPOSED)["outputs"], bank, lock),
        fresh_multi_object=crowded))


if __name__ == "__main__":
    run()
