"""Inspect localization misses without changing the frozen recrop decisions."""
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import match
from scripts.recrop_reserve import RELEASE
from scripts.recrop_verifier import (
    ART, ASSEMBLY, BUDGETS, NATIVE, POLICY, RELOCAL, RESULT,
    DinoEncoder, candidates, configure_cpu, cv2, rank_frames, read_bank, sealed,
)


def best_overlap(box, pool):
    overlap, candidate = max(((box_iou(box, p["bbox"]), p) for p in pool),
                             key=lambda pair: pair[0], default=(0., None))
    return dict(best_iou=overlap, candidate=candidate)


def run():
    state = sealed()
    result = read_json(RESULT)
    if result["policy"] != digest(POLICY) or any(digest(p) != v for p, v in result["evidence"].items()):
        raise ValueError("The reported evaluation evidence changed.")
    rows, bank = read_bank("archive")
    selected = [r for r in rows if r["id"] in {"crop-reserve-chemyx", "crop-reserve-9190822"}]
    base = {**read_json(RELOCAL / "released.json")["predictions"], **read_json(NATIVE / "released.json")["outputs"],
            **read_json(ASSEMBLY)["outputs"]}
    configure_cpu()
    cv2.setNumThreads(1)
    archive = rank_frames(selected, base, bank, state["heads"], DinoEncoder())
    archived = {r["id"]: {str(budget): [best_overlap(o["bbox"], pool) for o in r["objects"]]
                          for budget, pool in archive[r["id"]]["pools"].items()} for r in selected}
    fresh, bank = read_bank("reserve")
    baseline = read_json(RELEASE)["outputs"]
    _, matched = match(fresh, baseline, .50)
    misses = []
    for row in fresh:
        key = row["id"]
        for idx, obj in enumerate(row["objects"]):
            if idx in matched[key]:
                continue
            box, frame = obj["bbox"], bank[key]
            misses.append(dict(id=key, target_index=idx, baseline=best_overlap(box, baseline[key]),
                               first_pass=best_overlap(box, frame["seeds"]),
                               second_pass={str(b): best_overlap(box, candidates(frame, b)) for b in BUDGETS}))
    write_json(ART / "recrop_verifier_error_audit.json", dict(policy=digest(POLICY), result=digest(RESULT),
              code=digest(__file__), exposed_archive=archived, reserve_missed_references=misses,
              scope="Reference-assisted post-evaluation diagnosis. Evidence does not select any operating point."))
    for row in misses:
        print(row["id"], round(row["baseline"]["best_iou"], 3),
              {b: round(x["best_iou"], 3) for b, x in row["second_pass"].items()}, flush=True)


if __name__ == "__main__":
    run()
