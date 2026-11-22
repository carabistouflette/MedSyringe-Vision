"""Evaluate fixed SAM tile recovery policies against the released count."""

from medsyringe.detection.additive import duplicates, overlapping_tiles, project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.experiments.occlusion.occlusion_detector import ART, DATA, FRESH, frozen_frames, raster, target_ids
from scripts.experiments.occlusion.occlusion_proposals import oracle
from scripts.experiments.parts.part_verifier_study import (
    FREEZE, DinoEncoder, Segmenter, accept, configure_cpu, describe, infer, load,
)

CACHE = DATA / "sam-tiled-predictions.json"
OUTPUT = ART / "occlusion_sam_tiling_results.json"
POLICY = ART / "occlusion_sam_tiling_policy.json"


def handoff(frame, released, rule):
    combined = list(released)
    if len(released) > 1:
        return combined
    for candidate in sorted(frame["tiles"] if rule != "full" else frame["full"], key=lambda p: -p["score"]):
        box = candidate["bbox"]
        if candidate["score"] < .50 or any(duplicates(box, p["bbox"]) for p in combined):
            continue
        if rule == "agreement":
            match_full = any(box_iou(box, other["bbox"]) >= .50 for other in frame["full"])
            two_tiles = any(other["window"] != candidate["window"] and box_iou(box, other["bbox"]) >= .55
                            for other in frame["tiles"])
            if not (match_full or two_tiles):
                continue
        if rule == "anatomy" and not candidate["part_head_accepts"]:
            continue
        combined.append(dict(bbox=box, score=min(candidate["score"], .01)))
    return combined


def observations(image, segmenter, encoder, head):
    def proposals(view):
        return describe(view, infer(segmenter, view), encoder)

    tiles = []
    for window in overlapping_tiles(image.size, .66):
        for candidate in proposals(image.crop(window)):
            if bbox := project(candidate["bbox"], window, image.size):
                tiles.append(dict(bbox=bbox, score=candidate["score"], window=list(window),
                                  part_head_accepts=accept(candidate, head, "parts")))
    return dict(full=[dict(bbox=p["bbox"], score=p["score"]) for p in proposals(image)], tiles=tiles)


def run():
    lock = dict(part_model=digest(FREEZE), code=digest(__file__), reserve=digest(FRESH),
                minimum_score=.50, tile_fraction=.66, max_existing_to_route=1,
                rules=["full", "tile", "agreement", "anatomy"],
                replay_gate="Improve independent reserve matches while preserving all released matches "
                            "and introducing zero false positives in every evaluated cohort.")
    if not POLICY.exists():
        write_json(POLICY, lock)
    elif lock != read_json(POLICY):
        raise ValueError("Previously frozen SAM tile policy changed.")
    head = load()["parts"]
    cohorts, released = frozen_frames()
    cache = read_json(CACHE) if CACHE.exists() else dict(policy=digest(POLICY), frames={})
    if cache["policy"] != digest(POLICY):
        raise ValueError("Cached observations correspond to another policy version.")
    configure_cpu()
    model, appearance = None, None
    summary = {}
    for role, rows in cohorts.items():
        routed = [r for r in rows if len(released[r["id"]]) <= 1]
        for index, row in enumerate(routed, 1):
            key = row["id"]
            if key not in cache["frames"]:
                if model is None:
                    model, appearance = Segmenter(756, "xpu"), DinoEncoder()
                cache["frames"][key] = dict(sha256=row["sha256"],
                                            **observations(raster(row), model, appearance, head))
                write_json(CACHE, cache)
                print(role, index, "/", len(routed), key, flush=True)
            if cache["frames"][key]["sha256"] != row["sha256"]:
                raise ValueError(f"Changed raster for {key}")
        baseline = {r["id"]: released[r["id"]] for r in rows}
        report = evaluate(rows, baseline, 0.)
        target_set = target_ids(rows, baseline)
        arms = {}
        for policy in lock["rules"]:
            combined = {key: handoff(cache["frames"].get(key, {}), predictions, policy)
                        for key, predictions in baseline.items()}
            result = evaluate(rows, combined, 0.)
            arms[policy] = dict(metrics=result, added_false_positives=result["fp"]-report["fp"],
                                lost_targets=sorted(target_set-target_ids(rows, combined)))
        summary[role] = dict(released=report, variants=arms, routed=len(routed))
        summary[role]["oracle_at_half_iou"] = {
            r["id"]: {mode: oracle(r["objects"], cache["frames"][r["id"]][mode])["matched"]
                      for mode in ("full", "tiles")}
            for r in routed}
    prospective = summary["new-public-reserve"]
    passing = {rule: prospective["variants"][rule]["metrics"]["tp"] > prospective["released"]["tp"]
               and all(item["variants"][rule]["added_false_positives"] <= 0
                       and not item["variants"][rule]["lost_targets"] for item in summary.values())
               for rule in lock["rules"]}
    write_json(OUTPUT, dict(policy_sha256=digest(POLICY), cohorts=summary, meets_release_gate=passing))
    print(passing, flush=True)


if __name__ == "__main__":
    run()
