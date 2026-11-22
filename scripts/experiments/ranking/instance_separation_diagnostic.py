"""Trace the greedy addition failure and measure the independent proposal ceiling."""
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.experiments.ranking.instance_separation import ART, POLICY, RESULT, fuse, pair_scores, previous, sealed
from scripts.experiments.ranking.instance_separation_reserve import evidence
from scripts.experiments.crowds.native_candidate_verifier import compare, simple_metrics
from scripts.experiments.occlusion.occlusion_proposals import oracle

OUT = ART / "instance_separation_diagnostic.json"


def run():
    lock = sealed()
    result = read_json(RESULT)
    if result["policy"] != digest(POLICY) or any(digest(p) != h for p, h in result["inputs"].items()):
        raise ValueError("Frozen evidence changed.")
    fresh, base, raw, _ = evidence()
    coverage = {r["id"]: dict(objects=len(r["objects"]), released=len(base[r["id"]]),
                              proposal_matches=oracle(r["objects"], raw[r["id"]])["matched"]) for r in fresh}
    pairs = pair_scores(fresh, base, raw, lock["model"])
    before = simple_metrics(fresh, base)
    one_incumbent = {}
    for arm, probe in lock["decision"]["probes"].items():
        rule = dict(probe, minimum=1, distinct=1. if arm == "overlap" else probe["distinct"])
        if not any(t["rule"] == rule for t in lock["decision"]["trials"]):
            raise ValueError("The activation diagnostic must come from the development grid.")
        one_incumbent[arm] = compare(fresh, before, fuse(base, raw, pairs, rule))
    rows, old, bank, _ = previous()
    reference = next(r for r in rows if r["id"] == "refine-8670507")
    key = reference["id"]
    frozen = result["exposed"]["arms"]["probe-axis-identity"]["metrics"]["false_positives"]
    removed = [p["prediction"]["bbox"] for p in frozen if p["id"] == key and p["prediction"].get("source")
               == "instance separation"]
    original = bank[key]["seeds"]
    modified = [p for p in original if p["bbox"] not in removed]
    rule = lock["decision"]["probes"]["axis-identity"]
    counterfactual = fuse({key: old[key]}, {key: modified}, pair_scores([reference], old, {key: modified},
                                                                     lock["model"]), rule)
    measured = evaluate([reference], counterfactual, 0.)
    added = result["exposed"]["arms"]["probe-axis-identity"]["additions"][key]
    links = [dict(bbox=p["bbox"], detector_score=p["detector_score"], best_iou=max(
        box_iou(p["bbox"], o["bbox"]) for o in reference["objects"])) for p in added]
    write_json(OUT, dict(code=digest(__file__), evaluation=digest(RESULT), reserve=coverage,
                         reserve_one_incumbent_exploratory=one_incumbent,
                         exposed_probe_additions=links, removed_known_false_positive=removed,
                         exposed_without_known_error=measured,
                         scope="The removal is annotation-assisted and is not an automatic method."))
    print(coverage, (measured["tp"], measured["fp"], measured["fn"]), flush=True)


if __name__ == "__main__":
    run()
