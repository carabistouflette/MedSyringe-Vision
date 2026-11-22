"""Measure the true-versus-fragment ordering and additional archival regressions."""
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.experiments.ranking.complete_syringe_reserve import evidence
from scripts.experiments.ranking.complete_syringe_selection import ART, POLICY, RESULT, quality, sealed, select_boxes, summarize
from scripts.experiments.ranking.instance_separation import pair_scores, sealed as identity_policy
from scripts.experiments.occlusion.occlusion_proposals import oracle
from scripts.experiments.relocalization.relocalization_reserve import evidence as known
from scripts.experiments.relocalization.relocalization_selector import ASSEMBLY, NATIVE, RELOCAL, read_bank

OUT = ART / "complete_syringe_diagnostic.json"


def run():
    lock = sealed()
    result = read_json(RESULT)
    if result["policy"] != digest(POLICY) or any(digest(p) != h for p, h in result["inputs"].items()):
        raise ValueError("Frozen evaluation inputs changed.")
    rows, incumbent, bank, _ = known()
    row = next(r for r in rows if r["id"] == "refine-8670507")
    key = row["id"]
    candidates = quality([row], {key: bank[key]["seeds"]}, lock["model"])[key]
    pairs = pair_scores([row], incumbent, {key: bank[key]["seeds"]}, identity_policy()["model"])[key]
    targets = {}
    for i in (1, 5):
        target = row["objects"][i]["bbox"]
        p = max((p for p in candidates if box_iou(p["bbox"], target) >= .50),
                key=lambda p:p["score"], default=None)
        if p:
            targets[str(i)] = dict(candidate=p, iou=box_iou(p["bbox"],target),
                                   rank_score=.30*p["score"]+.70*p["complete"])
    incomplete = []
    for fp in result["exposed"]["arms"]["probe-body-joint"]["metrics"]["false_positives"]:
        if fp["id"] == key and fp["prediction"].get("source") == "complete syringe selection":
            p = fp["prediction"]
            incomplete.append(dict(box=p["bbox"], detector=p["detector_score"], complete=p["complete"],
                                   rank_score=.30*p["detector_score"]+.70*p["complete"],
                                   best_iou=fp["best_unmatched_iou"]))
    old, cache = read_bank("archive")
    previous = {**read_json(RELOCAL/"released.json")["predictions"],
                **read_json(NATIVE/"released.json")["outputs"], **read_json(ASSEMBLY)["outputs"]}
    archive = summarize(old,previous,{k:f["seeds"] for k,f in cache.items()},lock)
    reserve, base, raw, _ = evidence()
    ceilings = {r["id"]:dict(count=len(base[r["id"]]), objects=len(r["objects"]),
                             raw_matches=oracle(r["objects"],raw[r["id"]])["matched"]) for r in reserve}
    # Preserve the detector prefilter, but remove its contribution to ordering.
    # The transform cancels the detector term in the frozen 30/70 score.
    body_only = [dict(p, complete=(p["complete"] - .30*p["score"])/.70) for p in candidates]
    fixed = lock["decision"]["probes"]["body-joint"]
    pure = select_boxes(incumbent[key],body_only,pairs,fixed)
    pure_scores = evaluate([row],{key:pure},0.)
    write_json(OUT,dict(code=digest(__file__),result=digest(RESULT),prior_crowd=dict(
        count=len(incumbent[key]), reference_targets=targets,incomplete=incomplete,
        exposed_body_only_replay={k:pure_scores[k] for k in ("tp","fp","fn","exact_count")}),
        older_archive={k:v["delta"] for k,v in archive["arms"].items()}, reserve_proposal_limits=ceilings))
    print(targets,incomplete,ceilings,flush=True)


if __name__ == "__main__":
    run()
