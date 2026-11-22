"""Measure which existing review candidates to show first, without changing detections."""
import argparse
from pathlib import Path

from medsyringe.detection.additive import overlap_fraction
from medsyringe.data import digest,read_json,write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_consensus_audit import FREEZE as AUDIT_FREEZE
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DEVELOPMENT, cache, inputs, loaded
from scripts.experiments.detector_training.detection_hillclimb_rank import minimal
from scripts.experiments.detector_audits.detection_precision_confirmation import cohort

SELECTED = ART/"review_priority_selection.json"
RESULTS = ART/"review_priority_measured.json"
STRATEGIES = ("existing","score","distinct","score-and-distinct")


def prioritize(pending, accepted, strategy):
    if strategy=="existing":
        return list(range(len(pending)))
    ordered = []
    remaining = set(range(len(pending)))
    while remaining:
        def score(index):
            proposal = pending[index]
            quality = (proposal["score"] if "score" in strategy else 1.0-index*.09)
            if "distinct" in strategy:
                compare_with = [*accepted,*(pending[i] for i in ordered)]
                similar = max((max(box_iou(proposal["bbox"],other["bbox"]),
                                   .7*overlap_fraction(proposal["bbox"],other["bbox"])) for other in compare_with),
                              default=0.0)
                quality -= .50*similar
            return quality,-index
        best = max(remaining,key=score)
        remaining.remove(best)
        ordered.append(best)
    return ordered


def stage(data, base, strategy, limit):
    gathered = {}
    used = 0
    for row in data:
        key = row["id"]
        frame = base[key]
        accepted = [p for p in frame["ranked"] if p["score"]>=frame["floor"]]
        pending = [p for p in frame["ranked"] if p["score"]<frame["floor"]][:5]
        selected = prioritize(pending,accepted,strategy)[:limit]
        used += len(selected)
        gathered[key] = [{**p,"score":1.0} for p in (*accepted,*(pending[i] for i in selected))]
    score = minimal(data,gathered,1.)
    return score,used


def score(data, baseline, name):
    old,_ = stage(data,baseline,"existing",2)
    later,count = stage(data,baseline,name,2)
    head = compare(old,later)
    at_five = compare(stage(data,baseline,"existing",5)[0],stage(data,baseline,name,5)[0])
    automatic, _ = stage(data,baseline,"existing",0)
    return dict(top_two_old=old["tp"],top_two_candidate=later["tp"],top_two_guard=head,
                reachable_at_five=at_five["candidate_matches"],full_list_guard=at_five,
                accepted=automatic["tp"],references=old["objects"],suggestions_shown_first=count,
                images=len(data))


def select():
    if SELECTED.exists() or RESULTS.exists():
        raise ValueError("Existing review priority decision already recorded.")
    dev = {name:(inputs(name),loaded(name,"baseline")) for name in DEVELOPMENT}
    matrix = {strategy:{name:score(rows,baseline,strategy) for name,(rows,baseline) in dev.items()} for strategy in STRATEGIES}
    eligible = {k:v for k,v in matrix.items() if all(r["top_two_guard"]["passed"] and r["full_list_guard"]["passed"]
                                                    for r in v.values())}
    ranking = lambda item: (item[1]["public-development"]["top_two_candidate"],
                            item[1]["phlebotomy-development"]["top_two_candidate"], item[0]=="existing")
    chosen = max(eligible.items(),key=ranking)[0]
    inputs_used = [Path(__file__),AUDIT_FREEZE,*(cache(role,"baseline") for role in DEVELOPMENT)]
    write_json(SELECTED,dict(chosen=chosen,development=matrix,inputs={str(p):digest(p) for p in inputs_used},
                             scope="Ranks the same five exposed suggestions. Top-two view only. Automatic boxes unchanged."))
    print("Review strategy frozen",chosen,flush=True)


def audit():
    selected = read_json(SELECTED)
    if any(digest(p)!=h for p,h in selected["inputs"].items()):
        raise ValueError("Review policy or model predictions changed after selection.")
    if RESULTS.exists():
        raise ValueError("Review evaluation already recorded.")
    results = {}
    for group in ("fresh","commons-regression","previous-reserve","openimages","lookalikes"):
        refs,before = cohort(group)
        results[group] = score(refs,before,selected["chosen"])
    passed = all(row["top_two_guard"]["passed"] and row["full_list_guard"]["passed"] for row in results.values())
    passed = passed and results["fresh"]["top_two_candidate"] > results["fresh"]["top_two_old"]
    write_json(RESULTS,dict(passed=passed,policy=selected["chosen"],selection_sha256=digest(SELECTED),
                           results=results,meaning="Potential reference coverage from direct candidate inspection. "
                                                   "Operator accuracy and labor time remain unmeasured."))
    print(selected["chosen"],results,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("select","audit"))
    (select if parser.parse_args().action=="select" else audit)()
