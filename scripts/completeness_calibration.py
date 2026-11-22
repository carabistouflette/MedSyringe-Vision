"""Calibrate pure and blended completeness scores with identical conflict handling."""
import argparse
from itertools import product
from pathlib import Path

import cv2
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from scripts.complete_syringe_selection import (
    ART, POLICY as HEAD, identity_policy, pair_scores, predict, quality, sealed as source_head,
)
from scripts.native_candidate_bank import archive
from scripts.native_candidate_verifier import compare, simple_metrics
from scripts.occlusion_relocalization import prepare_reference

WORK = Path("data/completeness-calibration")
POLICY = ART / "completeness_calibration_policy.json"
RESULT = ART / "completeness_calibration_results.json"
MODES = ("pure", "blended")


def scored(crops):
    # The shared selector uses .30*score + .70*complete. Cancel its detector
    # term for pure ranking while retaining the original detector prefilter.
    return {
        "blended": crops,
        "pure": {k:[dict(p, complete=(p["complete"]-.30*p["score"])/.70,
                         original_complete=p["complete"]) for p in v] for k,v in crops.items()},
    }


def run_grid(rows, base, ranked, pairs):
    before = simple_metrics(rows, base)
    trials = []
    for mode, method, minimum, raw_floor, floor in product(
        MODES, ("overlap","axis-identity"), (0,1,2), (.12,.25,.40),
        (.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1.01)
    ):
        rule = dict(arm="body-greedy", method=method, minimum=minimum, detector_floor=raw_floor,
                    floor=floor, distinct=.65)
        trials.append(dict(mode=mode, rule=rule, **compare(rows,before,predict(base,ranked[mode],pairs,rule))))
    def ranking(t):
        return t["gain"],t["exact_count"],t["rule"]["floor"]
    return trials,{m:max((t for t in trials if t["safe"] and t["mode"]==m),key=ranking) for m in MODES}


def inputs():
    return {str(p):digest(p) for p in (HEAD,"scripts/complete_syringe_selection.py",
                                      "scripts/instance_separation.py")}


def freeze():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    cv2.setNumThreads(1)
    rows,base,sources,_ = prepare_reference()
    raw = archive(rows,"development")
    trials, finalists = run_grid(rows,base,scored(quality(rows,raw,source_head()["model"])),
                                 pair_scores(rows,base,raw,identity_policy()["model"]))
    winner = max((v for v in finalists.values() if v["gain"]>0),
                 key=lambda t:(t["gain"],t["exact_count"],t["mode"]=="blended"),default=None)
    probe = dict(arm="body-greedy",method="axis-identity",minimum=1,detector_floor=.25,floor=.25,distinct=.65)
    write_json(POLICY,dict(code=digest(__file__),inputs=inputs(),sources=sources,
                           trials=trials,finalists=finalists,winner=winner,probe=probe))
    print(finalists,winner,flush=True)


def sealed():
    state=read_json(POLICY)
    if state["code"]!=digest(__file__) or state["inputs"]!=inputs() or any(
        digest(p)!=h for p,h in state["sources"].items()
    ):
        raise ValueError("The frozen scoring protocol changed.")
    source_head()
    return state


def assess(rows,base,raw,lock):
    ranked=scored(quality(rows,raw,source_head()["model"]))
    pairs=pair_scores(rows,base,raw,identity_policy()["model"])
    before=simple_metrics(rows,base)
    arms={}
    for mode in MODES:
        for name,rule in ((mode,lock["finalists"][mode]["rule"]),(f"probe-{mode}",lock["probe"])):
            pred=predict(base,ranked[mode],pairs,rule)
            arms[name]=dict(delta=compare(rows,before,pred),metrics=evaluate(rows,pred,0.),
                            additions={k:p[len(base[k]):] for k,p in pred.items() if len(p)>len(base[k])})
    return dict(baseline=evaluate(rows,base,0.),arms=arms)


def score():
    from scripts.completeness_calibration_reserve import evidence
    from scripts.complete_syringe_reserve import evidence as previous
    from scripts.relocalization_reserve import evidence as exposed

    lock=sealed()
    cv2.setNumThreads(1)
    fresh,base,raw,provenance=evidence()
    past,old,cache,older=previous()
    crowd,crowd_base,frames,history=exposed()
    cohorts={
        "reserve":assess(fresh,base,raw,lock),
        "previous":assess(past,old,cache,lock),
        "exposed":assess(crowd,crowd_base,{k:f["seeds"] for k,f in frames.items()},lock),
    }
    winner=lock["winner"]["mode"] if lock["winner"] else None
    qualified=bool(winner and cohorts["reserve"]["arms"][winner]["delta"]["gain"]>0 and all(
        c["arms"][winner]["delta"]["safe"] for c in cohorts.values()
    ))
    write_json(RESULT,dict(policy=digest(POLICY), inputs={**provenance,**older,**history},
                           cohorts=cohorts,winner=winner,qualified=qualified))
    print(winner,qualified,{k:v["delta"] for k,v in cohorts["reserve"]["arms"].items()},flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("freeze","score"))
    {"freeze":freeze,"score":score}[cli.parse_args().stage]()
