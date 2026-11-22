"""Select additive verifiers on development regressions before a fresh audit."""
import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate, match
from medsyringe.detection.recall_guard import compare
from medsyringe.detection.verification import classifier_scores
from scripts.experiments.detector_training.optimize_detector import FINAL, counters
from scripts.experiments.transfer.rfdetr_compare import SELECTION, combine
from scripts.experiments.transfer.rfdetr_verifier import (ART, AUDITS, DECISION, apply, baseline_for, crop_examples,
                                     data_hashes, examples, extras, get_encoder, inputs,
                                     model_proposals, windows)

FROZEN = ART / "rfdetr_verifier_refined_selection.json"
DEVELOPMENT = ("calibration", "previous_zenodo", "previous_external")
FRESH = ("fresh", "pen_reserve", "syringe_reserve")
SEED = 8040


def protocol_inputs():
    # The older reserve and Commons data become development for this follow-up.
    return {**data_hashes(), str(DECISION): digest(DECISION), str(Path(__file__)): digest(__file__)}


def scene(cohort):
    rows = inputs(cohort)
    base = baseline_for(cohort, rows)
    rf = model_proposals(cohort, rows)
    floor = read_json(FINAL)["cutoff"]
    raw = combine(base, rf, floor, read_json(SELECTION)["locked"]["options"])
    extra_boxes = {k: [p["bbox"] for p in v] for k, v in extras(raw, floor).items()}
    return dict(rows=rows, base=base, hybrid=raw, boxes=extra_boxes, floor=floor)


def compare_at_cutoff(sample, predictions):
    accepted = {k: [p for p in value if p["score"] >= sample["floor"]]
                for k, value in predictions.items()}
    matches, used = match(sample["rows"], accepted, .5)
    misses = {r["id"]: {i for i in range(len(r["objects"])) if i not in used[r["id"]]}
              for r in sample["rows"]}
    return sum(target is None for _, _, target, _ in matches), misses


def kept_by(heads, sample, features):
    kept = None
    for h in heads:
        key = (h["encoder"], h["padding"])
        passed = apply(sample["hybrid"], features[key], h)
        ids = {id(p) for boxes in passed.values() for p in boxes if p.get("source") == "auxiliary"}
        kept = ids if kept is None else kept & ids
    return {k: [p for p in ps if p.get("source") != "auxiliary" or id(p) in kept]
            for k, ps in sample["hybrid"].items()}


def settings(heads, samples, vectors, checks, pen_scores):
    penalty = 0
    for cohort, sample in samples.items():
        predictions = kept_by(heads, sample, vectors[cohort])
        fp, misses = compare_at_cutoff(sample, predictions)
        penalty += fp
        if any(misses[k] - checks[cohort][1][k] for k in misses):
            return None
    pen_rejected = np.logical_or.reduce([pen_scores[h["token"]] < h["cutoff"] for h in heads]).sum()
    return int(pen_rejected), penalty


def select():
    samples = {k: scene(k) for k in DEVELOPMENT}
    checks = {k:compare_at_cutoff(s,s["hybrid"]) for k,s in samples.items()}
    positives, negatives = examples()
    pen_rows = inputs("pen_validation")
    vectors = defaultdict(dict)
    pen_scores, tested, finalists = {}, [], []
    for name, padding in product(("mobile","dino"),(0.,.18)):
        encoder = get_encoder(name)
        features, labels, weights = crop_examples(encoder, positives, negatives, padding)
        key = (name,padding)
        for k,s in samples.items():
            vectors[k][key] = windows(encoder,s["rows"],s["boxes"],padding)
        pen_inputs = windows(encoder,pen_rows,{r["id"]: r["pens"] for r in pen_rows},padding)
        for strength in (1.,10.):
            estimator = LogisticRegression(C=strength,random_state=SEED)
            estimator.fit(features,labels,sample_weight=weights)
            token = f"{name}/{padding}/{strength}"
            prototype = dict(encoder=name,padding=padding,token=token,C=strength,
                             weights=estimator.coef_[0].tolist(),bias=float(estimator.intercept_[0]))
            scores = lambda vector: classifier_scores(vector,prototype["weights"],prototype["bias"])
            pens = np.concatenate([scores(v) for v in pen_inputs.values()])
            pen_scores[token] = pens
            values = sorted(set(pens.tolist()) | {float(x) for vs in vectors.values()
                            for v in vs[key].values() for x in scores(v)})
            possible = []
            for threshold in (values[0]-1,*[(a+b)/2 for a,b in zip(values,values[1:])],values[-1]+1):
                heads = [{**prototype,"cutoff":threshold}]
                metrics = settings(heads,samples,vectors,checks,pen_scores)
                if metrics is not None:
                    possible.append((heads, metrics))
            top = min(possible,key=lambda pair: (pair[1][1],-pair[1][0]))
            finalists.append(top[0][0])
            tested.append((top[0],top[1]))
    # The joint pass is conservative. A box must clear both heads.
    for left,right in product([h for h in finalists if h["encoder"]=="mobile"],
                              [h for h in finalists if h["encoder"]=="dino"]):
        heads=[left,right]
        metrics=settings(heads,samples,vectors,checks,pen_scores)
        if metrics is not None:
            tested.append((heads,metrics))
    chosen, metrics=min(tested,key=lambda item:(item[1][1],-item[1][0],len(item[0])))
    settings_lock = dict(heads=chosen,development_fps=metrics[1],pen_calibration_rejections=metrics[0],
                         protocol=protocol_inputs(),development=list(DEVELOPMENT),
                         candidates=[dict(heads=[{k:v for k,v in h.items() if k!="weights"} for h in pair],
                                          pen_calibration_rejections=score[0],development_fps=score[1])
                                     for pair,score in tested])
    write_json(FROZEN,settings_lock)
    print("Frozen head(s)",[(h["encoder"],h["padding"],h["C"],h["cutoff"]) for h in chosen],
          "Development false positives",metrics[1],"pen crops rejected",metrics[0],flush=True)


def audit(cohort):
    record=read_json(FROZEN)
    if record["protocol"]!=protocol_inputs():
        raise ValueError("Frozen model inputs changed")
    s=scene(cohort)
    features={}
    for h in record["heads"]:
        key = h["encoder"],h["padding"]
        features[key]=windows(get_encoder(h["encoder"]),s["rows"],s["boxes"],h["padding"])
    predictions=kept_by(record["heads"],s,features)
    p={k:evaluate(s["rows"],value,s["floor"]) for k,value in
       (("baseline",s["base"]),("hybrid",s["hybrid"]),("verified",predictions))}
    summary=dict(**p,guard=compare(p["baseline"],p["verified"]),
                 extra_guard=compare(p["hybrid"],p["verified"]),protocol_sha256=digest(FROZEN))
    write_json(ART/f"rfdetr_refined_{cohort}.json",summary)
    print(cohort,*(counters(p[k]) for k in p),flush=True)


def summarize():
    runs={}
    for cohort in AUDITS:
        item=read_json(ART/f"rfdetr_refined_{cohort}.json")
        if item["protocol_sha256"]!=digest(FROZEN):
            raise ValueError("Refined audit came from a different protocol")
        runs[cohort]=dict(baseline=counters(item["baseline"]),hybrid=counters(item["hybrid"]),
                          verified=counters(item["verified"]),guard=item["guard"],
                          extra_guard=item["extra_guard"])
    gates=dict(preserved_all_matches=all(a["guard"]["passed"] and a["extra_guard"]["passed"]
                                       for a in runs.values()),
               no_extra_false_positives=all(a["verified"]["fp"]<=a["baseline"]["fp"]
                                            for a in runs.values()),
               novel_positive_recovery=any(runs[c]["verified"]["tp"]>runs[c]["baseline"]["tp"]
                                           for c in ("fresh","syringe_reserve")))
    write_json(ART/"rfdetr_refined_release_decision.json",dict(
        eligible_for_release=all(gates.values()),**gates,cohorts=runs,
        protocol_sha256=digest(FROZEN),independent_cohorts=list(FRESH)))
    print("Refined gate",gates,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["select","audit","summarize"])
    parser.add_argument("--cohort",choices=AUDITS)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=="select":
        select()
    elif args.stage=="summarize":
        summarize()
    elif args.cohort:
        audit(args.cohort)
    else:
        parser.error("A cohort is required for audit")


if __name__=="__main__":
    main()
