"""Test corroboration of weak proposals while preserving the released instances."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from medsyringe.additive import overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.recall_guard import compare
from scripts.detection_crop_semantics import DECISION, feature_cache, read_features, scores
from scripts.detection_hillclimb_experiment import ART, DEVELOPMENT, FLOOR, cache, inputs, loaded
from scripts.detection_hillclimb_rank import minimal
from scripts.detection_open_models import open_views, proposal_path

LOCK = ART/"detection_consensus_development.json"
MODELS = ("dino-base","owlv2")


def bound(box, others):
    return max((p["score"] for p in others if box_iou(box,p["bbox"])>=.42),default=0.)


def features(frame, text, joint, shape, open_models):
    semantic = scores(frame,text,None)
    combined = scores(frame,"joint",joint)
    shape_only = scores(frame,"dino",shape)
    measurements = []
    for i,p in enumerate(frame["candidates"]):
        box = p["bbox"]
        votes = {m:bound(box,values) for m,values in open_models.items()}
        floor = p.get("baseline_score",p["score"])
        measurements.append((p, floor, float(semantic[i]), float(combined[i]),float(shape_only[i]),votes))
    return measurements


def agree_with(second,mode):
    if mode=="none":
        return True
    dino = second["dino-base"]>=.27
    owl = second["owlv2"]>=.15
    return (dino and owl) if mode=="both-open" else (dino or owl)


def classify(frame, measures, policy):
    cutoff = frame["floor"]
    retained = []
    pending = []
    for p,prior,text,joint,shape,opinion in measures:
        if prior>=cutoff:
            rejected = (text<policy["reject"] and (not policy["guard_veto"] or shape<-.15))
            if not rejected:
                retained.append(p)
        elif p["score"]>=policy["prior"]:
            pending.append((p,text,joint,shape,opinion))
    retained = sorted(retained,key=lambda p:-p["score"])
    additions = []
    for p,text,joint,shape,opinion in pending:
        if joint<policy["joint"] or text<policy["text"] or (policy["shape"] and shape<.10):
            continue
        if not agree_with(opinion,policy["opinions"]):
            continue
        box=p["bbox"]
        if any(box_iou(box,q["bbox"])>=.42 or overlap_fraction(box,q["bbox"])>=.80 for q in (*retained,*additions)):
            continue
        additions.append(dict(bbox=box,score=cutoff,source="crop consensus",joint_margin=joint,
                              text_margin=text,shape_margin=shape,independent_support=opinion))
        if len(additions)>=policy["maximum"]:
            break
    return [*retained,*additions,*(p for p in frame["ranked"] if p["score"]<cutoff)]


def select():
    if LOCK.exists():
        raise ValueError("The consensus development policy has already been frozen.")
    stage = read_json(DECISION)
    joint = stage["heads"]["joint/2.0"]
    shape = stage["heads"]["dino/20.0"]
    cutoff = read_json(FLOOR)["cutoff"]
    data = {}
    for role in DEVELOPMENT:
        refs = inputs(role)
        base = loaded(role,"baseline")
        crops = read_features(role)
        auxiliary = {m:open_views(role,m)["wide"] for m in MODELS}
        evidence = {key:features(frame,"semantic-best",joint,shape,{m:v[key] for m,v in auxiliary.items()})
                    for key,frame in crops.items()}
        data[role]=(refs,base,evidence,minimal(refs,{k:f["ranked"] for k,f in base.items()},cutoff))
    start = perf_counter()
    rows = product((.26,.35,.47),(.35,.65,1.0),(-.008,.0,.008,.016,.022),(False,True),
                   ("none","one-open","both-open"),(-1e4,-.017,-.005),(False,True),(1,2))
    trials = []
    for prior,joint_value,text_value,dino,agreement,veto,two_veto,limit in rows:
        if veto == -1e4 and two_veto:
            continue
        settings = dict(prior=prior,joint=joint_value,text=text_value,shape=dino,opinions=agreement,
                        reject=veto,guard_veto=two_veto,maximum=limit)
        measured = {}
        for role,(refs,base,evidence,earlier) in data.items():
            fitted = {k:classify(frame,evidence[k],settings) for k,frame in base.items()}
            result = minimal(refs,fitted,cutoff)
            tp,fp,fn=(result[k] for k in ("tp","fp","fn"))
            stats = {"tp":tp,"fp":fp,"fn":fn,"f1":2*tp/max(2*tp+fp+fn,1),
                     "instance_gate":compare(earlier,result),"fp_delta":fp-earlier["fp"]}
            measured[role]=stats
        trials.append(dict(settings=settings,outcomes=measured))
    valid = [result for result in trials if all(x["instance_gate"]["passed"] and x["fp_delta"]<=0
                                               for x in result["outcomes"].values())]

    def objective(trial):
        external = trial["outcomes"]["public-development"]
        simulated = trial["outcomes"]["phlebotomy-development"]
        return (external["f1"],external["tp"],-external["fp"],simulated["f1"],
                trial["settings"]["opinions"]=="none",trial["settings"]["shape"],-trial["settings"]["maximum"])

    selected = max(valid,key=objective,default=None)
    inputs_used = [Path(__file__),DECISION,FLOOR,*(feature_cache(role) for role in DEVELOPMENT)]
    for role in DEVELOPMENT:
        inputs_used += [cache(role,"baseline"),*(proposal_path(role,key) for key in MODELS)]
    write_json(LOCK,dict(candidate=selected,seconds=perf_counter()-start,trial_count=len(trials),
                        valid_count=len(valid),trials=trials,heads=dict(joint=joint,shape=shape),
                        inputs={str(p):digest(p) for p in inputs_used}))
    print("Consensus variant",len(trials),"grid choices",len(valid),"safe",selected,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    select()
