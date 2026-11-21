"""Fit and select a coarse liquid region from burst-disjoint centerline annotations."""
from hashlib import sha256
from pathlib import Path
from statistics import median

import cv2
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression

from medsyringe.data import digest,read_json,write_json
from medsyringe.liquid_region import CHECKPOINT,HEAD,RegionFeatures
from scripts.sam3_study import configure_cpu

ART=Path("artifacts")
INPUT=ART/"liquid_region_scribbles.json"
CROPS=ART/"marking_detected_manifest.json"
CACHE=Path("data/ocr-v6/liquid-patch-features.npz")


def distance(points,shape,w,h):
    """Distance to the annotated segment in original-aspect image coordinates."""
    y,x=np.indices(shape,dtype=float)
    pixels=np.stack(((x+.5)/shape[1]*w,(y+.5)/shape[0]*h),-1)
    a,b=np.asarray(points)*[w,h]
    axis=b-a
    fraction=np.clip(((pixels-a)@axis)/max(float(axis@axis),1e-9),0,1)
    return np.linalg.norm(pixels-a-fraction[...,None]*axis,axis=-1)/max(w,h)


def prepare(rows):
    if CACHE.exists():
        cache=np.load(CACHE)
        if set(cache.files)=={r["id"] for r in rows}:
            return {name:cache[name] for name in cache.files}
    encoder=RegionFeatures("xpu")
    grids={}
    for row in rows:
        with Image.open(row["image"]) as image:
            grids[row["id"]]=encoder.extract(image)
    np.savez_compressed(CACHE,**grids)
    return grids


def sample(rows,marks,grids):
    features,labels,weights=[],[],[]
    for row in rows:
        identifier=row["id"]
        with Image.open(row["image"]) as image:
            ratio=distance(marks[identifier]["points"],(16,16),*image.size)
        certain=(ratio<=.043)|(ratio>=.13)
        X,correct=grids[identifier][certain],ratio[certain]<=.043
        counts=np.bincount(correct.astype(int),minlength=2)
        if not counts[1]:
            continue
        features.extend(X);labels.extend(correct)
        weights.extend(1./counts[correct.astype(int)])
    weights=np.asarray(weights)
    return np.asarray(features),np.asarray(labels),weights*len(weights)/weights.sum()


def evaluate(rows,marks,grids,weights,bias,threshold):
    evidence={}
    for row in rows:
        identifier=row["id"]
        with Image.open(row["image"]) as image:
            ratio=distance(marks[identifier]["points"],(96,96),*image.size)
        balanced=cv2.resize(grids[identifier]@weights+bias,(96,96))
        owned=balanced>=threshold
        if not owned.any():
            evidence[identifier]=dict(status="abstained")
            continue
        coverage=float((owned&(ratio<=.027)).sum()/max(1,(ratio<=.027).sum()))
        remote=float(np.mean(ratio[owned]>.115))
        evidence[identifier]=dict(status="measured",centerline_coverage=coverage,remote_fraction=remote,
                                  bad=coverage<.40 or remote>.25)
    answered=[r for r in evidence.values() if r["status"]=="measured"]
    return dict(images=len(rows),answered=len(answered),
                bad=sum(r["bad"] for r in answered),
                median_centerline=median(r["centerline_coverage"] for r in answered) if answered else None,
                evidence=evidence)


def main():
    configure_cpu()
    rows=read_json(CROPS)["files"]
    annotations=read_json(INPUT)["regions"]
    groups=sorted({r["burst"] for r in rows},key=lambda key:sha256(f"972:{key}".encode()).digest())
    fitgroups,validgroups,testgroups=groups[:11],groups[11:16],groups[16:]
    split=[[r for r in rows if r["burst"] in g] for g in (fitgroups,validgroups,testgroups)]
    train,validate,test=split
    grids=prepare(rows)
    X,y,sample_weight=sample(train,annotations,grids)
    choices=[]
    for regularization in (.2,1.,5.):
        head=LogisticRegression(C=regularization,random_state=972,max_iter=400)
        head.fit(X,y,sample_weight=sample_weight)
        weights,bias=head.coef_[0],float(head.intercept_[0])
        for threshold in (-.3,.2,.7,1.2,1.7,2.2):
            val=evaluate(validate,annotations,grids,weights,bias,threshold)
            choices.append((val,weights,bias,threshold,regularization))
    valid,weights,bias,threshold,regularization=max(
        choices,key=lambda item:(item[0]["answered"]-3*item[0]["bad"],-item[0]["bad"],
                                 item[0]["median_centerline"] or 0.))
    result=evaluate(test,annotations,grids,weights,bias,threshold)
    write_json(HEAD,dict(weights=weights.tolist(),bias=bias,cutoff=threshold,regularization=regularization,
                         data_sha256=digest(INPUT),backbone_sha256=digest(CHECKPOINT)))
    write_json(ART/"liquid_region_study.json",dict(
        scope="Single-session burst-disjoint centerline supervision; no pixel-accurate mask annotations.",
        training_groups=fitgroups,validation_groups=validgroups,holdout_groups=testgroups,
        validation=valid,heldout=result,head_sha256=digest(HEAD),
        choices=[dict(threshold=t,regularization=c,answered=v["answered"],bad=v["bad"])
                 for v,_,_,t,c in choices]))
    print("Validation",valid["answered"],valid["bad"],"holdout",result["answered"],result["bad"],flush=True)


if __name__=="__main__":
    main()
