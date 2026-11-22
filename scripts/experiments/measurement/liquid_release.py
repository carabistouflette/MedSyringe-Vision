"""Check lightness on drawn liquid interiors and seal the appearance runtime."""
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from PIL import Image,ImageOps

from medsyringe.data import digest,read_json,write_json
from medsyringe.measurement.liquid_appearance import relative_opacity
from medsyringe.measurement.liquid_region import BACKBONE,CHECKPOINT,HEAD
from medsyringe.runtime.local_models import ImageWorker
from scripts.experiments.measurement.liquid_fit import distance

ART=Path("artifacts")
REPORT=ART/"liquid_appearance_validation.json"
RELEASE=ART/"liquid_appearance_release.json"


def interval(image,points):
    rgb=np.asarray(ImageOps.contain(image.convert("RGB"),(420,420)))
    h,w=rgb.shape[:2]
    nearby=distance(points,(h,w),w,h)<.027
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    warm=(hsv[...,0]>=5)&(hsv[...,0]<=39)&(hsv[...,1]>=45)&(hsv[...,2]>=25)
    mask=nearby&warm
    if mask.sum()<35:
        return None
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2Lab)
    return float(np.median(lab[...,0][mask]))


def validate(rows,frames,marked):
    evidence={}
    for row in rows:
        identifier=row["id"]
        measurement=frames[identifier]
        measured=measurement["lightness"]
        if not measured:
            evidence[identifier]=dict(status="abstained",reason=measurement["reason"])
            continue
        with Image.open(row["image"]) as image:
            truth=interval(image,marked[identifier]["points"])
        l,t,r,b=measurement["region"]
        center=np.mean(marked[identifier]["points"],axis=0)
        lies_on_liquid=l-.02<=center[0]<=r+.02 and t-.04<=center[1]<=b+.04
        evidence[identifier]=dict(status="measured",lstar=measured["lstar"],
                                  reference_lstar=round(truth,2) if truth is not None else None,
                                  absolute_difference=round(abs(truth-measured["lstar"]),2) if truth is not None else None,
                                  centerline_supported=bool(lies_on_liquid))
    accepted=[r for r in evidence.values() if r["status"]=="measured"]
    differences=[r["absolute_difference"] for r in accepted if r["absolute_difference"] is not None]
    return dict(images=len(rows),answered=len(accepted),reference_comparisons=len(differences),
                centerline_supported=sum(r["centerline_supported"] for r in accepted),
                median_absolute_lstar_difference=median(differences) if differences else None,frames=evidence)


def synthetic_opacity():
    study=[]
    reference=np.broadcast_to(np.linspace(52,239,128,dtype=np.uint8)[None,:,None],(48,128,3)).copy()
    linear=((reference.astype(float)/255+.055)/1.055)**2.4
    for opacity in (.15,.45,.75):
        attenuated=np.round(np.clip(1.055*((1-opacity)*linear+.03)**(1/2.4)-.055,0,1)*255).astype(np.uint8)
        measured=relative_opacity(attenuated,reference,matched_setup=True)
        study.append(dict(expected_opacity=opacity,**measured))
    return study


def main():
    rows=read_json(ART/"marking_detected_manifest.json")["files"]
    annotations=read_json(ART/"liquid_region_scribbles.json")["regions"]
    frames=read_json(ART/"liquid_detected_results.json")["frames"]
    old=read_json(ART/"liquid_color_baseline.json")["frames"]
    group_study=read_json(ART/"liquid_region_study.json")
    held=[r for r in rows if r["burst"] in group_study["holdout_groups"]]
    result=validate(rows,frames,annotations)
    prior=validate(rows,old,annotations)
    check=validate(held,frames,annotations)
    synthetic=synthetic_opacity()
    passed=(check["answered"]>=6 and check["centerline_supported"]==check["answered"] and
            (check["median_absolute_lstar_difference"] or 0.)<10 and
            all(x["value"] is not None and abs(x["value"]-x["expected_opacity"])<.015 for x in synthetic))
    write_json(REPORT,dict(passed=passed,all=result,old=prior,heldout=check,
                           synthetic_opacity=synthetic,paired_real_references=0,
                           scope="Single-session visual lightness from identified interior pixels; synthetic attenuation check."))
    print("Lightness",result["answered"],result["centerline_supported"],
          "old",prior["answered"],prior["centerline_supported"],
          "holdout",check["answered"],check["centerline_supported"],"passed",passed,flush=True)
    if not passed:
        return
    sam=read_json(ART/"sam3_release.json")
    worker=ImageWorker(Path("."),Path(sam["executable"]),"sam3",sam["options"])
    sample=next(r for r in rows if r["id"]=="IMG_3229")
    with Image.open(sample["image"]) as image:
        started=perf_counter()
        obs=worker.run(image,operation="liquid")
    predicted=frames[sample["id"]]["lightness"]
    if bool(obs["lightness"])!=bool(predicted) or (predicted and
       abs(obs["lightness"]["lstar"]-predicted["lstar"])>.2):
        raise ValueError("Serialized liquid inference does not match the batch evaluation.")
    report=read_json(REPORT)
    report["worker_smoke"]=dict(image=sample["id"],observation=obs,seconds=perf_counter()-started)
    write_json(REPORT,report)
    paths=("medsyringe/measurement/liquid_appearance.py","medsyringe/measurement/liquid_region.py","scripts/experiments/measurement/liquid_measure.py")
    assets=[BACKBONE/name for name in read_json(CHECKPOINT)["files"]]
    write_json(RELEASE,dict(id="visible-liquid-lightness",evidence=str(REPORT),evidence_sha256=digest(REPORT),
                           code={p:digest(p) for p in paths},models={str(p):digest(p) for p in [HEAD,*assets]}))
    print("Activated",RELEASE,flush=True)


if __name__=="__main__":
    main()
