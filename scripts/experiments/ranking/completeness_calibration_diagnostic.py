"""Audit isolated misses and score the older release archive."""
import cv2

from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import match
from scripts.experiments.ranking.completeness_calibration import ART, POLICY, RESULT, assess, quality, sealed, source_head
from scripts.experiments.ranking.completeness_calibration_reserve import evidence
from scripts.experiments.occlusion.occlusion_proposals import oracle
from scripts.experiments.relocalization.relocalization_selector import ASSEMBLY, NATIVE, RELOCAL, read_bank

OUT=ART/"completeness_calibration_diagnostic.json"


def run():
    lock=sealed()
    result=read_json(RESULT)
    if result["policy"]!=digest(POLICY) or any(digest(p)!=h for p,h in result["inputs"].items()):
        raise ValueError("Frozen evaluation changed.")
    cv2.setNumThreads(1)
    rows,base,bank,_=evidence()
    _,matches=match(rows,base,.50)
    candidates=quality(rows,bank,source_head()["model"])
    misses=[]
    for row in rows:
        key=row["id"]
        for i,obj in enumerate(row["objects"]):
            if i in matches[key]:
                continue
            eligible=[p for p in candidates[key] if box_iou(p["bbox"],obj["bbox"])>=.50]
            best=max(eligible,key=lambda p:p["complete"],default=None)
            misses.append(dict(id=key,index=i,raw_oracle=oracle(row["objects"],bank[key])["matched"],
                              best_prefiltered=best))
    older,frames=read_bank("archive")
    prior={**read_json(RELOCAL/"released.json")["predictions"],
           **read_json(NATIVE/"released.json")["outputs"],**read_json(ASSEMBLY)["outputs"]}
    report=assess(older,prior,{k:f["seeds"] for k,f in frames.items()},lock)
    write_json(OUT,dict(code=digest(__file__),result=digest(RESULT),misses=misses,archive=report))
    print(misses,{k:v["delta"] for k,v in report["arms"].items()},flush=True)


if __name__=="__main__":
    run()
