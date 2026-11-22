"""Check broader visual exemplars after separating the historical dental-only trial."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from medsyringe.detection.additive import project
from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.detection.detectors import box_iou, model_hashes, suppress
from medsyringe.evaluate import evaluate
from scripts.experiments.detector_audits.detection_exemplar_comparison import DENTAL, INTERNET, ExemplarDetector, owl_box
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, inputs
from scripts.experiments.public_data.detection_public_benchmark import order
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source

OUTPUT=ART/"visual_exemplar_multidomain_results.json"
FORMS=("closest-anchor","multiple-anchors")


def fitting():
    selected=[]
    for path,role,domain in ((INTERNET,"openimages-fit","ordinary"),(DENTAL,"fit","dental")):
        rows=[r for r in read_json(path)["rows"] if r["split"]==role]
        for name,key in (("target","objects"),("counterexample","lookalikes")):
            examples=[r for r in rows if r[key] and (name=="target" or not r["objects"])]
            groups=set()
            for row in sorted(examples,key=lambda r:order(r["id"])):
                if row["group"] in groups:
                    continue
                groups.add(row["group"])
                selected.append(dict(row=row,box=row[key][0]["bbox"],side=name,domain=domain))
                if len(groups)==6:
                    break
    development=[{**r,"split":"validation"} for scope in DEVELOPMENT for r in inputs(scope)]
    photos=[{**r["row"],"split":"fit"} for r in selected]
    if leakage([*photos,*development]):
        raise ValueError("A visual exemplar crosses protected acquisition groups.")
    return selected


class MultipleAnchors(ExemplarDetector):
    @torch.inference_mode()
    def query(self,picture,box):
        patch=crop(picture,box)
        maps,regions=self.feature_map(patch)
        _,embedding=self.model.class_predictor(regions)
        predicted=owl_box(self.model.box_predictor(regions,maps)[0].cpu().numpy(),patch)
        value=np.asarray([box_iou(b,(0.,0.,1.,1.)) for b in predicted])
        order_=np.argsort(-value,kind="stable")[:5]
        weights=torch.as_tensor(value[order_],device=self.device,dtype=embedding.dtype).view(1,-1,1)
        return (embedding[:,order_,:]*weights).sum(dim=1,keepdim=True)/weights.sum().clamp(min=1e-5)


def path(mode,scope):
    return DATA/"proposals"/f"visual-exemplars-{mode}-{scope}.json"


def collect(mode,scope):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    templates=fitting()
    indexed={r["id"]:r for r in inputs(scope)}
    key=dict(code_sha256=digest(Path(__file__)),weights=model_hashes(Path("."),"owl"),device="xpu",
             references={k:v["sha256"] for k,v in indexed.items()},
             templates=[[x["row"]["id"],x["row"]["sha256"],x["side"],x["box"]] for x in templates])
    target=path(mode,scope)
    saved=read_json(target) if target.exists() else dict(key=key,frames={})
    if saved["key"]!=key:
        raise ValueError("The visual exemplar collection inputs changed.")
    model=None
    for identifier,row in indexed.items():
        if identifier in saved["frames"]:
            continue
        if model is None:
            cls=MultipleAnchors if mode=="multiple-anchors" else ExemplarDetector
            model=cls("xpu",templates)
        picture=source(row)
        start=perf_counter()
        full=model(picture)
        edge=min(picture.size)
        x,y=round((picture.width-edge)/2),round((picture.height-edge)/2)
        region=(x,y,x+edge,y+edge)
        crop_views=model(picture.crop(region)) if x or y else {}
        combined={}
        for name,items in full.items():
            mapped=[{**p,"bbox":box} for p in crop_views.get(name,[])
                    if (box:=project(p["bbox"],region,picture.size))]
            combined[f"{name}/full"]=items
            combined[f"{name}/wide"]=suppress([*items,*mapped],.43)
        saved["frames"][identifier]=dict(variants=combined,ms=(perf_counter()-start)*1000)
        write_json(target,saved)
        print("Broader exemplars",mode,scope,len(saved["frames"]),"/",len(indexed),flush=True)


def measure():
    if OUTPUT.exists():
        raise ValueError("The comparison already exists.")
    groups=Counter(f'{r["domain"]}/{r["side"]}' for r in fitting())
    findings={}
    for form in FORMS:
        findings[form]={}
        for scope in DEVELOPMENT:
            rows=inputs(scope)
            records=read_json(path(form,scope))["frames"]
            if set(records)!={r["id"] for r in rows}:
                raise ValueError("The development inference is incomplete.")
            variants=next(iter(records.values()))["variants"]
            findings[form][scope]=dict(median_ms=median(v["ms"] for v in records.values()),
                outcomes={name:short(evaluate(rows,{k:v["variants"][name] for k,v in records.items()},.20))
                          for name in variants})
    write_json(OUTPUT,dict(templates=dict(groups),findings=findings,
                           scope="Development only. Real public and dental positives and counterexamples are sealed fit crops."))
    print("Compared",len(FORMS),"visual template encoders using",dict(groups),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("collect","measure"))
    p.add_argument("--form",choices=FORMS)
    p.add_argument("--scope",choices=DEVELOPMENT)
    a=p.parse_args()
    if a.action=="measure":
        measure()
    elif a.form and a.scope:
        collect(a.form,a.scope)
    else:
        p.error("Select an encoder and a development cohort.")
