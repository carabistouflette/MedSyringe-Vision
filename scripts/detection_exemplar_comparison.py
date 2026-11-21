"""Compare target and counterexample prompts using two local detectors on development groups."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from medsyringe.additive import project
from medsyringe.attributes import crop
from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.detectors import box_iou, model_hashes, suppress
from medsyringe.evaluate import evaluate
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, FLOOR, inputs, loaded
from scripts.detection_dental_transfer_data import MANIFEST as DENTAL
from scripts.expanded_detection_study import short
from scripts.precision_study import source

INTERNET = ART/"openimages_additional_manifest.json"
MM_MODEL = Path("data/models/mm-grounding-dino-tiny")
MM_REVISION = "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e"
MODEL_LOCK = ART/"mm_grounder_backbone.json"
RESULT = ART/"detector_exemplar_development.json"
BACKENDS = ("image-prompts","openmmlab")
TERMS = ("syringe", "hypodermic syringe", "disposable plastic syringe", "dental cartridge syringe",
         "metal aspirating dental syringe", "syringe partly held in a hand", "syringe barrel and plunger")
NEGATIVE = ("vial", "pen", "pipette", "infusion pump", "dental mirror", "catheter", "loose hypodermic needle")


def fit_references():
    from hashlib import sha256

    batches = {}
    for manifest,label in ((INTERNET,"ordinary"),(DENTAL,"dental")):
        for row in read_json(manifest)["rows"]:
            if row["split"]!="fit":
                continue
            if row["objects"]:
                boxes,side = row["objects"],"target"
            elif row.get("lookalikes"):
                boxes,side = row["lookalikes"],"counterexample"
            else:
                continue
            batches.setdefault((label,side),[]).append((row,boxes[0]["bbox"]))
    collected = []
    for (domain,side),examples in batches.items():
        picked = sorted(examples,key=lambda p:sha256(("owl-exemplar-v1/"+p[0]["group"]).encode()).digest())[:6]
        collected.extend(dict(row=row,box=box,domain=domain,side=side) for row,box in picked)
    protected = [{**r,"split":"reserved"} for scope in DEVELOPMENT for r in inputs(scope)]
    if leakage([*(item["row"] for item in collected),*protected]):
        raise ValueError("A prompt exemplar overlaps development data.")
    return collected


def grounder_lock():
    files = ("README.md","preprocessor_config.json","config.json","vocab.txt",
             "tokenizer_config.json","special_tokens_map.json","model.safetensors")
    current = {file:digest(MM_MODEL/file) for file in files}
    if not MODEL_LOCK.exists():
        write_json(MODEL_LOCK,dict(repo="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                                   revision=MM_REVISION, license="Apache-2.0", files=current))
    data = read_json(MODEL_LOCK)
    if data["files"] != current or data["revision"] != MM_REVISION:
        raise ValueError("The OpenMMLab checkpoint differs from its sealed release.")
    return data


def owl_box(raw, image):
    cx,cy,w,h = raw.T
    square = max(image.size)
    xyxy = np.column_stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2))
    return np.clip(xyxy*(square/np.array([image.width,image.height,image.width,image.height])),0.,1.)


class ExemplarDetector:
    def __init__(self,device,templates):
        self.device=device
        path=Path("data/models/owl")
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(
            path,local_files_only=True,use_safetensors=True).eval().to(device)
        self.templates = templates
        self.embeddings = torch.cat([self.query(source(r["row"]),r["box"]) for r in templates],dim=1)
        self.target = [i for i,row in enumerate(templates) if row["side"]=="target"]
        self.negative = [i for i,row in enumerate(templates) if row["side"]=="counterexample"]
        self.dental = [i for i in self.target if templates[i]["domain"]=="dental"]

    def feature_map(self, image):
        processed = self.processor.image_processor(images=image,return_tensors="pt")["pixel_values"].to(self.device)
        maps = self.model.image_embedder(pixel_values=processed)[0]
        return maps,maps.flatten(1,2)

    @torch.inference_mode()
    def query(self, picture, box):
        patch = crop(picture,box)
        maps,features = self.feature_map(patch)
        _,embeddings = self.model.class_predictor(features)
        boxes = owl_box(self.model.box_predictor(features,maps)[0].cpu().numpy(),patch)
        overlap = np.array([box_iou(box,(0.,0.,1.,1.)) for box in boxes])
        index = int(overlap.argmax())
        return embeddings[:,index:index+1,:]

    @torch.inference_mode()
    def __call__(self,picture):
        maps,features = self.feature_map(picture)
        scores = self.model.class_predictor(features,self.embeddings)[0][0].sigmoid().cpu().numpy()
        objectness = self.model.objectness_predictor(features)[0].sigmoid().cpu().numpy().reshape(-1)
        boxes = owl_box(self.model.box_predictor(features,maps)[0].cpu().numpy(),picture)
        options = {}
        target = scores[:,self.target]
        counter = scores[:,self.negative].max(axis=1)
        alternatives = (target.max(axis=1),np.sort(target,axis=1)[:,-3:].mean(axis=1),scores[:,self.dental].max(axis=1))
        for label,values in zip(("peak","multiple","dental"),alternatives,strict=True):
            options[label]=predictions(boxes,values)
            options[label+"-evidence"]=predictions(boxes,values*np.sqrt(objectness)*(1-counter))
        return options


class OpenMMLabDetector:
    def __init__(self,device):
        self.device=device
        self.processor=AutoProcessor.from_pretrained(MM_MODEL,local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(
            MM_MODEL,local_files_only=True,use_safetensors=True).eval().to(device)
        self.processor.image_processor.size = dict(shortest_edge=640,longest_edge=1024)

    @torch.inference_mode()
    def run(self,image,terms):
        request=self.processor(images=image,text=". ".join(terms)+".",return_tensors="pt").to(self.device)
        detected=self.model(**request)
        frame=self.processor.post_process_grounded_object_detection(
            detected,request.input_ids,threshold=.035,target_sizes=[(image.height,image.width)])[0]
        labels=frame.get("text_labels",frame.get("labels"))
        if not all(isinstance(item,str) for item in labels):
            raise ValueError("Text-aware postprocessing did not return grounded labels.")
        identified = [i for i,label in enumerate(labels) if "syringe" in label.casefold()]
        boxes = frame["boxes"].cpu().numpy()[identified]
        size = np.array([image.width,image.height,image.width,image.height])
        return predictions(np.clip(boxes/size,0.,1.),frame["scores"].cpu().numpy()[identified])

    def __call__(self,image):
        return dict(title=self.run(image,TERMS[:1]),phrases=self.run(image,TERMS),
                    context=self.run(image,(*TERMS,*NEGATIVE)))


def predictions(boxes,scores):
    if len(scores)==0:
        return []
    active = np.flatnonzero((scores>=.009) & (boxes[:,2]-boxes[:,0]>=.005) & (boxes[:,3]-boxes[:,1]>=.005))
    active = sorted(active,key=lambda key:-float(scores[key]))[:100]
    return suppress([dict(bbox=boxes[i].tolist(),score=float(scores[i])) for i in active],.45)[:32]


def destination(backend,scope):
    return DATA/"proposals"/f"exemplar-{backend}-{scope}.json"


def collect(backend,scope,device):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = inputs(scope)
    templates = fit_references() if backend=="image-prompts" else []
    lock = model_hashes(Path("."),"owl") if templates else grounder_lock()["files"]
    key = dict(code_sha256=digest(Path(__file__)),weights=lock,device=device,
               data={r["id"]:r["sha256"] for r in rows},
               queries=[[r["row"]["id"],r["row"]["sha256"],r["box"],r["side"]] for r in templates])
    path=destination(backend,scope)
    saved=read_json(path) if path.exists() else dict(key=key,frames={})
    if saved["key"]!=key:
        raise ValueError("The detector, prompts or source images changed.")
    todo=[r for r in rows if r["id"] not in saved["frames"]]
    if not todo:
        return
    detector=ExemplarDetector(device,templates) if templates else OpenMMLabDetector(device)
    for row in todo:
        picture=source(row)
        start=perf_counter()
        first=detector(picture)
        side=min(picture.size)
        x,y=round((picture.width-side)/2),round((picture.height-side)/2)
        second={} if not (x or y) else detector(picture.crop((x,y,x+side,y+side)))
        merged={}
        for name,pred in first.items():
            crop_regions=[{**p,"bbox":box} for p in second.get(name,[])
                          if (box:=project(p["bbox"],(x,y,x+side,y+side),picture.size)) is not None]
            merged[name+"-full"]=pred
            merged[name+"-wide"]=suppress([*pred,*crop_regions],.43)
        saved["frames"][row["id"]]=dict(proposals=merged,ms=(perf_counter()-start)*1000)
        write_json(path,saved)
        print("Exemplar comparison",backend,scope,len(saved["frames"]),"/",len(rows),flush=True)


def measure():
    if RESULT.exists():
        raise ValueError("The development comparison already exists.")
    results={}
    for backend in BACKENDS:
        for scope in DEVELOPMENT:
            rows=inputs(scope)
            data=read_json(destination(backend,scope))
            if set(data["frames"])!={r["id"] for r in rows}:
                raise ValueError("Complete every development inference before comparing models.")
            results.setdefault(backend,{})[scope]=dict(
               median_ms=median(r["ms"] for r in data["frames"].values()),
               variants={name:short(evaluate(rows,{key:r["proposals"][name] for key,r in data["frames"].items()},.20))
                         for name in next(iter(data["frames"].values()))["proposals"]},
               proposals_sha256=digest(destination(backend,scope)))
    baseline={}
    for scope in DEVELOPMENT:
        rows=inputs(scope)
        base=loaded(scope,"baseline")
        baseline[scope]=short(evaluate(rows,{key:r["ranked"] for key,r in base.items()},read_json(FLOOR)["cutoff"]))
    write_json(RESULT,dict(results=results,baseline=baseline,evidence="Same development groups. Ranking AP is the "
                       "principal comparison. Operating metrics use nominal per-arm cutoffs for diagnostics only."))
    print("Independent generation experiments measured across development groups.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","measure"))
    parser.add_argument("--backend",choices=BACKENDS)
    parser.add_argument("--scope",choices=DEVELOPMENT)
    parser.add_argument("--device",choices=("cpu","xpu"),default="cpu")
    args=parser.parse_args()
    if args.action=="measure":
        measure()
    elif args.backend and args.scope:
        collect(args.backend,args.scope,args.device)
    else:
        parser.error("Collection requires a model and a development scope.")
