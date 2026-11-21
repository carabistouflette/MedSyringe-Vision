"""Compare newer open-vocabulary proposals before opening the public audit."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from medsyringe.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou, model_hashes, suppress
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, HELD_OUT, SELECTED, inputs
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

OPEN_MODELS = ("dino-base", "owlv2")
POSITIVE = ("syringe", "hypodermic syringe", "injection syringe", "syringe in an infusion pump")
NEGATIVE = ("pen", "pipette", "laboratory tube", "insulin pen", "infusion pump", "vial", "loose needle")
WEIGHTS = ART / "grounding_dino_base_backbone.json"
DINO_ROOT = Path("data/models/grounding-dino-base")
DINO_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


def model_lock(backend):
    if backend == "owlv2":
        return model_hashes(Path("."),"owl")
    if not WEIGHTS.exists():
        files = ("config.json","model.safetensors","preprocessor_config.json","tokenizer_config.json",
                 "special_tokens_map.json","tokenizer.json","vocab.txt")
        write_json(WEIGHTS,dict(model="IDEA-Research/grounding-dino-base",revision=DINO_REVISION,
                                license="Apache-2.0",files={name:digest(DINO_ROOT/name) for name in files}))
    locks = read_json(WEIGHTS)["files"]
    if any(digest(DINO_ROOT/name) != checksum for name,checksum in locks.items()):
        raise ValueError("Grounding DINO Base weights do not match the model lock.")
    return locks


class Detector:
    def __init__(self,backend,device):
        self.owl = backend == "owlv2"
        self.device = device
        path = Path("data/models/owl") if self.owl else DINO_ROOT
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)
        if not self.owl:
            self.processor.image_processor.size = {"shortest_edge":640,"longest_edge":1024}
        self.prompt = [[f"a photo of a {name}" for name in (*POSITIVE,*NEGATIVE)]] if self.owl else (
            ". ".join(POSITIVE)+".")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            path,local_files_only=True,use_safetensors=True).eval().to(device)

    @torch.inference_mode()
    def __call__(self,image):
        values = self.processor(images=image,text=self.prompt,return_tensors="pt").to(self.device)
        output = self.model(**values)
        if self.owl:
            logits = output.logits[0].sigmoid()
            positives = logits[:,:len(POSITIVE)].amax(dim=1).cpu().numpy()
            negatives = logits[:,len(POSITIVE):].amax(dim=1).cpu().numpy()
            all_boxes = output.pred_boxes[0].cpu().numpy()
            scale = max(image.size)/np.array([image.width,image.height,image.width,image.height])
            cx,cy,w,h = all_boxes.T
            boxes = np.column_stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2))*scale
        else:
            target = [(image.height,image.width)]
            nodes = self.processor.post_process_grounded_object_detection(
                output, values.input_ids, threshold=.10, target_sizes=target)[0]
            positives = nodes["scores"].cpu().numpy()
            negatives = np.zeros_like(positives)
            boxes = nodes["boxes"].cpu().numpy()/np.array([image.width,image.height,image.width,image.height])
        values = []
        for box,score,negative in zip(np.clip(boxes,0,1),positives,negatives,strict=True):
            if score >= .023 and box[2]-box[0]>=.006 and box[3]-box[1]>=.006:
                values.append(dict(bbox=box.tolist(),score=float(score),negative=float(negative)))
        return suppress(values,.50)[:40]


def proposal_path(scope,backend):
    return DATA/"proposals"/f"{scope}-{backend}.json"


def collection_key(scope,backend,device):
    return dict(weights=model_lock(backend),images={r["id"]:r.get("derived_sha256",r["sha256"]) for r in inputs(scope)},
                code_sha256=digest(__file__),device=device)


def collect(scope,backend,device):
    if scope in HELD_OUT and not SELECTED.exists():
        raise ValueError("Freeze model selection before generating audit predictions.")
    configure_cpu()
    if device=="xpu" and not torch.xpu.is_available():
        raise ValueError("The requested Intel XPU is unavailable.")
    rows = inputs(scope)
    key = collection_key(scope,backend,device)
    path = proposal_path(scope,backend)
    saved = read_json(path) if path.exists() else dict(key=key,frames={})
    if key != saved["key"]:
        raise ValueError("Open-vocabulary proposal cache does not match the model or image set.")
    detector = None
    for row in rows:
        identifier = row["id"]
        if identifier in saved["frames"]:
            continue
        image = source(row)
        if detector is None:
            detector = Detector(backend,device)
        started = perf_counter()
        full = detector(image)
        window = min(image.width, image.height)
        x,y = round((image.width-window)*.5), round((image.height-window)*.5)
        region = (x,y,x+window,y+window)
        focused = []
        if x or y:
            for result in detector(image.crop(region)):
                box = project(result["bbox"],region,image.size)
                if box:
                    focused.append({**result,"bbox":box})
        saved["frames"][identifier] = dict(full=full,focused=focused,ms=(perf_counter()-started)*1000)
        if len(saved["frames"])%4 == 0 or len(saved["frames"]) == len(rows):
            write_json(path,saved)
            print(scope,backend,len(saved["frames"]),"/",len(rows),flush=True)


def read_proposals(scope,backend):
    saved = read_json(proposal_path(scope,backend))
    if saved["key"]["code_sha256"] != digest(__file__) or set(saved["frames"]) != {r["id"] for r in inputs(scope)}:
        raise ValueError("Open-vocabulary predictions are incomplete or obsolete.")
    return saved["frames"]


def open_views(scope,backend):
    result = {view:{} for view in ("whole","wide","mixed","stable")}
    for key,frame in read_proposals(scope,backend).items():
        whole = frame["full"]
        focused = frame["focused"]
        joined = suppress([*whole,*focused],.43)
        semantic = suppress([{**p,"score": p["score"]-p["negative"] if backend=="owlv2" else p["score"]}
                             for p in (*whole,*focused) if p["score"]>p["negative"]],.43)
        stable = [p for p in joined if any(box_iou(p["bbox"],q["bbox"])>=.55 for q in whole)
                  and any(box_iou(p["bbox"],q["bbox"])>=.55 for q in focused)]
        for name,proposals in zip(result,(whole,joined,semantic,stable),strict=True):
            result[name][key] = proposals
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope",required=True,choices=(*DEVELOPMENT,*HELD_OUT))
    parser.add_argument("--backend",required=True,choices=OPEN_MODELS)
    parser.add_argument("--device",choices=("cpu","xpu"),default="cpu")
    args = parser.parse_args()
    collect(args.scope,args.backend,args.device)
