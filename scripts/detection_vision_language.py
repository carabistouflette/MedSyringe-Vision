"""Evaluate local language-guided localization as a second source of evidence."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from PIL import ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor

from medsyringe.additive import project
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import suppress
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, HELD_OUT, SELECTED, inputs
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

SOURCES = {
    "florence": ("florence-community/Florence-2-base-ft", "0b03b6f15a4a211370fb204aee4e7dd48887ea37",
                 "MIT", Path("data/models/florence-2-base-ft")),
    "qwen": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "Apache-2.0",
             Path("data/models/qwen3-vl-2b")),
}
PHRASE = "<CAPTION_TO_PHRASE_GROUNDING>"
FLORENCE_PROMPTS = ("medical syringe", "hypodermic syringe with its barrel and plunger")
QWEN_PROMPT = (
    "Locate every distinct medical syringe visibly present in the photo, including a syringe in a pump or in a hand. "
    "A syringe has a barrel and plunger. Include its visible needle or cap when attached. "
    "Exclude medical vials, injector pens, pipettes, tubing, empty pumps and loose needles without a syringe. "
    "Return a JSON array with one item {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"syringe\"} "
    "per visible syringe. Coordinates are integers from 0 to 1000. Return [] when none are visible."
)


def qwen_boxes(value):
    text = value.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-len("```")].strip()
    try:
        rows = json.loads(text)
    except (TypeError,ValueError):
        return None
    if not isinstance(rows,list) or len(rows)>50:
        return None
    boxes = []
    for row in rows:
        if not isinstance(row,dict) or row.get("label") != "syringe":
            return None
        points = row.get("bbox_2d")
        if not isinstance(points,list) or len(points)!=4 or any(type(n) is not int or not 0<=n<=1000 for n in points):
            return None
        bbox = [v/1000 for v in points]
        if not valid_box(bbox):
            return None
        boxes.append(bbox)
    return boxes


def model_lock(backend):
    repo,revision,license_name,folder = SOURCES[backend]
    path = ART / f"{backend}_grounding_backbone.json"
    if not path.exists():
        key = {p.name:digest(p) for p in sorted(folder.iterdir())
               if p.suffix in (".json", ".safetensors", ".txt") and p.is_file()}
        if not {"model.safetensors","config.json","tokenizer.json","preprocessor_config.json"} <= set(key):
            raise ValueError("The exact checkpoint has not been restored.")
        write_json(path,dict(repo=repo,revision=revision,license=license_name,files=key))
    record = read_json(path)
    if any(digest(folder/name)!=value for name,value in record["files"].items()):
        raise ValueError("The VLM weights do not match their lock.")
    return digest(path)


class LocalGrounding:
    def __init__(self,backend,device):
        self.processor = AutoProcessor.from_pretrained(SOURCES[backend][3], local_files_only=True)
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            SOURCES[backend][3],local_files_only=True,use_safetensors=True,
            dtype=torch.bfloat16 if backend=="qwen" else torch.float32,
            attn_implementation="sdpa" if backend=="qwen" else "eager").eval().to(device)
        self.backend = backend

    @torch.inference_mode()
    def qwen(self,image):
        view = ImageOps.contain(image,(768,768))
        messages = [{"role":"user", "content":[{"type":"image","image":view},
                                               {"type":"text","text":QWEN_PROMPT}]}]
        formatted = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch = self.processor(text=[formatted],images=[view],return_tensors="pt").to(self.device)
        seq = self.model.generate(**batch,do_sample=False,max_new_tokens=192)
        tokens = seq[:,batch.input_ids.shape[1]:]
        text = self.processor.batch_decode(tokens,skip_special_tokens=True)[0]
        boxes = qwen_boxes(text)
        return dict(boxes=boxes or [], parsed=boxes is not None, response=text)

    @torch.inference_mode()
    def florence(self,image):
        results = []
        for prompt in FLORENCE_PROMPTS:
            batch = self.processor(text=PHRASE+prompt,images=image,return_tensors="pt").to(self.device)
            seq = self.model.generate(**batch,do_sample=False,max_new_tokens=96,num_beams=3)
            text = self.processor.batch_decode(seq,skip_special_tokens=False)[0]
            answer = self.processor.post_process_generation(text,task=PHRASE,image_size=image.size)[PHRASE]
            boxes = [[min(1.,max(0.,float(v/extent))) for v,extent in zip(coords,(*image.size,*image.size),strict=True)]
                     for coords in answer["bboxes"]]
            results.append(dict(boxes=[p for p in boxes if valid_box(p)],parsed=True,response=text))
        return results

    def __call__(self,image):
        records = [self.qwen(image)] if self.backend=="qwen" else self.florence(image)
        boxes = [dict(bbox=bbox, score=.5) for entry in records for bbox in entry["boxes"]]
        return suppress(boxes,.50),records


def path(scope,backend):
    return DATA / "proposals" / f"{scope}-{backend}.json"


def collect(scope,backend,device):
    if scope in HELD_OUT and not SELECTED.exists():
        raise ValueError("The language-model policy must be frozen before opening the audit.")
    configure_cpu()
    if device=="xpu" and not torch.xpu.is_available():
        raise ValueError("Local XPU unavailable.")
    rows = inputs(scope)
    key = dict(weights_sha256=model_lock(backend),script_sha256=digest(__file__),device=device,
               hashes={r["id"]:r.get("derived_sha256",r["sha256"]) for r in rows})
    destination = path(scope,backend)
    payload = read_json(destination) if destination.exists() else dict(key=key,frames={})
    if payload["key"] != key:
        raise ValueError("The language-model cache differs from its inputs.")
    runner = None
    for row in rows:
        identifier = row["id"]
        if identifier in payload["frames"]:
            continue
        if runner is None:
            runner = LocalGrounding(backend,device)
        image = source(row)
        start = perf_counter()
        boxes,records = runner(image)
        extra = []
        crop_records = []
        if backend=="florence":
            side = min(image.size)
            x,y = (image.width-side)//2,(image.height-side)//2
            if x or y:
                region = (x,y,x+side,y+side)
                focused,crop_records = runner(image.crop(region))
                extra = [{**p,"bbox":b} for p in focused if (b:=project(p["bbox"],region,image.size))]
        payload["frames"][identifier] = dict(full=boxes,focused=extra,records=records,
                                            crop_records=crop_records,ms=(perf_counter()-start)*1000)
        if len(payload["frames"])%2==0 or len(payload["frames"])==len(rows):
            write_json(destination,payload)
            print(scope,backend,len(payload["frames"]),"/",len(rows),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope",required=True,choices=(*DEVELOPMENT,*HELD_OUT))
    parser.add_argument("--backend",required=True,choices=SOURCES)
    parser.add_argument("--device",default="xpu",choices=("cpu","xpu"))
    args = parser.parse_args()
    collect(args.scope,args.backend,args.device)
