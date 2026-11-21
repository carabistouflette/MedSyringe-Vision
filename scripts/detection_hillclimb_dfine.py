"""Fine-tune a one-class D-FINE Nano and collect native multi-scale detections."""
import argparse
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
from transformers import DFineForObjectDetection

from medsyringe.additive import project
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import best_threshold, evaluate
from scripts.detection_hillclimb_corpus import ART, DATA, LOCK, SEED
from scripts.detection_hillclimb_experiment import DEVELOPMENT, HELD_OUT, inputs, tiles
from scripts.expanded_detection_study import short
from scripts.precision_study import source

MODEL = Path("data/models/dfine-nano-coco")
REVISION = "066438d3d8f0da137a37b38fdf3368fd4afceced"
RECORD = ART / "dfine_training.json"
RUN = DATA / "runs/dfine"
SELECTION = ART / "dfine_selection.json"
EDGE = 384


def pixels(image, edge):
    array = np.array(image.resize((edge,edge),Image.Resampling.BILINEAR),dtype=np.float32)
    return torch.from_numpy(array).permute(2,0,1)/255.


def target(row):
    boxes = torch.tensor([p["bbox"] for p in row["objects"]],dtype=torch.float32).reshape(-1,4)
    return dict(class_labels=torch.zeros(len(boxes),dtype=torch.int64),boxes=box_convert(boxes,"xyxy","cxcywh"))


def outputs(logits, boxes):
    scores = logits.sigmoid().squeeze(-1)
    bounds = box_convert(boxes,"cxcywh","xyxy").clamp(0,1)
    detections = []
    for score, bbox in zip(scores.tolist(),bounds.tolist(),strict=True):
        if score >= .005 and valid_box(bbox):
            detections.append(dict(bbox=bbox,score=score))
    return suppress(detections,.45)


@torch.inference_mode()
def predict_batches(model, evidence, batch_size=6):
    model.eval()
    findings = {}
    for i in range(0,len(evidence),batch_size):
        segment = evidence[i:i+batch_size]
        batch = torch.stack([pixels for _,pixels in segment])
        result = model(pixel_values=batch)
        for name, logits, boxes in zip([x[0] for x in segment],result.logits.cpu(),result.pred_boxes.cpu(),strict=True):
            findings[name] = outputs(logits,boxes)
    return findings


def training(epochs, resume=False):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    architecture = dict(source="ustc-community/dfine-nano-coco",revision=REVISION,license="Apache-2.0",
                        files={str(p):digest(p) for p in sorted(MODEL.iterdir()) if p.is_file()})
    samples = read_json(LOCK)["rows"]
    training = [r for r in samples if r["split"] == "fit"]
    checks = [r for r in samples if r["split"] == "development"]
    cached = [pixels(source(r),EDGE) for r in training]
    validation = [(r["id"],pixels(source(r),EDGE)) for r in checks]
    model = DFineForObjectDetection.from_pretrained(MODEL,num_labels=1,id2label={0:"syringe"},
                         label2id={"syringe":0},ignore_mismatched_sizes=True,local_files_only=True,
                         num_denoising=30,use_pretrained_backbone=False)
    model.model.backbone.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00035,weight_decay=.0005)
    RUN.mkdir(parents=True,exist_ok=True)
    history = []
    prior_seconds = 0.
    last = RUN/"last.pt"
    if resume:
        saved = torch.load(last,map_location="cpu",weights_only=True)
        if saved["input_hash"] != digest(LOCK):
            raise ValueError("The resumable training inputs changed.")
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        generator.set_state(saved["generator"])
        history = saved["history"]
        prior_seconds = float(saved["elapsed_seconds"]) if "elapsed_seconds" in saved else float(read_json(RECORD)["elapsed_seconds"])
    started = perf_counter()
    for epoch in range(len(history),epochs):
        model.train()
        model.model.backbone.eval()
        ids = torch.randperm(len(training),generator=generator).tolist()
        losses = []
        for offset in range(0,len(ids),4):
            indices = ids[offset:offset+4]
            ground_truth = [target(training[j]) for j in indices]
            stack = [cached[j] for j in indices]
            for i,sample in enumerate(stack):
                if torch.rand((),generator=generator).item() < .5:
                    sample = sample.flip(-1)
                    ground_truth[i]["boxes"][:,0] = 1-ground_truth[i]["boxes"][:,0]
                exposure = .85+.30*torch.rand((),generator=generator).item()
                stack[i] = (sample*exposure).clamp(0,1)
            loss = model(pixel_values=torch.stack(stack),labels=ground_truth).loss
            if not torch.isfinite(loss):
                raise ValueError("D-FINE training loss is nonfinite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            losses.append(loss.detach().item())
            if len(losses)%12 == 0:
                print(f"D-FINE epoch {epoch+1}, batch {len(losses)}, loss {losses[-1]:.3f}",flush=True)
        estimates = predict_batches(model,validation)
        cutoff = best_threshold(checks,estimates)
        outcome = short(evaluate(checks,estimates,cutoff))
        item = dict(epoch=epoch+1,mean_train_loss=float(np.mean(losses)),cutoff=cutoff,**outcome)
        history.append(item)
        torch.save(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
                       generator=generator.get_state(),rng=torch.get_rng_state(),input_hash=digest(LOCK),
                       elapsed_seconds=prior_seconds+perf_counter()-started),last)
        if max(history,key=lambda v:(v["ap50_95"],v["f1"])) is item:
            model.save_pretrained(RUN/"best")
        print("D-FINE development",item,flush=True)
    paths = {str(p):digest(p) for p in (RUN/"best").iterdir() if p.is_file()}
    write_json(RECORD,dict(source=architecture,seed=SEED,edge=EDGE,epochs=epochs,device="cpu",
                           corpus_sha256=digest(LOCK),code_sha256=digest(__file__),torch=version("torch"),
                           transformers=version("transformers"), elapsed_seconds=prior_seconds+perf_counter()-started,
                           fit_images=len(training),validation_images=len(checks),training=history,files=paths,
                           selection="Highest box AP50–95 on the three phlebotomy filename groups."))


def proposal_cache(scope):
    return DATA/"proposals"/f"{scope}-dfine.json"


def collect(scope):
    if scope in HELD_OUT and not SELECTION.exists():
        raise ValueError("Freeze the D-FINE operating point before held-out inference.")
    manifest = read_json(RECORD)
    if manifest["corpus_sha256"] != digest(LOCK) or any(digest(p) != h for p,h in manifest["files"].items()):
        raise ValueError("The local D-FINE checkpoint differs from its provenance.")
    rows = inputs(scope)
    destination = proposal_cache(scope)
    key = dict(checkpoint_sha256=digest(RECORD), code_sha256=digest(__file__),
               rows={r["id"]: r.get("derived_sha256",r["sha256"]) for r in rows})
    previous = read_json(destination) if destination.exists() else dict(key=key,frames={})
    if previous["key"] != key:
        raise ValueError("The D-FINE proposal cache is stale.")
    frames=previous["frames"]
    if len(frames) == len(rows):
        return
    model = DFineForObjectDetection.from_pretrained(RUN/"best",local_files_only=True).eval()
    for row in rows:
        if row["id"] in frames:
            continue
        image = source(row)
        started = perf_counter()
        windows = list(tiles(image))
        evidence = [("whole",pixels(image,EDGE))]+[(str(i),pixels(image.crop(box),EDGE))
                                                 for i,box in enumerate(windows)]
        found = predict_batches(model,evidence)
        projected = []
        for i,window in enumerate(windows):
            for p in found[str(i)]:
                box = project(p["bbox"], window,image.size)
                if box and p["score"] >= .008:
                    projected.append({**p,"bbox":box,"view":i})
        frames[row["id"]] = dict(full=found["whole"],windows=projected,ms=(perf_counter()-started)*1000)
        if len(frames)%4 == 0 or len(frames)==len(rows):
            write_json(destination,previous)
            print(scope,"dfine",len(frames),"/",len(rows),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("train","collect"))
    parser.add_argument("--epochs",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--scope",choices=(*DEVELOPMENT,*HELD_OUT))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.action == "train":
        training(args.epochs,args.resume)
    elif args.scope:
        collect(args.scope)
    else:
        parser.error("Collection requires --scope.")
