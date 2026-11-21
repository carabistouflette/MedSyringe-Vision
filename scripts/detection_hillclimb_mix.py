"""Continue the Nano detector on a disjoint mix of simulated and ordinary photos."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from transformers import DFineForObjectDetection

from medsyringe.additive import project
from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.evaluate import best_threshold, evaluate
from scripts.detection_hillclimb_corpus import ART, DATA, LOCK as CORPUS, SEED
from scripts.detection_hillclimb_dfine import EDGE, RECORD as INITIAL, RUN as FIRST, pixels, predict_batches, target
from scripts.detection_hillclimb_experiment import DEVELOPMENT, HELD_OUT, SELECTED, inputs, tiles
from scripts.expanded_detection_study import short
from scripts.expanded_proposals import image_for, public_rows, rows_for
from scripts.precision_study import source

LOCK = ART/"multidomain_training_inputs.json"
LOG = ART/"dfine_multidomain_training.json"
FOLDER = DATA/"runs"/"dfine-multidomain"
PHOTO_LABELS = Path("artifacts/expanded/annotations.json")


def group_sample(rows,limit):
    counter=Counter()
    selected=[]
    for r in sorted(rows,key=lambda row:sha256(f"{SEED}:mix:{row['id']}".encode()).digest()):
        group=r["group"]
        if counter[group] < limit:
            selected.append(r)
            counter[group]+=1
    return selected


def prepare():
    if LOCK.exists():
        raise ValueError("The cross-source training inputs are already frozen.")
    original=read_json(CORPUS)["rows"]
    simulate=[r for r in original if r["split"]=="fit"]
    phlebotomy=group_sample(simulate,4)
    shots=read_json(PHOTO_LABELS)["rows"]
    usable=[row for row in shots if row["split"]=="fit"]
    selected={"phlebotomy":phlebotomy,"capture":group_sample(usable,5),
              "public-previous-fit":public_rows("fit")}
    protected=[*inputs("phlebotomy-development"),*inputs("public-development"),*inputs("public-audit"),
               *rows_for("reserve"),*(r for r in shots if r["split"]!="fit")]
    dental=ART/"dental_manifest.json"
    if dental.exists():
        protected.extend(read_json(dental)["rows"])
    fit=[{**r,"split":"new-fit","training_domain":key} for key,rows in selected.items() for r in rows]
    preserve=[{**r,"split":"protected"} for r in protected]
    clashes=leakage([*fit,*preserve])
    ids={r["id"]:r["group"] for r in fit}
    blocked={ids[c["images"][0]] for c in clashes}
    fit=[r for r in fit if r["group"] not in blocked]
    if leakage([*fit,*preserve]):
        raise ValueError("The multi-domain adaptation inputs overlap a protected group.")
    measures={}
    for key in selected:
        rows=[r for r in fit if r["training_domain"]==key]
        measures[key]=dict(images=len(rows),references=sum(len(r["objects"]) for r in rows),
                           negatives=sum(not r["objects"] for r in rows),groups=len({r["group"] for r in rows}))
    if any(not value["images"] for value in measures.values()):
        raise ValueError("A required training domain has no eligible images.")
    write_json(LOCK,dict(rows=fit,counts=measures,excluded_groups=sorted(blocked),
                         phlebotomy_sha256=digest(CORPUS),photo_sha256=digest(PHOTO_LABELS),
                         public_development_sha256=digest(ART/"public_manifest.json"),
                         original_model_sha256=digest(INITIAL)))
    print("Multi-domain fit",measures,"excluded",sorted(blocked),flush=True)


def train(location,epochs):
    seed=SEED+41
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    record=read_json(INITIAL)
    manifest=read_json(LOCK)
    if manifest["original_model_sha256"]!=digest(INITIAL):
        raise ValueError("The warm-start checkpoint changed.")
    if any(digest(p)!=h for p,h in record["files"].items()):
        raise ValueError("The warm-start weights changed.")
    FOLDER.mkdir(parents=True,exist_ok=True)
    train_data=manifest["rows"]
    arrays=[pixels(image_for(row,location) if "file" in row else source(row),EDGE) for row in train_data]
    validation={g:[(r["id"],pixels(source(r),EDGE)) for r in inputs(g)] for g in DEVELOPMENT}
    model=DFineForObjectDetection.from_pretrained(FIRST/"best",local_files_only=True)
    model.model.backbone.requires_grad_(False)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.00018,weight_decay=.0005)
    trials=[]
    from time import perf_counter

    started=perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        model.model.backbone.eval()
        order=torch.randperm(len(train_data),generator=generator).tolist()
        losses=[]
        for offset in range(0,len(order),4):
            ids=order[offset:offset+4]
            if len(ids)==1:
                ids=[order[offset-1],*ids]
            truths=[target(train_data[i]) for i in ids]
            samples=[]
            for i,member in enumerate(ids):
                image=arrays[member]
                if torch.rand((),generator=generator).item()<.5:
                    image=image.flip(-1)
                    truths[i]["boxes"][:,0]=1-truths[i]["boxes"][:,0]
                factor=.84+.32*torch.rand((),generator=generator).item()
                samples.append((image*factor).clamp(0,1))
            loss=model(pixel_values=torch.stack(samples),labels=truths).loss
            if not torch.isfinite(loss):
                raise ValueError("Cross-source adaptation produced a nonfinite loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            losses.append(loss.detach().item())
            if len(losses)%12==0:
                print("Multi-domain",epoch,len(losses),round(losses[-1],3),flush=True)
        scores={}
        for scope in DEVELOPMENT:
            references=inputs(scope)
            predictions=predict_batches(model,validation[scope])
            scores[scope]=short(evaluate(references,predictions,best_threshold(references,predictions)))
        item=dict(epoch=epoch,loss=float(np.mean(losses)),outcomes=scores,elapsed_seconds=perf_counter()-started)
        trials.append(item)
        torch.save(dict(weights=model.state_dict(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
                        history=trials, input_sha256=digest(LOCK)),FOLDER/"last.pt")
        def metric(value):
            scopes=value["outcomes"]
            return (min(g["ap50"] for g in scopes.values()),sum(g["ap50_95"] for g in scopes.values()))
        if max(trials,key=metric) is item:
            model.save_pretrained(FOLDER/"best")
        print("Multi-domain checkpoint",item,flush=True)
    files={str(p):digest(p) for p in (FOLDER/"best").iterdir() if p.is_file()}
    write_json(LOG,dict(seed=seed,trials=trials,selected_epoch=max(trials,key=metric)["epoch"],files=files,
                        dataset_sha256=digest(LOCK),initial_sha256=digest(INITIAL),train_device="cpu",
                        code_sha256=digest(__file__),elapsed_seconds=perf_counter()-started))


def proposal_path(scope):
    return DATA/"proposals"/f"{scope}-dfine-multidomain.json"


def collect(scope):
    if scope in HELD_OUT and not SELECTED.exists():
        raise ValueError("Freeze the rule before held-out inference.")
    authority=read_json(LOG)
    if any(digest(p)!=h for p,h in authority["files"].items()):
        raise ValueError("Multi-domain detector weights have changed.")
    references=inputs(scope)
    target_path=proposal_path(scope)
    key=dict(training_sha256=digest(LOG), rows={r["id"]:r.get("derived_sha256",r["sha256"]) for r in references})
    data=read_json(target_path) if target_path.exists() else dict(key=key,frames={})
    if data["key"]!=key:
        raise ValueError("Multi-domain inference inputs changed.")
    model=DFineForObjectDetection.from_pretrained(FOLDER/"best",local_files_only=True)
    frames=data["frames"]
    from time import perf_counter

    for row in references:
        identifier=row["id"]
        if identifier in frames:
            continue
        image=source(row)
        tick=perf_counter()
        windows=list(tiles(image))
        inputs_list=[("full",pixels(image,EDGE)),*((str(i),pixels(image.crop(box),EDGE)) for i,box in enumerate(windows))]
        result=predict_batches(model,inputs_list)
        views=[]
        for i,window in enumerate(windows):
            for p in result[str(i)]:
                if p["score"]>=.008 and (box:=project(p["bbox"],window,image.size)):
                    views.append({**p,"bbox":box,"view":i})
        frames[identifier]=dict(full=result["full"],windows=views,ms=(perf_counter()-tick)*1000)
        if len(frames)%4==0 or len(frames)==len(references):
            write_json(target_path,data)
            print(scope,"multi-domain proposals",len(frames),"/",len(references),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","train","collect"))
    parser.add_argument("--source",type=Path)
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--scope",choices=(*DEVELOPMENT,*HELD_OUT))
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.action=="prepare":
        prepare()
    elif args.action=="train" and args.source:
        train(args.source,args.epochs)
    elif args.action=="collect" and args.scope:
        collect(args.scope)
    else:
        parser.error("Train needs --source; collect needs --scope.")
