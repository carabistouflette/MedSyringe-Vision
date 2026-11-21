"""Run a matched-budget training ablation with additional independent photos."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import DFineForObjectDetection

from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.evaluate import best_threshold, evaluate
from scripts.detection_hillclimb_dfine import EDGE, RECORD, RUN, pixels, predict_batches, target
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, SELECTED, inputs
from scripts.detection_hillclimb_mix import LOCK as MIX_LOCK
from scripts.expanded_detection_study import short
from scripts.expanded_proposals import image_for
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

SEED = 91143
PHOTO = ART/"public_additional_fit.json"
MANIFEST = ART/"dfine_photo_ablation_manifest.json"
DESTINATION = DATA/"runs"/"dfine-photo-ablation"
LOG = ART/"dfine_photo_ablation_training.json"
ARMS = ("previous-real-mix", "additional-real-photos")


def plan():
    if SELECTED.exists():
        raise ValueError("The study's deployment decision is already frozen.")
    prior = read_json(MIX_LOCK)["rows"]
    added = read_json(PHOTO)["rows"]
    protected = [r for split in (*DEVELOPMENT,"public-audit","previous-reserve") for r in inputs(split)]
    protected.extend(read_json(ART/"dental_manifest.json")["rows"])
    both = [{**r,"split":"fit"} for r in (*prior,*added)]
    if len({r["id"] for r in both}) != len(both) or leakage([*both,*[{**r,"split":"protected"} for r in protected]]):
        raise ValueError("Photo adaptation overlaps protected data or contains repeated IDs.")
    previous_photo = [r for r in prior if r["training_domain"] != "phlebotomy"]
    simulated = [r for r in prior if r["training_domain"] == "phlebotomy"]
    def sample(rows,size,salt):
        return sorted(rows,key=lambda r:sha256(f"{SEED}:{salt}:{r['id']}".encode()).digest())[:size]

    # Keep the same training steps and positive/negative balance for both arms.
    keep_sim = [*sample([r for r in simulated if r["objects"]],28,"sim-pos"),
                *sample([r for r in simulated if not r["objects"]],20,"sim-neg")]
    weights = (36, 24)
    mixture = {}
    for arm in ARMS:
        pool = [*previous_photo,*added] if arm == ARMS[1] else previous_photo
        positive = [r for r in pool if r["objects"]]
        negative = [r for r in pool if not r["objects"]]
        if arm == ARMS[1]:
            pos_new = [r for r in added if r["objects"]]
            neg_new = [r for r in added if not r["objects"]]
            sampled = [*pos_new,*sample([r for r in previous_photo if r["objects"]],weights[0]-len(pos_new),"old-pos"),
                       *neg_new,*sample([r for r in previous_photo if not r["objects"]],weights[1]-len(neg_new),"old-neg")]
        else:
            # Use every real source before filling the matched sampling budget.
            sampled = [*sample(positive,weights[0],"old-pos"),
                       *sample(negative,weights[1],"old-neg")]
            for split_rows, limit in zip((positive,negative),weights,strict=True):
                missing = limit - min(len(split_rows),limit)
                sampled.extend(sample(split_rows,missing,"repeat"))
        mixture[arm] = [*keep_sim,*sampled]
    if any(len(rows)!=sum(weights)+len(keep_sim) for rows in mixture.values()):
        raise ValueError("Adaptation arms have different training budgets.")
    bindings = {str(p):digest(p) for p in (PHOTO,MIX_LOCK,RECORD)}
    manifest = dict(arms={k:[r["id"] for r in v] for k,v in mixture.items()}, inputs=bindings,
                    rows={r["id"]:r for r in both}, seed=SEED,
                    counts={arm:dict(samples=len(rows), distinct_images=len({r["id"] for r in rows}),
                                    groups=len({r["group"] for r in rows}),
                                    domains=dict(Counter(r.get("training_domain","additional public photo") for r in rows)),
                                    positives=sum(bool(r["objects"]) for r in rows)) for arm,rows in mixture.items()})
    if MANIFEST.exists() and manifest != read_json(MANIFEST):
        raise ValueError("The balanced training recipe differs from its frozen manifest.")
    write_json(MANIFEST,manifest)
    print("Matched plans",manifest["counts"],flush=True)
    return manifest


def augmented(tensor,labels,rng):
    boxes = labels["boxes"]
    if torch.rand((),generator=rng).item()<.5:
        tensor = tensor.flip(-1)
        boxes[:,0] = 1-boxes[:,0]
    if torch.rand((),generator=rng).item()<.35:
        tensor = tensor.rot90(1,(-2,-1))
        x,y,width,height = boxes.clone().T
        labels["boxes"] = torch.stack((y,1-x,height,width),dim=1)
    exposure = .84+.32*torch.rand((),generator=rng).item()
    contrast = .87+.26*torch.rand((),generator=rng).item()
    return ((tensor-.5)*contrast+.5).mul(exposure).clamp(0,1)


def trainer(location, epochs):
    if LOG.exists():
        raise ValueError("Photo adaptation is already measured.")
    configure_cpu()
    manifest = plan()
    source_proof = read_json(RECORD)
    if any(digest(path)!=value for path,value in source_proof["files"].items()):
        raise ValueError("The initialization checkpoint differs from its recorded bytes.")
    rows = manifest["rows"]
    used = {identifier for arm in ARMS for identifier in manifest["arms"][arm]}
    rasters = {key:pixels(image_for(rows[key],location) if "file" in rows[key] else source(rows[key]),EDGE)
               for key in used}
    checks = {scope:[(r["id"],pixels(source(r),EDGE)) for r in inputs(scope)] for scope in DEVELOPMENT}
    arms = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        rng = torch.Generator().manual_seed(SEED)
        model = DFineForObjectDetection.from_pretrained(RUN/"best",local_files_only=True)
        model.model.backbone.requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00019,weight_decay=.0005)
        directory = DESTINATION/arm
        directory.mkdir(parents=True,exist_ok=True)
        last = directory/"last.pt"
        history = []
        elapsed = 0.
        if last.is_file():
            saved = torch.load(last,map_location="cpu",weights_only=True)
            if saved["plan_sha256"]!=digest(MANIFEST) or saved["script_sha256"]!=digest(__file__):
                raise ValueError("Cannot resume from a different training plan.")
            model.load_state_dict(saved["weights"])
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["torch_rng"])
            rng.set_state(saved["augmentation_rng"])
            history=saved["history"]
            elapsed=history[-1]["elapsed_seconds"]
        start = perf_counter()
        for epoch in range(len(history)+1,epochs+1):
            losses = []
            model.train()
            model.model.backbone.eval()
            identifiers = manifest["arms"][arm]
            order = torch.randperm(len(identifiers),generator=rng)
            for items in order.split(4):
                labels = [target(rows[identifiers[i]]) for i in items]
                images = [augmented(rasters[identifiers[i]],label,rng) for i,label in zip(items,labels,strict=True)]
                value = model(pixel_values=torch.stack(images),labels=labels).loss
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite loss in the photo ablation.")
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
                losses.append(value.item())
            outcomes = {}
            for scope in DEVELOPMENT:
                predictions = predict_batches(model,checks[scope])
                truth = inputs(scope)
                outcomes[scope] = short(evaluate(truth,predictions,best_threshold(truth,predictions)))
            item = dict(epoch=epoch,loss=float(np.mean(losses)),development=outcomes,
                        elapsed_seconds=elapsed+perf_counter()-start)
            history.append(item)
            rank = lambda value: (value["development"]["public-development"]["ap50"],
                                  value["development"]["phlebotomy-development"]["ap50"])
            if max(history,key=rank) is item:
                model.save_pretrained(directory/"best")
            temporary = last.with_suffix(".tmp")
            torch.save(dict(plan_sha256=digest(MANIFEST),script_sha256=digest(__file__),history=history,
                            weights=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                            augmentation_rng=rng.get_state()),temporary)
            temporary.replace(last)
            print(arm,item,flush=True)
        arms[arm] = dict(history=history,weights={str(p):digest(p) for p in sorted((directory/"best").glob("*"))},
                         resume_checkpoint=str(last), resume_checkpoint_sha256=digest(last))
    write_json(LOG,dict(arms=arms,plan_sha256=digest(MANIFEST),code_sha256=digest(__file__),epochs=epochs,
                        device="local CPU", torch_version=torch.__version__))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=3)
    args=parser.parse_args()
    trainer(args.source,args.epochs)
