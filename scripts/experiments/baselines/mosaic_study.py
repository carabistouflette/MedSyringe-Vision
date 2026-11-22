"""Test train-only multi-object mosaics with accurately remapped boxes."""
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.functional import interpolate

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.detection.supervised import SSD_BASE, detector, predict
from scripts.experiments.detector_training.optimize_detector import counters
from scripts.experiments.transfer.supervised_study import ART, ROOT, SEED, batches, prepare, rank, sets

EDGE = 320


def mosaic(images, boxes, split):
    x, y = split
    windows = ((0, 0, x, y), (x, 0, EDGE, y), (0, y, x, EDGE), (x, y, EDGE, EDGE))
    frame = images[0].new_zeros((3, EDGE, EDGE))
    targets = []
    for image, labels, (l,t,r,b) in zip(images, boxes, windows):
        frame[:, t:b, l:r] = interpolate(image[None], size=(b-t,r-l), mode="bilinear",
                                         align_corners=False, antialias=True)[0]
        targets.append(labels*labels.new_tensor([r-l,b-t,r-l,b-t])
                       + labels.new_tensor([l,t,l,t]))
    output = torch.cat(targets)
    return frame, dict(boxes=output, labels=torch.ones(len(output), dtype=torch.long))


def main():
    torch.set_num_threads(4)
    train, val, historical = sets()
    train_frames, val_frames = prepare(train), prepare(val)
    positives = torch.tensor([i for i,r in enumerate(train) if r["objects"]])
    negatives = torch.tensor([i for i,r in enumerate(train) if not r["objects"]])
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    extra_gen = torch.Generator().manual_seed(SEED+1)
    weights = digest(Path(SSD_BASE))
    model = detector(ROOT, weights, empty_weight=1.)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.002, weight_decay=.001)
    file = Path("data/ssd-study/mosaic.pth")
    file.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    started = perf_counter()
    for epoch in range(1,9):
        model.train()
        model.backbone.eval()
        for ims, targets in batches(train, train_frames, gen, True):
            pos = positives[torch.randint(len(positives),(2,),generator=extra_gen)].tolist()
            neg = negatives[torch.randint(len(negatives),(2,),generator=extra_gen)].tolist()
            candidates = pos+neg
            ids = [candidates[i] for i in torch.randperm(4,generator=extra_gen).tolist()]
            split = torch.randint(112,209,(2,),generator=extra_gen).tolist()
            truth = [torch.tensor([obj["bbox"] for obj in train[i]["objects"]]).reshape(-1,4)
                     for i in ids]
            ims[-1], targets[-1] = mosaic([train_frames[i][1] for i in ids], truth, split)
            loss = sum(model(ims, targets).values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(),10.)
            optimizer.step()
        predictions = predict(model, val_frames)
        cutoff = best_threshold(val, predictions)
        stats = dict(epoch=epoch, cutoff=cutoff, **counters(evaluate(val,predictions,cutoff)))
        trials.append(stats)
        if max(trials,key=rank) is stats:
            torch.save(model.state_dict(),file)
        print("Mosaic", epoch, stats["f1"], flush=True)
    chosen = max(trials,key=rank)
    model.load_state_dict(torch.load(file,weights_only=True,map_location="cpu"))
    original = read_json(ART/"supervised_study.json")
    result = evaluate(historical,predict(model,prepare(historical)),chosen["cutoff"])
    write_json(ART/"mosaic_study.json",dict(selected=chosen,trials=trials,
        standard_augmentation=max(original["runs"]["hard_negatives_augmented"],key=rank),
        historical=counters(result), checkpoint=str(file),checkpoint_sha256=digest(file),
        train_scenes=len(train), elapsed_seconds=perf_counter()-started, seed=SEED))
    print("Historical", counters(result))


if __name__ == "__main__":
    main()
