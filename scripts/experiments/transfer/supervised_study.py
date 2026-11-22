"""Train and compare real one-class detection heads on grouped source scenes."""
import argparse
from pathlib import Path
from time import perf_counter

import torch

from medsyringe.data import digest, leakage, read_json, train_rows, write_json
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.detection.supervised import SSD_BASE, detector, predict, tensor
from scripts.experiments.detector_training.optimize_detector import counters
from scripts.experiments.detector_audits.precision_study import source

ROOT, ART = Path("."), Path("artifacts")
SEED = 607
TRAIN_VARIANTS = {
    "standard": (0.0, False),
    "hard_negatives": (1.0, False),
    "hard_negatives_augmented": (1.0, True),
}


def sets():
    precision = read_json(ART / "precision_study_manifest.json")
    historical = read_json(ART / "test_manifest.json")["rows"]
    train = [*train_rows(ROOT), *precision["train"], *precision["test"],
             *read_json(ART / "validation_manifest.json")["rows"],
             *read_json(ART / "v2_holdout_manifest.json")["rows"]]
    validation = read_json(ART / "semantic_holdout_manifest.json")["rows"]
    if leakage([*[{**r, "split": "training"} for r in train], *validation, *historical]):
        raise ValueError("The training, validation and historical regression sets overlap.")
    return train, validation, historical


def prepare(rows):
    return [(row["id"], tensor(source(row))) for row in rows]


def batches(rows, tensors, generator, augment, edge=320):
    perm = torch.randperm(len(rows), generator=generator).tolist()
    for start in range(0, len(perm), 4):
        indices = perm[start:start+4]
        # A final single image would imply single-value normalization at deep layers.
        if len(indices) == 1:
            indices = [perm[start-1], *indices]
        images, targets = [], []
        for i in indices:
            box = torch.tensor([obj["bbox"] for obj in rows[i]["objects"]], dtype=torch.float32).reshape(-1, 4)
            image = tensors[i][1]
            if augment:
                if torch.rand((), generator=generator) < .5:
                    image = image.flip(-1)
                    box[:, [0, 2]] = 1-box[:, [2, 0]]
                factor = 0.8 + .4*float(torch.rand((), generator=generator))
                image = (image*factor).clamp(0, 1)
            images.append(image)
            targets.append(dict(boxes=box*edge, labels=torch.ones(len(box), dtype=torch.long)))
        yield images, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    training, val, historical = sets()
    train_images, val_images = prepare(training), prepare(val)
    weight_dir = ROOT / "data/ssd-study"
    weight_dir.mkdir(parents=True, exist_ok=True)
    base_sha = digest(ROOT / SSD_BASE)
    trials = {}
    for name, (empty_weight, augment) in TRAIN_VARIANTS.items():
        torch.manual_seed(SEED)
        rng = torch.Generator().manual_seed(SEED)
        model = detector(ROOT, base_sha, empty_weight)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.002, weight_decay=0.001)
        started = perf_counter()
        record = []
        for epoch in range(1, args.epochs+1):
            model.train()
            model.backbone.eval()
            epoch_losses = []
            for images, targets in batches(training, train_images, rng, augment):
                losses = model(images, targets)
                optimizer.zero_grad()
                sum(losses.values()).backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0)
                optimizer.step()
                epoch_losses.append({k: float(v.detach()) for k, v in losses.items()})
            predictions = predict(model, val_images)
            cutoff = best_threshold(val, predictions)
            result = evaluate(val, predictions, cutoff)
            item = dict(epoch=epoch, cutoff=cutoff, **counters(result),
                        losses={k: sum(v[k] for v in epoch_losses)/len(epoch_losses)
                                for k in epoch_losses[0]}, elapsed_seconds=perf_counter()-started)
            record.append(item)
            if max(record, key=rank) is item:
                torch.save(model.state_dict(), weight_dir / f"{name}.pth")
            print(name, epoch, *(f"{item[k]:.3f}" for k in ("precision", "recall", "f1", "ap50_95")), flush=True)
        trials[name] = record
        write_json(ART / "supervised_progress.json", trials)
    winning_name = max(trials, key=lambda name: rank(max(trials[name], key=rank)))
    selected = max(trials[winning_name], key=rank)
    path = weight_dir / f"{winning_name}.pth"
    model = detector(ROOT)
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    predictions = predict(model, prepare(historical))
    result = evaluate(historical, predictions, selected["cutoff"])
    manifest = dict(training_ids=[r["id"] for r in training], validation_ids=[r["id"] for r in val],
                    regression_ids=[r["id"] for r in historical])
    write_json(ART / "supervised_study.json", dict(
        role="Development and previously inspected historical regression; no independent claim",
        manifest=manifest, training_images=len(training), negative_training_images=sum(not r["objects"] for r in training),
        seed=SEED, base_checkpoint_sha256=base_sha, selected_variant=winning_name,
        checkpoint=str(path), checkpoint_sha256=digest(path), selected=selected, runs=trials,
        historical=counters(result)))
    print("Selected", winning_name, selected, "historical", counters(result), flush=True)


def rank(result):
    return result["f1"], result["ap50_95"], result["precision"]


if __name__ == "__main__":
    main()
