"""Compare aspect-preserving training, resolution, and burst subsampling."""
from pathlib import Path
from time import perf_counter

from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.supervised import SSD_BASE, detector, predict
from scripts.optimize_detector import counters
from scripts.precision_study import source
from scripts.supervised_study import ART, ROOT, SEED, batches, rank, sets


def make_views(rows, edge):
    adjusted, views, mapping = [], [], {}
    for row in rows:
        im = source(row)
        im.thumbnail((edge, edge), Image.Resampling.BILINEAR)
        # Let small public photos fill the same canvas as full-size originals.
        ratio = edge/max(im.size)
        w, h = (max(1, round(s*ratio)) for s in im.size)
        patch = im.resize((w, h), Image.Resampling.BILINEAR)
        x, y = (edge-w)//2, (edge-h)//2
        canvas = Image.new("RGB", (edge, edge), (127, 127, 127))
        canvas.paste(patch, (x, y))
        views.append((row["id"], pil_to_tensor(canvas).float()/255))
        mapping[row["id"]] = (x, y, w, h)
        objects = []
        for target in row["objects"]:
            l, t, r, b = target["bbox"]
            objects.append({**target, "bbox": [(l*w+x)/edge, (t*h+y)/edge,
                                                (r*w+x)/edge, (b*h+y)/edge]})
        adjusted.append({**row, "objects": objects})
    return adjusted, views, mapping


def inverse(mapping, predictions, edge):
    result = {}
    for identifier, proposals in predictions.items():
        x, y, w, h = mapping[identifier]
        projected = []
        for p in proposals:
            l, t, r, b = p["bbox"]
            mapped = [max(0, min(1, v)) for v in ((l*edge-x)/w, (t*edge-y)/h,
                                                   (r*edge-x)/w, (b*edge-y)/h)]
            if mapped[0] < mapped[2] and mapped[1] < mapped[3]:
                projected.append({**p, "bbox": mapped})
        result[identifier] = projected
    return result


def model_for(edge, checksum=None):
    net = detector(ROOT, checksum, empty_weight=1.0)
    net.transform.fixed_size = (edge, edge)
    return net


def subsample(rows):
    seen = set()
    out = []
    for row in rows:
        burst = row["objects"][0]["burst"] if row["id"].startswith("IMG_") else row["id"]
        if burst not in seen:
            seen.add(burst)
            out.append(row)
    return out


def main():
    torch.set_num_threads(4)
    training, validation, historical = sets()
    audit = read_json(ART / "supervised_external_manifest.json")["rows"]
    if leakage([*[{**row, "split": "training"} for row in training], *validation, *historical, *audit]):
        raise ValueError("Source-related rows cross partition boundaries.")
    checksum = digest(ROOT / SSD_BASE)
    out = Path("data/scale-study")
    out.mkdir(parents=True, exist_ok=True)
    records, clean_views = {}, {}
    for name, edge, chosen in (("letterbox_320", 320, training),
                                ("letterbox_512", 512, training),
                                ("letterbox_512_burst_sample", 512, subsample(training))):
        if edge not in clean_views:
            clean_views[edge] = (make_views(validation, edge), make_views(audit, edge))
        (_, val_images, val_maps), (_, audit_images, audit_maps) = clean_views[edge]
        rows, train_images, _ = make_views(chosen, edge)
        torch.manual_seed(SEED)
        rng = torch.Generator().manual_seed(SEED)
        net = model_for(edge, checksum)
        optimizer = torch.optim.AdamW(net.head.parameters(), lr=.002, weight_decay=.001)
        start = perf_counter()
        epochs = []
        for epoch in range(1, 5):
            net.train()
            net.backbone.eval()
            for ims, targets in batches(rows, train_images, rng, True, edge=edge):
                losses = net(ims, targets)
                optimizer.zero_grad()
                sum(losses.values()).backward()
                torch.nn.utils.clip_grad_norm_(net.head.parameters(), 10.)
                optimizer.step()
            predictions = inverse(val_maps, predict(net, val_images, edge=edge), edge)
            cutoff = best_threshold(validation, predictions)
            scores = counters(evaluate(validation, predictions, cutoff))
            row = dict(epoch=epoch, **scores, cutoff=cutoff)
            epochs.append(row)
            if max(epochs, key=rank) is row:
                torch.save(net.state_dict(), out / f"{name}.pth")
            print(name, epoch, f"F1={row['f1']:.3f}", flush=True)
        best = max(epochs, key=rank)
        path = out / f"{name}.pth"
        net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        predictions = inverse(audit_maps, predict(net, audit_images, edge=edge), edge)
        records[name] = dict(edge=edge, training_images=len(chosen), epochs=epochs, selected=best,
                            historical_audit=counters(evaluate(audit, predictions, best["cutoff"])),
                            sha256=digest(path), checkpoint=str(path), seconds=perf_counter()-start)
        print(name, "historical audit", records[name]["historical_audit"], flush=True)
        write_json(ART / "scale_study.json", dict(base_sha256=checksum, seed=SEED, trials=records,
            scope="All partitions were previously consulted; exploratory comparison only."))


if __name__ == "__main__":
    main()
