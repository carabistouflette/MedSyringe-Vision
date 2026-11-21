"""Import a provenance-verified Zenodo training sample and compare matched updates."""
import argparse
from hashlib import file_digest, sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image
import torch
import yaml

from medsyringe.data import dhash, digest, leakage, read_json, valid_box, write_json
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.supervised import detector, predict
from scripts.additive_study import FRESH
from scripts.optimize_detector import counters
from scripts.supervised_study import ART, ROOT, SEED, batches, prepare, rank, sets

ARCHIVE = Path("data/zenodo/phlebotomy.zip")
EXPECTED_MD5 = "58dd8f06c95025693b33efe794848175"
DOI = "https://doi.org/10.5281/zenodo.16924786"
MANIFEST = ART / "zenodo_training_manifest.json"


def syringe_boxes(text, class_id):
    objects = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or int(parts[0]) != class_id:
            continue
        points = [float(x) for x in parts[1:]]
        if len(points) < 6 or len(points) % 2 or not all(0 <= x <= 1 for x in points):
            raise ValueError("Malformed syringe polygon.")
        x, y = points[0::2], points[1::2]
        box = [min(x), min(y), max(x), max(y)]
        if valid_box(box):
            objects.append(dict(bbox=box))
    return objects


def prepare_public(blocked):
    with ARCHIVE.open("rb") as file:
        if file_digest(file, "md5").hexdigest() != EXPECTED_MD5:
            raise ValueError("The archive differs from the published Zenodo checksum.")
    archive_hash = digest(ARCHIVE)
    kept, hashes, parents = [], set(), set()
    known = [int(r["dhash"], 16) for r in blocked if r.get("dhash")]
    by_label = [0, 0]
    targets = [40, 120]
    skipped = 0
    with ZipFile(ARCHIVE) as archive:
        members = archive.namelist()
        config_file = next(n for n in members if PurePosixPath(n).name == "data.yaml")
        config = yaml.safe_load(archive.read(config_file))
        names = config["names"]
        classes = names.items() if isinstance(names, dict) else enumerate(names)
        class_id = int(next(i for i, name in classes if name.strip().casefold() == "syringe"))
        images = [n for n in members if PurePosixPath(n).parent.parts[-2:] == ("train", "images")
                  and PurePosixPath(n).suffix.lower() in (".jpg", ".jpeg", ".png")]
        member_set = set(members)
        for name in sorted(images, key=lambda n: sha256((str(SEED)+n).encode()).digest()):
            path = PurePosixPath(name)
            label = str(path.parent.parent / "labels" / (path.stem+".txt"))
            if label not in member_set:
                continue
            objects = syringe_boxes(archive.read(label).decode(), class_id)
            label_id = int(bool(objects))
            if by_label[label_id] == targets[label_id]:
                continue
            parent = path.stem.split(".rf.")[0]
            if parent in parents:
                continue
            data = archive.read(name)
            with Image.open(BytesIO(data)) as im:
                width, height = im.size
                key = dhash(im)
            checksum, visual = sha256(data).hexdigest(), int(key, 16)
            if checksum in hashes or any((visual^other).bit_count() <= 4 for other in known):
                skipped += 1
                continue
            identifier = f"zenodo-{checksum[:16]}"
            relative = Path("zenodo/sample") / (identifier + path.suffix.lower())
            target = ROOT / "data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            kept.append(dict(id=identifier, group="zenodo:16924786", split="public-training",
                             image=str(relative), sha256=checksum, dhash=key, license="CC BY 4.0",
                             objects=objects, width=width, height=height, member=name))
            by_label[label_id] += 1
            hashes.add(checksum)
            known.append(visual)
            parents.add(parent)
            if by_label == targets:
                break
    write_json(MANIFEST, dict(rows=kept, doi=DOI, archive_sha256=archive_hash,
                             archive_train_images=len(images), class_id=class_id,
                             positives=by_label[1], negatives=by_label[0],
                             excluded_near_hash_matches=skipped))
    print("Zenodo training-only sample", by_label, "of", len(images), "images", flush=True)
    if min(by_label) < 8:
        raise ValueError("Insufficient labeled data for a controlled positive/negative experiment.")
    return kept


def train(rows, control, public, rounds, use_external):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    net = detector(ROOT, empty_weight=1.0)
    initial = read_json(ART / "supervised_study.json")
    checkpoint = Path(initial["checkpoint"])
    if digest(checkpoint) != initial["checkpoint_sha256"]:
        raise ValueError("Starting SSD checkpoint differs from the frozen study.")
    net.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    net.train()
    net.backbone.eval()
    optimizer = torch.optim.AdamW(net.head.parameters(), lr=.0005, weight_decay=.001)
    loss_history = []
    for j in range(rounds):
        indices = torch.randint(len(control), (4,), generator=rng).tolist()
        foreign = torch.randint(len(public), (2,), generator=rng).tolist()
        if use_external:
            indices[-2:] = [len(control) + i for i in foreign]
        images = (control + public)
        chosen = [images[i] for i in indices]
        image_tensors, targets = next(batches([rows[i] for i in indices], chosen, rng, True))
        optimizer.zero_grad()
        loss = sum(net(image_tensors, targets).values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.head.parameters(), 10.)
        optimizer.step()
        loss_history.append(float(loss.detach()))
        if (j+1) % 25 == 0:
            print("Transfer" if use_external else "Control", j+1, "/", rounds, flush=True)
    return net, loss_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    old, calibration, historical = sets()
    audited = read_json(FRESH)["rows"]
    seen = read_json(ART / "supervised_external_manifest.json")["rows"]
    blocked = [*old, *calibration, *historical, *audited, *seen]
    public_rows = prepare_public(blocked)
    if leakage([*blocked, *public_rows]):
        raise ValueError("A public training image collides with a reference group.")
    # Match training update count, starting weights, exposure jitter and optimizer.
    control, public = prepare(old), prepare(public_rows)
    val, check = prepare(calibration), prepare(audited)
    results = {}
    for tag, mix in (("continued_existing_data", False), ("half_public_data", True)):
        net, losses = train([*old, *public_rows], control, public, args.steps, mix)
        predictions = predict(net, val)
        cutoff = best_threshold(calibration, predictions)
        record = dict(cutoff=cutoff, calibration=counters(evaluate(calibration, predictions, cutoff)),
                      external_regression=counters(evaluate(audited, predict(net, check), cutoff)),
                      losses=losses)
        path = Path("data/zenodo") / f"{tag}.pth"
        torch.save(net.state_dict(), path)
        record.update(checkpoint=str(path), checkpoint_sha256=digest(path))
        results[tag] = record
        print(tag, record["calibration"], record["external_regression"], flush=True)
    write_json(ART / "public_transfer_study.json", dict(results=results, updates=args.steps,
                 sample_sha256=digest(MANIFEST), doi=DOI,
                 scope="Development and previously consulted external cohort; no independent performance claim"))


if __name__ == "__main__":
    main()
