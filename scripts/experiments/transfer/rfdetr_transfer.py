"""Prepare COCO inputs and run local RF-DETR segmentation and box transfer."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from shutil import copyfile
from time import perf_counter
from zipfile import ZipFile

from PIL import Image

from medsyringe.data import dhash, digest, leakage, read_json, valid_box, write_json
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.public_data.public_transfer_study import ARCHIVE, MANIFEST
from scripts.experiments.transfer.supervised_study import sets

ART = Path("artifacts")
WORK = Path("data/rfdetr")
BASE = WORK / "rf-detr-seg-nano.pt"
BASE_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
SEED = 2602
PUBLIC_VALID = {"0005", "0012", "0037"}
ANNOTATIONS = "_annotations.coco.json"
AUDIT = ART / "rfdetr_audit_manifest.json"


def public_clip(member):
    match = re.match(r"frame_(\d+)_", PurePosixPath(member).name)
    if not match:
        raise ValueError(f"Could not establish the source clip of {member}")
    return match[1]


def polygons(contents, category):
    for line in contents.splitlines():
        fields = line.split()
        if not fields or int(fields[0]) != category:
            continue
        points = [float(x) for x in fields[1:]]
        x, y = points[::2], points[1::2]
        if len(points) < 6 or len(points) % 2 or not all(0 <= p <= 1 for p in points):
            raise ValueError("Invalid normalized polygon")
        vertices = list(zip(x, y))
        area = abs(sum(a*d-b*c for (a, b), (c, d) in
                       zip(vertices, vertices[1:] + vertices[:1]))) / 2
        bbox = [min(x), min(y), max(x), max(y)]
        if not area or not valid_box(bbox):
            raise ValueError("Degenerate polygon")
        yield dict(bbox=bbox, polygon=points, area=area)


def export(rows, destination):
    destination.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for image_id, row in enumerate(rows, 1):
        image = source(row)
        width, height = image.size
        name = f"{image_id:05d}.png"
        image.info.clear()
        image.save(destination / name, compress_level=1)
        images.append(dict(id=image_id, file_name=name, width=width, height=height))
        for obj in row["objects"]:
            left, top, right, bottom = obj["bbox"]
            bbox = [left*width, top*height, (right-left)*width, (bottom-top)*height]
            item = dict(id=len(annotations)+1, image_id=image_id, category_id=1,
                        bbox=bbox, area=bbox[2]*bbox[3], iscrowd=0)
            if "polygon" in obj:
                item["area"] = obj["area"]*width*height
                item["segmentation"] = [[v*(width if j%2 == 0 else height)
                                         for j, v in enumerate(obj["polygon"])]]
            annotations.append(item)
    write_json(destination / ANNOTATIONS,
               dict(images=images, annotations=annotations,
                    categories=[dict(id=1, name="syringe", supercategory="syringe")]))
    return dict(ids=[r["id"] for r in rows], groups=dict(Counter(r["group"] for r in rows)),
                images=len(images), objects=len(annotations), negatives=sum(not r["objects"] for r in rows),
                annotation_sha256=digest(destination / ANNOTATIONS))


def prepare_public():
    manifest = read_json(MANIFEST)
    if digest(ARCHIVE) != manifest["archive_sha256"]:
        raise ValueError("Zenodo archive checksum mismatch")
    splits = {"train": [], "valid": []}
    excluded = []
    with ZipFile(ARCHIVE) as archive:
        for row in manifest["rows"]:
            image = PurePosixPath(row["member"])
            try:
                clip = public_clip(image)
            except ValueError:
                excluded.append(row["id"])
                continue
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            objects = list(polygons(archive.read(str(label)).decode(), manifest["class_id"]))
            if [p["bbox"] for p in objects] != [p["bbox"] for p in row["objects"]]:
                raise ValueError(f"Converted polygons changed the box labels of {row['id']}")
            splits["valid" if clip in PUBLIC_VALID else "train"].append(
                {**row, "objects": objects, "group": "zenodo-clip-" + clip})
    if leakage([{**r, "split": split} for split, rows in splits.items() for r in rows]):
        raise ValueError("Zenodo split leakage")
    return splits, dict(archive_sha256=manifest["archive_sha256"], unknown_clip_excluded=excluded)


def prepare_adapt():
    training, validation, _ = sets()
    # Only one representative from each identifiable Fil rouge burst. All
    # Fil rouge samples used in this study still belong to a single session.
    groups = {}
    for r in training:
        key = r["objects"][0].get("burst", r["group"]) if r["objects"] else r["group"]
        groups.setdefault(key, []).append(r)
    chosen = [min(rows, key=lambda r: sha256(f"{SEED}:{r['id']}".encode()).digest())
              for rows in groups.values()]
    return {"train": chosen, "valid": validation}, {}


def prepare(stage):
    rows, provenance = (prepare_public if stage == "public" else prepare_adapt)()
    information = {key: export(split, WORK / stage / key) for key, split in rows.items()}
    write_json(ART / f"rfdetr_{stage}_data.json", dict(**provenance, **information))
    print({k: {m: v[m] for m in ("images", "objects", "negatives")}
           for k, v in information.items()}, flush=True)


def prepare_audit():
    """Reserve the untouched, clip-disjoint portion of the official test directory."""
    manifest = read_json(MANIFEST)
    if digest(ARCHIVE) != manifest["archive_sha256"]:
        raise ValueError("Zenodo archive checksum mismatch")
    clips = {public_clip(r["member"]) for r in manifest["rows"]
             if re.match(r"frame_\d+_", PurePosixPath(r["member"]).name)}
    training, calibration, historical = sets()
    consulted = [*training, *calibration, *historical, *manifest["rows"]]
    for name in ("additive_external_manifest.json", "supervised_external_manifest.json"):
        consulted.extend(read_json(ART / name)["rows"])
    known = [int(r["dhash"], 16) for r in consulted if r.get("dhash")]
    selected, parents = [], set()
    skipped = Counter()
    folder = WORK / "audit"
    folder.mkdir(exist_ok=True)
    with ZipFile(ARCHIVE) as archive:
        members = set(archive.namelist())
        for name in sorted(members, key=lambda item: sha256(f"{SEED}:{item}".encode()).digest()):
            path = PurePosixPath(name)
            if path.parent.parts != ("test", "images") or path.suffix.lower() != ".jpg":
                continue
            try:
                clip = public_clip(path)
            except ValueError:
                skipped["unknown_clip"] += 1
                continue
            if clip in clips:
                skipped["consulted_clip"] += 1
                continue
            parent = path.stem.split(".rf.")[0]
            if parent in parents:
                skipped["same_source_frame"] += 1
                continue
            label = str(path.parent.parent / "labels" / (path.stem + ".txt"))
            if label not in members:
                skipped["missing_label"] += 1
                continue
            contents = archive.read(name)
            with Image.open(BytesIO(contents)) as image:
                visual = dhash(image)
                width, height = image.size
            if any((int(visual, 16) ^ old).bit_count() <= 4 for old in known):
                skipped["near_hash"] += 1
                continue
            checksum = sha256(contents).hexdigest()
            identifier = f"zenodo-audit-{checksum[:16]}"
            target = folder / (identifier + ".jpg")
            target.write_bytes(contents)
            objects = list(polygons(archive.read(label).decode(), manifest["class_id"]))
            selected.append(dict(id=identifier, image=str(target.relative_to("data")),
                                 member=name, width=width, height=height, sha256=checksum,
                                 dhash=visual, group="zenodo-clip-" + clip, split="rfdetr-audit",
                                 license="CC BY 4.0", objects=objects))
            parents.add(parent)
            known.append(int(visual, 16))
    if leakage([*[{**r, "split": "consulted"} for r in consulted], *selected]):
        raise ValueError("Overlap between the reserved frames and consulted images")
    if not any(r["objects"] for r in selected) or not any(not r["objects"] for r in selected):
        raise ValueError("The audit requires both positive and negative frames")
    write_json(AUDIT, dict(rows=selected, excluded=dict(skipped),
                          archive_sha256=manifest["archive_sha256"],
                          training_manifest_sha256=digest(MANIFEST), seed=SEED))
    print(len(selected), "reserved", sum(bool(r["objects"]) for r in selected), "positives",
          len({r["group"] for r in selected}), "clips", dict(skipped), flush=True)


def train(run, stage, epochs, resolution, weights, resume):
    import torch
    from rfdetr import RFDETRSegNano

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    if weights == BASE and digest(BASE) != BASE_SHA:
        raise ValueError("Unexpected publisher checkpoint SHA-256")
    destination = WORK / "runs" / run
    data = ART / f"rfdetr_{stage}_data.json"
    manifest = read_json(data)
    for key in ("train", "valid"):
        if digest(WORK / stage / key / ANNOTATIONS) != manifest[key]["annotation_sha256"]:
            raise ValueError(f"The frozen {stage}/{key} annotations changed")
    segment = stage == "public"
    args = dict(device="cpu", pretrain_weights=str(weights), segmentation_head=segment,
                freeze_encoder=True, num_classes=1)
    config = dict(dataset_dir=str(WORK / stage), output_dir=str(destination), device="cpu",
                  epochs=epochs, batch_size=4, lr=0.0002, lr_encoder=0,
                  multi_scale=False, num_workers=0, checkpoint_interval=1, use_ema=False,
                  tensorboard=False, progress_bar="tqdm", save_dataset_grids=False,
                  run_test=False, num_sanity_val_steps=0, resolution=resolution, seed=SEED,
                  train_log_on_step=True)
    if resume:
        config["resume"] = str(resume)
    provenance = dict(model=args, train=config, data_sha256=digest(data),
                      initial_sha256=digest(weights))
    record = ART / f"rfdetr_{run}_training.json"
    write_json(record, dict(status="started", **provenance))
    started = perf_counter()
    model = RFDETRSegNano(**args)
    model.train(**config)
    checkpoint = destination / "checkpoint_best_total.pth"
    copyfile(destination / "metrics.csv", ART / f"rfdetr_{run}_metrics.csv")
    write_json(record, dict(status="completed", **provenance, elapsed_seconds=perf_counter()-started,
                            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train"])
    parser.add_argument("stage", choices=["public", "adapt", "audit"])
    parser.add_argument("--run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=240)
    parser.add_argument("--weights", type=Path, default=BASE)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.stage == "audit":
            prepare_audit()
        else:
            prepare(args.stage)
    elif args.stage == "audit":
        parser.error("The reserved audit can only be prepared.")
    else:
        train(args.run or args.stage, args.stage, args.epochs, args.resolution, args.weights, args.resume)


if __name__ == "__main__":
    main()
