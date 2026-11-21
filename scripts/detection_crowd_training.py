"""Test matched RF-DETR training with separated and overlapping reference boxes."""
import argparse
from itertools import combinations
from pathlib import Path
from random import Random
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

from PIL import Image

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_parallel_development import OUTPUT as STRESS, isolated
from scripts.detection_real_box_transfer import ANNOTATIONS, DESIGN, EDGE, benchmark_sources, coco
from scripts.detection_box_continuation import RECORD as INITIAL

FOLDER = DATA / "crowd-training"
PLAN = ART / "crowd_training_design.json"
ARMS = ("separated", "overlapping")
SEED = 20261122
SIDE = 576
COUNTS = (2, 3, 4) * 8


def render(patches, background, angle):
    scene = Image.new("RGBA", (SIDE, SIDE))
    labels = []
    for source, foreground, mask, (x, y) in patches:
        scene.alpha_composite(foreground, (x, y))
        full = Image.new("L", scene.size)
        full.paste(mask, (x, y))
        box = full.rotate(angle, Image.Resampling.NEAREST).getbbox()
        if not box or min(box[:2]) <= 0 or max(box[2:]) >= SIDE:
            raise ValueError("A rendered syringe fell outside the reference raster.")
        labels.append(dict(kind="syringe", bbox=[coord / SIDE for coord in box],
                           parent=source["id"], parent_group=source["group"],
                           parent_sha256=source["sha256"]))
    rotated = scene.rotate(angle, Image.Resampling.BICUBIC)
    picture = Image.new("RGB", scene.size, (background,) * 3)
    picture.paste(rotated.convert("RGB"), (0, 0), rotated.getchannel("A"))
    maximum = max(box_iou(a["bbox"], b["bbox"]) for a, b in combinations(labels, 2))
    return picture, labels, maximum


def scene_pair(pool, count, rng):
    for _ in range(100):
        # Each parent group contributes at most one instance within a composite.
        items = [rng.choice(pool[group]) for group in rng.sample(sorted(pool), count)]
        resized = []
        for item in items:
            width, height = item["foreground"].size
            ratio = min(2.6, rng.uniform(240, 322) / width)
            size = (round(width * ratio), round(height * ratio))
            resized.append((item["source"], item["foreground"].resize(size, Image.Resampling.BICUBIC),
                            item["mask"].resize(size, Image.Resampling.NEAREST)))
        cursor = (SIDE - sum(item[1].height for item in resized) - 7 * (count - 1)) // 2
        patches = []
        for row, foreground, mask in resized:
            patches.append((row, foreground, mask, ((SIDE - foreground.width) // 2 + rng.randint(-12, 12),
                                                     cursor)))
            cursor += foreground.height + 7
        lum = rng.choice((132, 158, 181, 207, 227))
        angle = rng.choice((-56, -42, -29, 27, 39, 52)) + rng.uniform(-2, 2)
        control = render(patches, lum, 0.)
        stress = render(patches, lum, angle)
        if stress[2] >= .42:
            return (control, stress)
    raise ValueError("The training generator could not meet the fixed overlap requirement.")


def build():
    if PLAN.exists() or FOLDER.exists():
        raise ValueError("The crowd training comparison is already prepared.")
    real, simulated, development = benchmark_sources()
    known = {row["id"]: row for row in [*real, *simulated]}
    base = [known[item["id"]] for item in read_json(DESIGN)["arms"]["mix-simulator"]]
    stress = read_json(STRESS)["rows"]
    blocked = {r["group"] for r in development}
    blocked.update(obj["source_group"] for row in stress for obj in row["objects"])
    pool = {}
    for row in simulated:
        if row["group"] not in blocked:
            for label in row["objects"]:
                if item := isolated(row, label):
                    pool.setdefault(row["group"], []).append(item)
    if len(pool) < max(COUNTS):
        raise ValueError("Too few independent simulator fit groups to make paired compositions.")
    rng = Random(SEED)
    design = {arm: [] for arm in ARMS}
    reports = {}
    with TemporaryDirectory(prefix="medsyringe-crowd-training-", dir=DATA) as temporary:
        root = Path(temporary) / "prepared"
        root.mkdir()
        insertions = {arm: {} for arm in ARMS}
        for index, count in enumerate(COUNTS, start=1):
            pair = scene_pair(pool, count, rng)
            for arm, (picture, labels, overlap) in zip(ARMS, pair, strict=True):
                identifier = f"paired-crowd-{arm}-{index:03d}"
                # Pass the generated raster to the shared COCO exporter as a verified local dependency.
                path = root / f"{identifier}.png"
                picture.save(path)
                row = dict(id=identifier, sha256=digest(path), objects=labels, max_iou=overlap)
                design[arm].append(row)
                insertions[arm][identifier] = (picture.size, path)
        order = list(range(len(base) + len(COUNTS)))
        rng.shuffle(order)
        shared = {}
        validation = coco(development, root / "validation", shared)
        for arm in ARMS:
            joined = base + design[arm]
            assets = {**shared, **insertions[arm]}
            # Share base rasters between arms irrespective of synthetic placement.
            reports[arm] = coco([joined[i] for i in order], root / arm / "train", assets)
            shared.update({key: assets[key] for key in (row["id"] for row in base)})
            (root / arm / "valid").symlink_to(Path("../validation"), target_is_directory=True)
            for _, path in insertions[arm].values():
                path.unlink()
        keys = ("images", "targets", "negatives")
        if len({tuple(details[k] for k in keys) for details in reports.values()}) != 1:
            raise ValueError("Training image, target or negative budgets differ.")
        root.rename(FOLDER)
    write_json(PLAN, dict(seed=SEED, base_sha256=digest(DESIGN), validation=validation,
                          stress_sha256=digest(STRESS), implementation=digest(__file__),
                          order=order, composites=design, training=reports,
                          protocol="Each added pair uses the same fit-only masked devices, positions and background. "
                                   "One raster is rendered horizontally and the other with diagonal box overlap. "
                                   "Both adapt the same checkpoint with equal computation and use the same validation. "
                                   "Separate public and original user reserves do not participate in training."))
    print("Paired crowd training prepared", reports, flush=True)


def train(arm, resume=False):
    import torch
    from rfdetr import RFDETRNano

    plan = read_json(PLAN)
    if plan["base_sha256"] != digest(DESIGN) or plan["stress_sha256"] != digest(STRESS):
        raise ValueError("The matched input design changed.")
    source = read_json(INITIAL)
    if source["status"] != "completed" or digest(source["checkpoint"]) != source["checkpoint_sha256"]:
        raise ValueError("The shared initialization checkpoint is incomplete or changed.")
    data = FOLDER / arm
    if any(digest(data / split / ANNOTATIONS) != reference["sha256"]
           for split, reference in (("train", plan["training"][arm]), ("valid", plan["validation"]))):
        raise ValueError("Prepared COCO labels changed.")
    record = ART / f"crowd_training_{arm}.json"
    previous = read_json(record) if record.exists() else None
    if previous and previous["status"] == "completed":
        raise ValueError("This matched arm is already complete.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    output = DATA / "runs" / f"crowd-training-{arm}"
    config = dict(dataset_dir=str(data), output_dir=str(output), device="cpu", epochs=2,
                  batch_size=4, resolution=EDGE, seed=SEED, lr=.000075, lr_encoder=0,
                  num_workers=0, num_sanity_val_steps=0, compute_val_loss=True, use_ema=False,
                  multi_scale=False, save_dataset_grids=False, run_test=False, train_log_on_step=False)
    provenance = dict(initial_sha256=source["checkpoint_sha256"], data_sha256=digest(PLAN),
                      code_sha256=digest(__file__), config=config)
    if resume:
        if not previous or any(previous[k] != provenance[k] for k in ("initial_sha256", "data_sha256", "config")):
            raise ValueError("Resume requires the same initialization, training data and update budget.")
        saved = output / "last.ckpt"
        state = torch.load(saved, map_location="cpu", weights_only=False)
        completed = int(state["epoch"]) + 1
        if not state["optimizer_states"] or not 0 < completed < config["epochs"]:
            raise ValueError("The local checkpoint cannot restore a partial training run.")
        del state
        elapsed = previous.get("resume_history", [])
        preserved_curve = ART / f"crowd_training_{arm}_before_resume_{len(elapsed) + 1}.csv"
        copyfile(output / "metrics.csv", preserved_curve)
        provenance["resume_history"] = [*elapsed, dict(
            completed_epochs=completed, checkpoint_sha256=digest(saved),
            previous_code_sha256=previous["code_sha256"], curve=str(preserved_curve),
            curve_sha256=digest(preserved_curve))]
    elif previous:
        raise ValueError("This arm already started. Use --resume to preserve its updates.")
    write_json(record, dict(status="started", **provenance))
    model = RFDETRNano(device="cpu", pretrain_weights=source["checkpoint"], resolution=EDGE,
                       num_classes=1, freeze_encoder=True)
    start = perf_counter()
    model.train(**config, **({"resume": str(saved)} if resume else {}))
    checkpoint = output / "checkpoint_best_total.pth"
    copyfile(output / "metrics.csv", ART / f"crowd_training_{arm}_metrics.csv")
    write_json(record, dict(status="completed", checkpoint=str(checkpoint), seconds=perf_counter() - start,
                            checkpoint_sha256=digest(checkpoint), **provenance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "train"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        if args.resume:
            parser.error("--resume requires a training arm.")
        build()
    elif args.arm:
        train(args.arm, args.resume)
    else:
        parser.error("Select a matched training arm.")
