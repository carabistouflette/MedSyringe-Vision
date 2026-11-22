"""Train matched public-only checkpoints with and without dataset augmentation."""
import argparse
from pathlib import Path
from random import Random
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

from medsyringe.data import digest, leakage, read_json, write_json
from scripts.experiments.detector_training.detection_context_study import inputs as transparent
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.transfer.detection_real_box_transfer import ANNOTATIONS, EDGE, INITIAL, benchmark_sources, checkpoint, coco
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.public_data.public_fit_augmentation import FAMILIES, derivative
from scripts.experiments.public_data.public_training_intake import FIL_ROUGE, close, raster_hashes, references

ARMS = ("control", *FAMILIES)
PLAN = ART / "public_augmentation_design.json"
FOLDER = DATA / "public-augmentation"
SEED = 20261124
EXTRA = 40


def inputs():
    fit, _, earlier = benchmark_sources()
    validation = [r for r in earlier if r["domain"] != "phlebotomy-development"] + transparent()
    if leakage([*({**r, "split": "fit"} for r in fit),
                *({**r, "split": "development"} for r in validation)]):
        raise ValueError("Public training and real development share an acquisition.")
    _, forbidden, differences = references()
    for row in fit:
        path = Path("data") / row["image"]
        if (row["sha256"] in forbidden or row.get("derived_sha256") in forbidden
                or close(raster_hashes(path), differences)):
            raise ValueError(f"A public training parent matches protected Fil Rouge evidence: {row['id']}.")
    return [{**r, "split": "public-fit"} for r in fit], validation


def build():
    if PLAN.exists() or FOLDER.exists():
        raise ValueError("The matched augmentation inputs already exist.")
    fit, validation = inputs()
    rand = Random(SEED)
    order = list(fit)
    rand.shuffle(order)
    parents, seen = [], set()
    for item in order:
        group = item["group"]
        if group not in seen:
            seen.add(group)
            parents.append(item)
        if len(parents) == EXTRA:
            break
    if len(parents) != EXTRA:
        raise ValueError("Insufficient distinct acquisitions for the fixed augmentation budget.")
    with TemporaryDirectory(prefix="public-augmentation-", dir=DATA) as temporary:
        root = Path(temporary) / "prepared"
        root.mkdir()
        shared = {}
        validation_record = coco(validation, root / "validation", shared)
        additions = {arm: [] for arm in ARMS}
        staged = {arm: {} for arm in ARMS}
        for index, row in enumerate(parents):
            picture = source(row)
            # Match parents and accepted transforms across all arms. A failed crop never drops the label.
            for attempt in range(15):
                seed = SEED + 100 * index + attempt
                generated = {family: derivative(picture, row, family, seed) for family in FAMILIES}
                if all(result is not None for result in generated.values()):
                    break
            else:
                raise ValueError(f"No valid seeded transform for {row['id']}.")
            additions["control"].append(row)
            for arm, (image, objects, metadata) in generated.items():
                key = f"{row['id']}--{arm}"
                path = root / (key + ".png")
                image.save(path)
                staged[arm][key] = (image.size, path)
                additions[arm].append(dict(id=key, parent=row["id"], group=row["group"], sha256=digest(path),
                                           objects=objects, **metadata))
        results = {}
        positions = list(range(len(fit) + EXTRA))
        rand.shuffle(positions)
        for arm in ARMS:
            all_rows = fit + additions[arm]
            assets = {**shared, **staged[arm]}
            results[arm] = coco([all_rows[i] for i in positions], root / arm / "train", assets)
            shared.update({row["id"]: assets[row["id"]] for row in fit})
            (root / arm / "valid").symlink_to(Path("../validation"), target_is_directory=True)
            for _, path in staged[arm].values():
                path.unlink()
        if len({(values["images"], values["targets"], values["negatives"]) for values in results.values()}) != 1:
            raise ValueError("Arm image, object and negative budgets differ.")
        root.rename(FOLDER)
    write_json(PLAN, dict(seed=SEED, arms=results, validation=validation_record,
                          fit_images={r["id"]: dict(hash=r["sha256"], group=r["group"]) for r in fit},
                          fit_derivatives={k: v for k, v in additions.items() if k != "control"},
                          protected_fil_rouge={str(p): digest(p) for p in FIL_ROUGE},
                          labels="Only reviewed syringe boxes are exported for this controlled training trial.",
                          implementation={str(p): digest(p) for p in (__file__, "scripts/experiments/public_data/public_fit_augmentation.py")},
                          protocol="Fit sources are public photographs only. Each arm adds one view per selected "
                                   "acquisition with the same annotation budget. Validation uses different public "
                                   "photographs and retains their original image bytes."))
    print("Prepared public augmentation comparison", results, flush=True)


def train(arm):
    import torch
    from rfdetr import RFDETRNano

    metadata = read_json(PLAN)
    if any(digest(p) != d for p, d in metadata["protected_fil_rouge"].items()):
        raise ValueError("Protected Fil Rouge evidence changed.")
    dataset = FOLDER / arm
    if (digest(dataset / "train" / ANNOTATIONS) != metadata["arms"][arm]["sha256"]
            or digest(dataset / "valid" / ANNOTATIONS) != metadata["validation"]["sha256"]):
        raise ValueError("Prepared training or validation labels changed.")
    proof = ART / f"public_augmentation_{arm}.json"
    if proof.exists() and read_json(proof)["status"] == "completed":
        raise ValueError("This augmentation arm already finished.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    work = DATA / "runs" / f"public-augmentation-{arm}"
    cfg = dict(dataset_dir=str(dataset), output_dir=str(work), device="cpu", epochs=1,
               batch_size=4, resolution=EDGE, seed=SEED, lr=.00015, lr_encoder=0,
               num_workers=0, num_sanity_val_steps=0, compute_val_loss=True, use_ema=False,
               multi_scale=False, save_dataset_grids=False, run_test=False, progress_bar=None, train_log_on_step=False)
    binding = dict(pretrained_sha256=checkpoint(), design_sha256=digest(PLAN), config=cfg)
    write_json(proof, dict(status="started", **binding))
    detector = RFDETRNano(device="cpu", pretrain_weights=str(INITIAL), num_classes=1, freeze_encoder=True)
    begin = perf_counter()
    detector.train(**cfg)
    model = work / "checkpoint_best_total.pth"
    copyfile(work / "metrics.csv", ART / f"public_augmentation_{arm}_metrics.csv")
    write_json(proof, dict(status="completed", checkpoint=str(model), checkpoint_sha256=digest(model),
                           training_seconds=perf_counter() - begin, **binding))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "train"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "build":
        build()
    elif args.arm:
        train(args.arm)
    else:
        parser.error("Select a fixed augmentation arm.")
