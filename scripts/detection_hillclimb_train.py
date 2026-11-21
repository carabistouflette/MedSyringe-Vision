"""Run a fixed-budget RF-DETR transfer experiment on local archived masks."""
import argparse
from pathlib import Path
from shutil import copyfile
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from scripts.detection_hillclimb_corpus import ART, COCO, DATA, LOCK, SEED
from scripts.rfdetr_transfer import ANNOTATIONS

INITIAL = Path("data/rfdetr/runs/public/checkpoint_best_total.pth")
INITIAL_RECORD = Path("artifacts/rfdetr_public_training.json")


def run(arm, epochs, resume):
    import torch
    from rfdetr import RFDETRSegNano

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    key = read_json(COCO)
    if key["corpus_sha256"] != digest(LOCK):
        raise ValueError("The fixed training design changed.")
    if read_json(INITIAL_RECORD)["checkpoint_sha256"] != digest(INITIAL):
        raise ValueError("The initial model differs from the archived training checkpoint.")
    if any(digest(DATA / arm / role / ANNOTATIONS) != key[role]["annotation_sha256"]
           for role in ("train", "valid")):
        raise ValueError("The trainer's COCO files differ from their manifest.")
    output = DATA / "runs" / arm
    config = dict(dataset_dir=str(DATA/arm), output_dir=str(output), device="cpu", epochs=epochs,
                  batch_size=4, lr=.00012, lr_encoder=0, multi_scale=False, num_workers=0,
                  checkpoint_interval=1, use_ema=False, tensorboard=False, progress_bar=None,
                  save_dataset_grids=False, run_test=False, num_sanity_val_steps=0,
                  resolution=240, seed=SEED, train_log_on_step=False)
    if resume:
        config["resume"] = str(resume)
    initialization = dict(device="cpu", pretrain_weights=str(INITIAL), segmentation_head=True,
                          freeze_encoder=True, num_classes=1)
    record = ART / f"rfdetr_{arm}_training.json"
    report = dict(model=initialization, train=config, data_sha256=digest(COCO),
                  initial_sha256=digest(INITIAL))
    write_json(record, dict(status="started", **report))
    started = perf_counter()
    model = RFDETRSegNano(**initialization)
    try:
        model.train(**config)
    except Exception as exc:
        write_json(record, dict(status="failed", elapsed_seconds=perf_counter()-started,
                               error=f"{type(exc).__name__}: {exc}", **report))
        raise
    checkpoint = output / "checkpoint_best_total.pth"
    copyfile(output / "metrics.csv", ART / f"rfdetr_{arm}_metrics.csv")
    write_json(record, dict(status="completed", elapsed_seconds=perf_counter()-started,
              checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), **report))
    print("Completed", arm, checkpoint, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("whole-frame",), default="whole-frame")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.arm, args.epochs, args.resume)
