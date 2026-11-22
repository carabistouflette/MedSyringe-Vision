"""Measure a fixed local continuation of the best mixed-supervision RF-DETR run."""
import argparse
from csv import DictReader
from pathlib import Path
from shutil import copyfile
from statistics import median
from time import perf_counter

from medsyringe.detection.additive import overlapping_tiles, project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import best_threshold, evaluate
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.transfer.detection_real_box_evaluate import training
from scripts.experiments.transfer.detection_real_box_transfer import ANNOTATIONS, DATASET, DESIGN, EDGE, SEED, benchmark_sources
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_compare import ranked_region
from scripts.experiments.expanded.sam3_study import configure_cpu

ARM = "mix-simulator"
RECORD = ART / "reviewed_box_continuation.json"
EVALUATION = ART / "reviewed_box_continuation_development.json"
CURVE = ART / "reviewed_box_continuation_metrics.csv"
RAW = DATA / "proposals" / "box-continuation-review.json"
RUN = DATA / "runs" / "box-transfer-continuation"


def train(resume):
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    previous = training(ARM)
    design = read_json(DESIGN)
    train_root = DATASET / ARM
    if RECORD.exists() and read_json(RECORD)["status"] == "completed":
        raise ValueError("This continuation is already complete.")
    if (digest(train_root/"train"/ANNOTATIONS) != design["training"][ARM]["sha256"]
        or digest(train_root/"valid"/ANNOTATIONS) != design["development"]["sha256"]):
        raise ValueError("The locked box-supervised inputs changed.")
    config = dict(previous["configuration"], output_dir=str(RUN), epochs=2, lr=.000075)
    if resume:
        config["resume"] = str(resume)
    write_json(RECORD, dict(status="started", initial_sha256=previous["checkpoint_sha256"],
                           dataset_sha256=digest(DESIGN), config=config, implementation=digest(__file__)))
    detector = RFDETRNano(device="cpu", pretrain_weights=previous["checkpoint"],
                          num_classes=1, freeze_encoder=True)
    started = perf_counter()
    detector.train(**config)
    output = RUN / "checkpoint_best_total.pth"
    copyfile(RUN / "metrics.csv", CURVE)
    write_json(RECORD, dict(**{k: v for k, v in read_json(RECORD).items() if k != "status"},
                           status="completed", checkpoint=str(output), checkpoint_sha256=digest(output),
                           cpu_seconds=perf_counter()-started))
    print("Completed fixed-budget transfer continuation.", flush=True)


def evaluate_boxes():
    from rfdetr import RFDETRNano

    configure_cpu()
    state = read_json(RECORD)
    if state["status"] != "completed" or digest(state["checkpoint"]) != state["checkpoint_sha256"]:
        raise ValueError("The continuation checkpoint needs to be complete and verified.")
    _, _, rows = benchmark_sources()
    identity = dict(weights=state["checkpoint_sha256"], source=digest(__file__),
                    images={r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows})
    cache = read_json(RAW) if RAW.exists() else dict(key=identity, frames={})
    if cache["key"] != identity or EVALUATION.exists():
        raise ValueError("Expected a new evaluation or the same resumable cached inputs.")
    remaining = [r for r in rows if r["id"] not in cache["frames"]]
    if remaining:
        model = RFDETRNano.from_checkpoint(state["checkpoint"], device="cpu")
        for row in remaining:
            img = source(row)
            start = perf_counter()
            whole = suppress(list(ranked_region(model, img, EDGE)), .45)
            mapped = []
            for window in overlapping_tiles(img.size, .66):
                for value in ranked_region(model, img.crop(window), EDGE):
                    if value["score"] >= .008 and (box := project(value["bbox"], window, img.size)):
                        mapped.append({**value, "bbox": box})
            cache["frames"][row["id"]] = dict(whole=whole, wide=suppress([*whole,*mapped], .45),
                                             seconds=perf_counter()-start)
            write_json(RAW, cache)
            print("Continuation development", len(cache["frames"]), "/", len(rows), flush=True)
    domains = {k: [r for r in rows if r["domain"] == k] for k in sorted({r["domain"] for r in rows})}
    outcomes = {}
    for view in ("whole", "wide"):
        predictions = {k: v[view] for k, v in cache["frames"].items()}
        chosen = best_threshold(rows, predictions)
        outcomes[view] = dict(dev_threshold=chosen, scores={
            name: dict(fixed=.20, standard=short(evaluate(refs, predictions, .20)),
                       descriptive_upper_bound=short(evaluate(refs, predictions, chosen)))
            for name, refs in domains.items()})
    with CURVE.open(newline="") as file:
        fields = ("mAP_50", "mAP_50_95", "F1", "precision", "recall")
        epochs = [{k: float(r[f"val/{k}"]) for k in fields} for r in DictReader(file) if r["val/F1"]]
    write_json(EVALUATION, dict(checkpoint_sha256=state["checkpoint_sha256"], data_sha256=digest(DESIGN),
                                result=outcomes, validation_curves=epochs, cached_output_sha256=digest(RAW),
                                median_cpu_seconds=median(f["seconds"] for f in cache["frames"].values()),
                                scope="Two additional epochs with a fresh optimizer at half the initial learning rate. "
                                      "Whole and fixed tiled inference on the existing development data."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.action == "train":
        train(args.resume)
    else:
        evaluate_boxes()
