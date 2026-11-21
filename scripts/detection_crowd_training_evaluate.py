"""Compare three RF-DETR checkpoints on aligned and overlapping development scenes."""
import argparse
from statistics import median
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import evaluate
from scripts.detection_crowd_training import ARMS, EDGE, INITIAL, PLAN
from scripts.detection_deskew_study import orientation, restored_box
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_mosaic_development import MANIFEST as MOSAICS
from scripts.detection_parallel_development import OUTPUT as PARALLEL
from scripts.detection_real_box_transfer import benchmark_sources
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.rfdetr_compare import ranked_region
from scripts.sam3_study import configure_cpu
from PIL import Image

MODELS = ("initial", *ARMS)
VIEWS = ("ranked", "suppressed", "deskewed")
FLOOR = .20
OUTPUT = ART / "crowd_training_development_results.json"


def records():
    _, _, ordinary = benchmark_sources()
    return dict(ordinary=ordinary, mosaic=read_json(MOSAICS)["rows"],
                parallel=read_json(PARALLEL)["rows"])


def state(model_name, rows):
    file = INITIAL if model_name == "initial" else ART / f"crowd_training_{model_name}.json"
    checkpoint = read_json(file)
    if checkpoint["status"] != "completed" or digest(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
        raise ValueError(f"Invalid trained checkpoint for {model_name}.")
    if model_name != "initial" and digest(PLAN) != checkpoint["data_sha256"]:
        raise ValueError("The checkpoint used different training references.")
    return checkpoint["checkpoint"], dict(
        checkpoint=checkpoint["checkpoint_sha256"], manifests={k: digest(p) for k, p in
                    (("mosaic", MOSAICS), ("parallel", PARALLEL), ("matched", PLAN))},
        input_hashes={r["id"]: r["sha256"] for cohort in rows.values() for r in cohort},
        source_sha256=digest(__file__)
    )


def path(model_name):
    return DATA / "proposals" / f"crowd-training-{model_name}.json"


def collect(model_name):
    from rfdetr import RFDETRNano

    configure_cpu()
    scopes = records()
    file, key = state(model_name, scopes)
    cache = path(model_name)
    saved = read_json(cache) if cache.exists() else dict(key=key, frames={})
    if saved["key"] != key:
        raise ValueError(f"A previous {model_name} run used different inputs.")
    model = RFDETRNano.from_checkpoint(file, device="cpu")
    if model.class_names != ["syringe"] or model.model_config.resolution != EDGE:
        raise ValueError("The RF-DETR checkpoint is incompatible with the matched study.")
    for scope, rows in scopes.items():
        frames = saved["frames"].setdefault(scope, {})
        for row in rows:
            identifier = row["id"]
            if identifier in frames:
                continue
            picture = source(row)
            start = perf_counter()
            raw = list(ranked_region(model, picture, EDGE))
            suppressed = suppress(raw, .45)
            straight = suppressed
            amount = orientation(picture) if scope == "parallel" else None
            if amount is not None:
                background = picture.getpixel((0, 0))
                rotated = picture.rotate(amount, Image.Resampling.BICUBIC, expand=True, fillcolor=background)
                boxes = suppress(list(ranked_region(model, rotated, EDGE)), .45)
                straight = [{**p, "bbox": restored} for p in boxes
                            if (restored := restored_box(p["bbox"], picture.size, rotated.size, amount))]
            frames[identifier] = dict(ranked=raw, suppressed=suppressed, deskewed=straight,
                                      seconds=perf_counter() - start)
            if len(frames) % 4 == 0 or len(frames) == len(rows):
                write_json(cache, saved)
                print(model_name, scope, len(frames), "/", len(rows), flush=True)


def measure():
    if OUTPUT.exists():
        raise ValueError("The matched development evaluation already exists.")
    scopes = records()
    outcomes = {}
    for name in MODELS:
        saved = read_json(path(name))
        if saved["key"] != state(name, scopes)[1]:
            raise ValueError(f"The collected {name} outputs differ from the registered experiment.")
        metrics = {}
        for scope, rows in scopes.items():
            frames = saved["frames"][scope]
            if set(frames) != {r["id"] for r in rows}:
                raise ValueError(f"Incomplete {name} observations for {scope}.")
            views = VIEWS if scope == "parallel" else VIEWS[:2]
            subsets = {"all": rows}
            if scope == "ordinary":
                for domain in sorted({r["domain"] for r in rows}):
                    subsets[domain] = [r for r in rows if r["domain"] == domain]
            metrics[scope] = {
                subset: {view: short(evaluate(selected, {k: v[view] for k, v in frames.items()}, FLOOR))
                         for view in views}
                for subset, selected in subsets.items()
            }
            metrics[scope]["median_frame_seconds"] = median(v["seconds"] for v in frames.values())
        outcomes[name] = metrics
    write_json(OUTPUT, dict(outcomes=outcomes, score_floor=FLOOR, nms_iou=.45,
                            collection_sha256={name: digest(path(name)) for name in MODELS},
                            protocol="Compare three fixed training initializations at the same score floor. "
                                     "The deskewed view is tested on overlap development only. "
                                     "These scenes do not qualify a release."))
    print({name: values["parallel"]["all"] for name, values in outcomes.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.action == "measure":
        measure()
    elif args.model:
        collect(args.model)
    else:
        parser.error("Select a checkpoint for collection.")
