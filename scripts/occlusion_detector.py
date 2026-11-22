"""Fit a small-syringe RF-DETR challenger and measure full-frame and tiled inference."""
import argparse
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

from medsyringe.additive import duplicates, overlapping_tiles, project
from medsyringe.data import digest, leakage, read_image, read_json, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import best_threshold, evaluate, match
from scripts.detection_box_continuation import RECORD as INITIAL
from scripts.detection_real_box_transfer import ANNOTATIONS, EDGE, benchmark_sources, coco
from scripts.low_count_prospective import WORK as TRAY
from scripts.occlusion_proposals import oracle
from scripts.part_verifier_new_acquisition import row as oriez
from scripts.part_verifier_study import REFERENCES, WORK as PART, crowd
from scripts.rfdetr_compare import ranked_region

DATA = Path("data/occlusion-study")
ART = crowd.ART
PLAN = ART / "occlusion_detector_plan.json"
TRAIN = ART / "occlusion_detector_training.json"
FREEZE = ART / "occlusion_detector_selection.json"
RESULTS = ART / "occlusion_detector_results.json"
FRESH = ART / "occlusion_detector_reserve.json"
SEED = 631
ARMS = ("initial-full", "initial-tiled", "specialist-full", "specialist-tiled")


def path_for(row):
    path = Path(row["image"])
    return path if path.parts[0] == "data" or path.is_absolute() else Path("data") / path


def raster(row):
    path = path_for(row)
    if digest(path) != row.get("derived_sha256", row["sha256"]):
        raise ValueError(f"Raster changed for {row['id']}.")
    return read_image(path)


def crop_labels(objects, window):
    """Return transformed labels only for windows with no cut-through objects."""
    left, top, right, bottom = window
    labels = []
    for obj in objects:
        x, y, X, Y = obj["bbox"]
        if x >= right or X <= left or y >= bottom or Y <= top:
            continue
        if x < left or y < top or X > right or Y > bottom:
            return None
        box = [(x-left)/(right-left), (y-top)/(bottom-top),
               (X-left)/(right-left), (Y-top)/(bottom-top)]
        labels.append(dict(bbox=box))
    return labels


def focused_window(row, image):
    if not row["objects"]:
        return None
    smallest = min(row["objects"], key=lambda obj: max(obj["bbox"][2]-obj["bbox"][0],
                                                       obj["bbox"][3]-obj["bbox"][1]))
    x, y, X, Y = smallest["bbox"]
    if max(X-x, Y-y) > .38 and len(row["objects"]) == 1:
        return None
    for fraction in (.44, .60, .74):
        left = min(1-fraction, max(0., (x+X-fraction) / 2))
        top = min(1-fraction, max(0., (y+Y-fraction) / 2))
        w, h = image.size
        window = (round(left*w), round(top*h), round((left+fraction)*w), round((top+fraction)*h))
        normalized = [v / unit for v, unit in zip(window, (w, h, w, h), strict=True)]
        if labels := crop_labels(row["objects"], normalized):
            return window, labels
    return None


def partitions():
    roles = read_json(REFERENCES)["roles"]
    real, _, dev = benchmark_sources()
    seen = set()
    fit = []
    for row in [*roles["fit"], *real]:
        if row["sha256"] not in seen:
            seen.add(row["sha256"])
            fit.append(row)
    protected = [*roles["regression"], *roles["transfer"], *roles["fresh"], oriez()]
    collisions = leakage([*
        ({**r, "split": "fit"} for r in fit), *
        ({**r, "split": "development"} for r in dev), *
        ({**r, "split": "challenge"} for r in protected)])
    fit_ids = {r["id"] for r in fit}
    dropped = {i for pair in collisions for i in pair["images"] if i in fit_ids}
    fit = [r for r in fit if r["id"] not in dropped]
    remaining = leakage([*({**r, "split": "fit"} for r in fit),
                         *({**r, "split": "development"} for r in dev),
                         *({**r, "split": "challenge"} for r in protected)])
    if remaining:
        raise ValueError(f"Outstanding cross-role overlap: {remaining[:3]}")
    return fit, dev, sorted(dropped)


def prepare():
    initial = read_json(INITIAL)
    if initial["status"] != "completed" or digest(initial["checkpoint"]) != initial["checkpoint_sha256"]:
        raise ValueError("Initialization checkpoint changed.")
    prepared = DATA / "coco"
    if PLAN.exists() or prepared.exists():
        raise FileExistsError(prepared)
    (prepared / "crops").mkdir(parents=True)
    fit, dev, dropped = partitions()
    ranked = list(fit)
    parents = []
    assets = {}
    for row in fit:
        image = raster(row)
        if found := focused_window(row, image):
            window, labels = found
            key = f"focus-{row['id']}"
            patch = image.crop(window)
            destination = prepared / "crops" / f"{key}.png"
            patch.save(destination)
            ranked.append(dict(id=key, sha256=digest(destination), objects=labels))
            assets[key] = (patch.size, destination)
            parents.append(dict(id=key, parent=row["id"], sha256=digest(destination),
                                parent_sha256=row["sha256"], group=row["group"], window=window,
                                objects=len(labels)))
    for row in [*fit, *dev]:
        img = raster(row)
        filename = prepared / "crops" / f"source-{row['id']}.png"
        img.save(filename)
        assets[row["id"]] = (img.size, filename)
    Random(SEED).shuffle(ranked)
    summaries = {key: coco(entries, prepared / key, assets)
                 for key, entries in (("train", ranked), ("valid", dev))}
    labels = {key: [dict(id=r["id"], sha256=r["sha256"], group=r["group"]) for r in rows]
              for key, rows in (("fit", fit), ("development", dev))}
    rasters = {str(path): digest(path) for key in ("train", "valid")
               for path in sorted((prepared / key).glob("*.png"))}
    write_json(PLAN, dict(seed=SEED, code_sha256=digest(__file__), initial_sha256=initial["checkpoint_sha256"],
                          counts=summaries, roles=labels, crops=parents, excluded=dropped, rasters=rasters))
    print(summaries, "crop views", len(parents), "exclusions", dropped, flush=True)


def train():
    import torch
    from rfdetr import RFDETRNano

    if TRAIN.exists():
        raise FileExistsError(TRAIN)
    plan = read_json(PLAN)
    initial = read_json(INITIAL)
    prepared = DATA / "coco"
    if any(digest(prepared / key / ANNOTATIONS) != v["sha256"] for key, v in plan["counts"].items()):
        raise ValueError("COCO exports changed.")
    if any(digest(p) != expected for p, expected in plan["rasters"].items()):
        raise ValueError("Prepared pixels changed.")
    if digest(initial["checkpoint"]) != plan["initial_sha256"]:
        raise ValueError("Initialization file changed.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    output = DATA / "rfdetr"
    config = dict(dataset_dir=str(prepared), output_dir=str(output), device="cpu", epochs=1,
                  batch_size=4, resolution=EDGE, seed=SEED, lr=.000075, lr_encoder=0,
                  num_workers=0, num_sanity_val_steps=0, compute_val_loss=True, use_ema=False,
                  multi_scale=False, save_dataset_grids=False, run_test=False, train_log_on_step=False)
    started = perf_counter()
    model = RFDETRNano(device="cpu", pretrain_weights=initial["checkpoint"], resolution=EDGE,
                       num_classes=1, freeze_encoder=True)
    model.train(**config)
    checkpoint = output / "checkpoint_best_total.pth"
    write_json(TRAIN, dict(status="completed", config=config, plan_sha256=digest(PLAN),
                          checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                          curve_sha256=digest(output / "metrics.csv"), seconds=perf_counter()-started))


def inference(rows):
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    predictions = {arm: {} for arm in ARMS}
    time_ms = {arm: [] for arm in ARMS}
    for name, record in (("initial", INITIAL), ("specialist", TRAIN)):
        info = read_json(record)
        if info["status"] != "completed" or digest(info["checkpoint"]) != info["checkpoint_sha256"]:
            raise ValueError(f"Invalid checkpoint for {name}")
        model = RFDETRNano.from_checkpoint(info["checkpoint"], device="cpu")
        for row in rows:
            image = raster(row)
            started = perf_counter()
            raw = list(ranked_region(model, image, EDGE))
            choices = {f"{name}-full": suppress(raw, .45)}
            time_ms[f"{name}-full"].append(1000*(perf_counter()-started))
            tiles = []
            for window in overlapping_tiles(image.size, .66):
                for candidate in ranked_region(model, image.crop(window), EDGE):
                    if bbox := project(candidate["bbox"], window, image.size):
                        tiles.append({**candidate, "bbox": bbox})
            choices[f"{name}-tiled"] = suppress([*raw, *tiles], .45)
            time_ms[f"{name}-tiled"].append(1000*(perf_counter()-started))
            for arm, candidates in choices.items():
                predictions[arm][row["id"]] = candidates
        print(name, len(rows), "images evaluated", flush=True)
        del model
    return predictions, {arm: median(values) for arm, values in time_ms.items()}


def select():
    if FREEZE.exists():
        raise FileExistsError(FREEZE)
    _, dev, _ = partitions()
    planned = read_json(PLAN)["roles"]["development"]
    observed = [dict(id=r["id"], sha256=r["sha256"], group=r["group"]) for r in dev]
    if observed != planned:
        raise ValueError("The predeclared development rows changed.")
    predictions, times = inference(dev)
    thresholds = {arm: best_threshold(dev, ranked) for arm, ranked in predictions.items()}
    values = {arm: evaluate(dev, predictions[arm], threshold) for arm, threshold in thresholds.items()}
    write_json(DATA / "development-predictions.json", predictions)
    write_json(FREEZE, dict(floors=thresholds, development=values, median_latency_ms=times,
                            train_sha256=digest(TRAIN), code_sha256=digest(__file__),
                            fusion="Append confidence-qualified candidates unless they overlap a released or appended box.",
                            release="Gain on prospective evidence, zero added false positives and preserve every matched reference."))


def fuse(released, ranked, floor):
    combined = list(released)
    for candidate in ranked:
        if candidate["score"] >= floor and not any(duplicates(candidate["bbox"], p["bbox"]) for p in combined):
            combined.append({**candidate, "score": min(candidate["score"], .01)})
    return combined


def frozen_frames():
    roles = read_json(REFERENCES)["roles"]
    baseline = {key: frame["objects"] for key, frame in
                read_json(crowd.CACHE / "released.json")["frames"].items()}
    baseline.update(read_json(TRAY / "released.json")["outputs"])
    baseline.update(read_json(PART / "released.json")["frames"])
    known = oriez()
    baseline[known["id"]] = read_json(PART / "oriez-baseline.json")["objects"]
    fresh = read_json(FRESH)["rows"]
    cache = read_json(DATA / "reserve-released.json")
    if cache["reference_sha256"] != digest(FRESH):
        raise ValueError("Prospective comparison does not match the annotated reserve.")
    baseline.update(cache["frames"])
    return {"regression": roles["regression"], "transfer": roles["transfer"],
            "previous-lookalikes": roles["fresh"],
            "prior-small-occlusion": [known], "new-public-reserve": fresh}, baseline


def target_ids(rows, predictions):
    _, used = match(rows, predictions, .50)
    return {(key, i) for key, indices in used.items() for i in indices}


def score():
    sealed = read_json(FREEZE)
    if sealed["train_sha256"] != digest(TRAIN) or sealed["code_sha256"] != digest(__file__):
        raise ValueError("Frozen operating points changed.")
    cohorts, baseline = frozen_frames()
    all_rows = [r for block in cohorts.values() for r in block]
    predictions, elapsed = inference(all_rows)
    report = {}
    for role, rows in cohorts.items():
        released = {r["id"]: baseline[r["id"]] for r in rows}
        initial = target_ids(rows, released)
        observed = evaluate(rows, released, 0.)
        variants = {}
        for arm, floor in sealed["floors"].items():
            ranked = predictions[arm]
            combined = {r["id"]: fuse(released[r["id"]], ranked[r["id"]], floor) for r in rows}
            lost = sorted(initial - target_ids(rows, combined))
            additive = evaluate(rows, combined, 0.)
            variants[arm] = dict(standalone=evaluate(rows, ranked, floor), additive=additive, lost_targets=lost,
                                 added_false_positives=additive["fp"]-observed["fp"],
                                 oracle={r["id"]: oracle(r["objects"], ranked[r["id"]]) for r in rows})
        report[role] = dict(released=observed, arms=variants)
    gate = {arm: all(not group["arms"][arm]["lost_targets"] and
                     group["arms"][arm]["added_false_positives"] <= 0 for group in report.values())
            and report["new-public-reserve"]["arms"][arm]["additive"]["tp"] >
                report["new-public-reserve"]["released"]["tp"] for arm in ARMS}
    write_json(DATA / "scored-predictions.json", predictions)
    write_json(RESULTS, dict(selection_sha256=digest(FREEZE), reserve_sha256=digest(FRESH),
                            median_latency_ms=elapsed, cohorts=report, meets_release_gate=gate))
    print(gate, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "select", "score"))
    {"prepare": prepare, "train": train, "select": select, "score": score}[parser.parse_args().stage]()
