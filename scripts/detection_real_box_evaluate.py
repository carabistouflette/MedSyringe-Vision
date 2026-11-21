"""Compare matched RF-DETR training on the protected development domains."""
import argparse
from csv import DictReader
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.additive import overlapping_tiles, project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import best_threshold, evaluate
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, loaded
from scripts.detection_real_box_transfer import ARMS, DESIGN, EDGE, benchmark_sources
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.rfdetr_compare import ranked_region
from scripts.sam3_study import configure_cpu

OUTPUT = ART / "reviewed_box_transfer_comparison.json"
VIEWS = ("whole", "wide")


def path(arm):
    return DATA / "proposals" / f"reviewed-box-transfer-{arm}.json"


def training(arm):
    run = ART / f"reviewed_box_transfer_{arm}.json"
    record = read_json(run)
    if (record["status"] != "completed" or record["dataset_sha256"] != digest(DESIGN)
            or record["checkpoint_sha256"] != digest(record["checkpoint"])):
        raise ValueError("Transfer run is incomplete or has invalid provenance.")
    return record


def key(arm, rows):
    return dict(dataset=digest(DESIGN), checkpoint=training(arm)["checkpoint_sha256"],
                inputs={r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows},
                implementation=digest(__file__))


def collect(arm):
    from rfdetr import RFDETRNano

    configure_cpu()
    _, _, rows = benchmark_sources()
    binding = key(arm, rows)
    destination = path(arm)
    saved = read_json(destination) if destination.exists() else dict(key=binding, frames={})
    if saved["key"] != binding:
        raise ValueError("Existing evaluation cache has different inputs.")
    todo = [r for r in rows if r["id"] not in saved["frames"]]
    if not todo:
        return
    model = RFDETRNano.from_checkpoint(training(arm)["checkpoint"], device="cpu")
    if model.class_names != ["syringe"] or model.model_config.resolution != EDGE:
        raise ValueError("The class names and inference resolution differ from training.")
    for row in todo:
        picture = source(row)
        start = perf_counter()
        whole = suppress(list(ranked_region(model, picture, EDGE)), .45)
        wide = list(whole)
        for view in overlapping_tiles(picture.size, .66):
            for proposal in ranked_region(model, picture.crop(view), EDGE):
                if proposal["score"] >= .008 and (box := project(proposal["bbox"], view, picture.size)):
                    wide.append({**proposal, "bbox": box})
        saved["frames"][row["id"]] = dict(whole=whole, wide=suppress(wide, .45),
                                          seconds=perf_counter() - start)
        count = len(saved["frames"])
        if count % 4 == 0 or count == len(rows):
            write_json(destination, saved)
            print(arm, "box transfer development", count, "/", len(rows), flush=True)


def curve(arm):
    with (ART / f"reviewed_box_transfer_{arm}_metrics.csv").open(newline="") as file:
        columns = ("F1", "mAP_50", "mAP_50_95", "precision", "recall")
        return [dict(epoch=int(r["epoch"])+1, **{k: float(r[f"val/{k}"]) for k in columns})
                for r in DictReader(file) if r["val/mAP_50"]]


def measure():
    if OUTPUT.exists():
        raise ValueError("The controlled development study has already been measured.")
    _, _, rows = benchmark_sources()
    domains = {kind: [r for r in rows if r["domain"] == kind] for kind in sorted({r["domain"] for r in rows})}
    outcomes = {}
    for arm in ARMS:
        stored = read_json(path(arm))
        if stored["key"] != key(arm, rows) or set(stored["frames"]) != {r["id"] for r in rows}:
            raise ValueError("The checkpoint must be evaluated on every fixed development input.")
        frames = stored["frames"]
        scored = {}
        for view in VIEWS:
            ranked = {name: f[view] for name, f in frames.items()}
            selected = best_threshold(rows, ranked)
            scored[view] = dict(development_upper_bound_threshold=selected, domains={
                kind: dict(common_floor=short(evaluate(refs, ranked, .20)),
                           development_upper_bound=short(evaluate(refs, ranked, selected)))
                for kind, refs in domains.items()})
        outcomes[arm] = dict(scored=scored, curves=curve(arm), checkpoint=training(arm)["checkpoint_sha256"],
                             collection_sha256=digest(path(arm)),
                             median_cpu_whole_plus_tiles_seconds=median(f["seconds"] for f in frames.values()))
    baseline = {}
    for scope in DEVELOPMENT:
        stored = loaded(scope, "baseline")
        cutoff, = {f["floor"] for f in stored.values()}
        baseline[scope] = short(evaluate(domains[scope], {k: v["ranked"] for k, v in stored.items()}, cutoff))
    write_json(OUTPUT, dict(dataset_sha256=digest(DESIGN), matched_arms=outcomes, earlier_hybrid=baseline,
                            scope="Development only and two equal-budget epochs. The reported upper bound fixes "
                                  "the best pooled development score for each branch, with no claim about reserves."))
    for arm in ARMS:
        print(arm, outcomes[arm]["curves"][-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "measure":
        measure()
    elif args.arm:
        collect(args.arm)
    else:
        parser.error("Select a completed training arm.")
