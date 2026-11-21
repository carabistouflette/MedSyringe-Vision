"""Compare RF-DETR transfer and additive fusion at a locked operating point."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter

from PIL import Image
import torch

from medsyringe.additive import additions, duplicates, overlapping_tiles, project
from medsyringe.data import digest, read_json, train_rows, valid_box, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.recall_guard import compare
from scripts.additive_study import FRESH as EXTERNAL
from scripts.optimize_detector import FINAL, counters
from scripts.precision_study import baseline, source
from scripts.refine_detector import ART, OLD
from scripts.rfdetr_transfer import AUDIT, MANIFEST, WORK
from scripts.supervised_study import sets

SELECTION = ART / "rfdetr_selection.json"
RUNS = ("public", "direct", "transfer")
VIEWS = ("full", "full_and_tiles")
FLOOR = .001


def ranked_region(model, image, edge):
    resized = image.resize((edge, edge), Image.Resampling.BILINEAR)
    detections = model.predict(resized, threshold=FLOOR, include_source_image=False)
    if not len(detections.xyxy):
        return
    for bbox, score, category in zip(detections.xyxy, detections.confidence,
                                     detections.data["class_name"], strict=True):
        if category == "__background__":
            continue
        if category != "syringe":
            raise ValueError(f"Unexpected RF-DETR label: {category}")
        box = [min(1., max(0., float(v) / edge)) for v in bbox]
        if valid_box(box):
            yield dict(bbox=box, score=float(score))


def checkpoint_record(run):
    path = ART / f"rfdetr_{run}_training.json"
    info = read_json(path)
    if info["status"] != "completed" or digest(info["checkpoint"]) != info["checkpoint_sha256"]:
        raise ValueError(f"Training/checkpoint verification failed for {run}")
    return info


def auxiliary(rows, name, run, views):
    from rfdetr import RFDETRSegNano

    record = checkpoint_record(run)
    edge = record["train"]["resolution"]
    cache = WORK / f"proposals_{name}_{run}.json"
    key = dict(checkpoint=record["checkpoint_sha256"], code=digest(__file__), views=list(views),
               rows={r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows})
    previous = read_json(cache) if cache.exists() else {}
    if previous.get("key") == key:
        return previous["payload"]
    model = RFDETRSegNano.from_checkpoint(record["checkpoint"], device="cpu")
    if model.class_names != ["syringe"] or model.model_config.resolution != edge:
        raise ValueError("Checkpoint metadata does not match the training manifest")
    predictions = {view: {} for view in views}
    durations = {view: [] for view in views}
    for i, row in enumerate(rows, 1):
        image = source(row)
        identifier = row["id"]
        started = perf_counter()
        full = suppress(list(ranked_region(model, image, edge)), .45)
        if "full" in views:
            predictions["full"][identifier] = full
            durations["full"].append(perf_counter()-started)
        if "full_and_tiles" in views:
            tiles = []
            for window in overlapping_tiles(image.size, .66):
                for proposal in ranked_region(model, image.crop(window), edge):
                    box = project(proposal["bbox"], window, image.size)
                    if box:
                        tiles.append({**proposal, "bbox": box})
            predictions["full_and_tiles"][identifier] = suppress([*full, *tiles], .45)
            durations["full_and_tiles"].append(perf_counter()-started)
        if i % 10 == 0 or i == len(rows):
            print(name, run, i, "/", len(rows), flush=True)
    payload = dict(predictions=predictions, latency_ms={
        view: dict(median=1000*median(times), p95=1000*quantiles(times, n=20, method="inclusive")[-1])
        for view, times in durations.items()})
    write_json(cache, dict(key=key, payload=payload))
    return payload


def baseline_predictions(rows, partition):
    if (ART / "precision_release.json").exists():
        raise ValueError("The study's baseline must match the active, unmodified v3 detector")
    return baseline(read_json(OLD), read_json(FINAL), rows, partition)


def combine(base, extra, floor, options):
    return {identifier: additions(predictions, extra[identifier], floor, **options)
            for identifier, predictions in base.items()}


def oracle(misses, base, extra, floor):
    results = []
    for miss in misses:
        identifier, box = miss["id"], miss["box"]
        ranked = extra[identifier]
        accepted = [p for p in base[identifier] if p["score"] >= floor]
        admissible = [p for p in ranked if not any(duplicates(p["bbox"], q["bbox"]) for q in accepted)]
        results.append(dict(id=identifier, box=box,
                            max_iou=max((box_iou(p["bbox"], box) for p in ranked), default=0.),
                            max_nonduplicate_iou=max((box_iou(p["bbox"], box) for p in admissible), default=0.),
                            max_matching_score=max((p["score"] for p in admissible
                                                    if box_iou(p["bbox"], box) >= .5), default=None)))
    return results


def public_calibration_rows():
    ids = set(read_json(ART / "rfdetr_public_data.json")["valid"]["ids"])
    rows = [r for r in read_json(MANIFEST)["rows"] if r["id"] in ids]
    if {r["id"] for r in rows} != ids:
        raise ValueError("The public calibration split is incomplete")
    return rows


def manifest_paths():
    return dict(calibration=ART / "semantic_holdout_manifest.json",
                public_calibration=ART / "rfdetr_public_data.json",
                public_manifest=MANIFEST, fresh=AUDIT,
                historical=ART / "test_manifest.json", previous_external=EXTERNAL,
                fil_rouge=ART / "source_inventory.json",
                fil_rouge_boxes=ART / "annotations.json")


def pipeline_hashes():
    paths = (Path(__file__), Path("scripts/rfdetr-requirements.txt"), Path("medsyringe/additive.py"),
             Path("medsyringe/evaluate.py"), Path("medsyringe/recall_guard.py"))
    return {path.name: digest(path) for path in paths}


def calibrate():
    paths = manifest_paths()
    project_rows = read_json(paths["calibration"])["rows"]
    public_rows = public_calibration_rows()
    subsets = dict(project=project_rows, zenodo_validation=public_rows)
    rows = [*project_rows, *public_rows]
    base = {**baseline_predictions(project_rows, "precision-fresh"),
            **baseline_predictions(public_rows, "rfdetr-public-validation")}
    floor = read_json(FINAL)["cutoff"]
    current = evaluate(rows, base, floor)
    reference = {k: evaluate(v, base, floor) for k, v in subsets.items()}
    standalones, trials, diagnosis, timings = [], [], {}, {}
    for run in RUNS:
        result = auxiliary(rows, "calibration", run, VIEWS)
        for view, proposals in result["predictions"].items():
            threshold = best_threshold(rows, proposals)
            standalones.append(dict(run=run, view=view, threshold=threshold,
                                    **counters(evaluate(rows, proposals, threshold)),
                                    per_source={k: counters(evaluate(v, proposals, threshold))
                                                for k, v in subsets.items()}))
            key = f"{run}/{view}"
            diagnosis[key] = oracle(current["false_negatives"], base, proposals, floor)
            timings[key] = result["latency_ms"][view]
            supports = [(None, .30), *product((.25, .40, .55), (.30, .50))]
            thresholds = sorted({.025, .05, .15, .25, .40, .55, .70, threshold})
            for aux_cutoff, (support, iou) in product(thresholds, supports):
                options = dict(aux_cutoff=aux_cutoff, support=support, iou=iou)
                hybrid = combine(base, proposals, floor, options)
                stats = evaluate(rows, hybrid, floor)
                subset_stats = {k: counters(evaluate(v, hybrid, floor)) for k, v in subsets.items()}
                trials.append(dict(run=run, view=view, options=options, **counters(stats),
                                   per_source=subset_stats, **compare(current, stats)))
    eligible = [t for t in trials if t["passed"] and t["tp"] > current["tp"]
                and t["precision"] >= current["precision"]
                and all(t["per_source"][k]["precision"] >= s["precision"]
                        for k, s in reference.items())]
    def rank(item):
        return (item["f1"], item["precision"], item["recall"], item["view"] == "full",
                item["options"]["aux_cutoff"])

    selected = max(eligible, key=rank, default=None)
    # Lock one illustrative configuration for the reserve even if no setting
    # qualifies. In that case it remains ineligible regardless of audit scores.
    preserved = [t for t in trials if t["passed"]]
    recoveries = [t for t in preserved if t["tp"] > current["tp"]]
    diagnostic = max(recoveries or preserved, key=rank)
    locked = selected or diagnostic
    record = dict(
        baseline=counters(current), baseline_per_source={k: counters(v) for k, v in reference.items()},
        selected=selected, locked=locked, calibration_trials=trials,
        standalone=standalones, missed_instance_analysis=diagnosis, added_latency_ms=timings,
        manifest_sha256={name: digest(path) for name, path in paths.items()},
        pipeline_sha256=pipeline_hashes(),
        checkpoint_sha256={run: checkpoint_record(run)["checkpoint_sha256"] for run in RUNS},
        release_sha256=digest(FINAL))
    write_json(SELECTION, record)
    print("Baseline", counters(current), "eligible", len(eligible), "locked", locked, flush=True)


def audit(cohort):
    selection = read_json(SELECTION)
    paths = manifest_paths()
    if (selection["release_sha256"] != digest(FINAL) or
            selection["pipeline_sha256"] != pipeline_hashes() or
            selection["manifest_sha256"] != {name: digest(path) for name, path in paths.items()} or
            selection["checkpoint_sha256"] != {run: checkpoint_record(run)["checkpoint_sha256"] for run in RUNS}):
        raise ValueError("Frozen inputs changed after calibration")
    split = dict(fresh="rfdetr-reserve", fil_rouge="train", historical="test",
                 previous_external="additive-audit")[cohort]
    rows = train_rows(Path(".")) if cohort == "fil_rouge" else read_json(paths[cohort])["rows"]
    base = baseline_predictions(rows, split)
    floor = read_json(FINAL)["cutoff"]
    current = evaluate(rows, base, floor)
    setting = selection["locked"]
    run, view = setting["run"], setting["view"]
    # All three transfer controls are scored on the reserve with separately
    # frozen validation thresholds. Their ranks never change the final choice.
    variants = RUNS if cohort == "fresh" else (run,)
    extra, standalone, timings = {}, {}, {}
    for variant in variants:
        result = auxiliary(rows, cohort, variant, (view,))
        extra[variant] = result["predictions"][view]
        cutoff = next(s["threshold"] for s in selection["standalone"]
                      if (s["run"], s["view"]) == (variant, view))
        standalone[variant] = evaluate(rows, extra[variant], cutoff)
        timings[variant] = result["latency_ms"][view]
    candidate = evaluate(rows, combine(base, extra[run], floor, setting["options"]), floor)
    write_json(ART / f"rfdetr_evaluation_{cohort}.json", dict(
        baseline=current, candidate=candidate, guard=compare(current, candidate),
        standalone=standalone, added_latency_ms=timings,
        missed_instance_analysis=oracle(current["false_negatives"], base, extra[run], floor),
        protocol_sha256=digest(SELECTION)))
    print(cohort, "v3", counters(current), "hybrid", counters(candidate), flush=True)


def summarize():
    selection = read_json(SELECTION)
    evidence = {}
    for cohort in ("fresh", "fil_rouge", "historical", "previous_external"):
        item = read_json(ART / f"rfdetr_evaluation_{cohort}.json")
        if item["protocol_sha256"] != digest(SELECTION):
            raise ValueError("Evaluation was run against a different protocol")
        evidence[cohort] = dict(baseline=item["baseline"], candidate=item["candidate"],
                                guard=item["guard"])
    reserve = evidence["fresh"]
    eligible = (selection["selected"] is not None
                and all(item["guard"]["passed"] for item in evidence.values())
                and all(item["candidate"]["precision"] >= item["baseline"]["precision"]
                        for item in evidence.values())
                and reserve["candidate"]["tp"] > reserve["baseline"]["tp"])
    write_json(ART / "rfdetr_release_decision.json", dict(
        selected_on_calibration=selection["selected"] is not None,
        eligible_for_release=eligible,
        scope="The reserve is independent by Zenodo clip only. Fil rouge is development data.",
        protocol_sha256=digest(SELECTION), cohorts={
            cohort: dict(baseline=counters(item["baseline"]), candidate=counters(item["candidate"]),
                         guard=item["guard"]) for cohort, item in evidence.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "audit", "summarize", "baseline"])
    parser.add_argument("--cohort", choices=["fresh", "fil_rouge", "historical", "previous_external"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "calibrate":
        calibrate()
    elif args.stage == "audit":
        if not args.cohort:
            parser.error("An audit cohort is required.")
        audit(args.cohort)
    elif args.stage == "baseline":
        baseline_predictions(read_json(AUDIT)["rows"], "rfdetr-reserve")
    else:
        summarize()


if __name__ == "__main__":
    main()
