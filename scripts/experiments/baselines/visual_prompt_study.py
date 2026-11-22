"""Isolated local YOLOE visual prompting benchmark; not a product dependency."""
import argparse
from itertools import combinations, product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from medsyringe.detection.additive import additions, overlapping_tiles, project
from medsyringe.data import digest, leakage, read_json, train_rows, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.baselines.additive_study import FRESH
from scripts.experiments.detector_training.optimize_detector import FINAL, counters
from scripts.experiments.detector_audits.precision_study import DATA, baseline, source
from scripts.experiments.detector_training.refine_detector import ART, OLD
from scripts.experiments.transfer.supervised_study import ROOT, sets
from scripts.experiments.public_data.visual_prompt_data import references

WEIGHTS = Path("data/models/yoloe/yoloe-26n-seg.pt")
PILOT = ART / "visual_prompt_selection.json"


def visual_proposals(rows, partition):
    # The optional package lives in a separate experimental environment.
    import ultralytics
    from ultralytics import YOLOE, settings
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    settings.update({"sync": False})
    refs = references()
    fingerprint = dict(weights=digest(WEIGHTS), version=ultralytics.__version__,
                       references={k: [r["sha256"], r["objects"]] for k, r in refs.items()},
                       code=digest(__file__), rows={r["id"]: r["sha256"] for r in rows})
    file = Path("data/cache") / f"yoloe-visual-{partition}.json"
    saved = read_json(file) if file.exists() else {}
    if saved.get("fingerprint") == fingerprint:
        return saved["predictions"]
    results = {r["id"]: {} for r in rows}
    elapsed = 0.0
    for key, row in refs.items():
        ref = source(row)
        ref.thumbnail((640, 640))
        w, h = ref.size
        box = np.asarray(row["objects"][0]["bbox"]) * [w, h, w, h]
        model = YOLOE(str(WEIGHTS))
        model.predict(ref, refer_image=ref, predictor=YOLOEVPSegPredictor, device="cpu", imgsz=640, verbose=False,
                      conf=.015, visual_prompts={"bboxes": np.array([box]),
                                                 "cls": np.array([0])})
        for obj in rows:
            im = source(obj)
            w, h = im.size
            windows = [(0, 0, w, h), *overlapping_tiles(im.size)]
            started = perf_counter()
            outputs = model.predict([im.crop(wnd) for wnd in windows], device="cpu", imgsz=640,
                                    verbose=False, conf=.015, max_det=80)
            elapsed += perf_counter()-started
            variants = {"full": [], "tiles": []}
            for i, (window, out) in enumerate(zip(windows, outputs)):
                for xyxy, confidence in zip(out.boxes.xyxyn.tolist(), out.boxes.conf.tolist()):
                    box = project(xyxy, window, im.size)
                    if box:
                        variants["tiles" if i else "full"].append(dict(bbox=box, score=confidence))
            results[obj["id"]][key] = variants
        print(partition, key, len(rows), "images", flush=True)
    write_json(file, dict(fingerprint=fingerprint, predictions=results,
                          seconds_for_all_prompt_and_view_variants=elapsed))
    return results


def combine(base, visual, cutoff, chosen):
    result = {}
    for identifier, ranked in base.items():
        alternatives = [p for ref in chosen["references"] for view in chosen["views"]
                        for p in visual[identifier][ref][view]]
        result[identifier] = additions(ranked, alternatives, cutoff, support=chosen["support"],
                                      aux_cutoff=chosen["aux_cutoff"], iou=chosen["iou"])
    return result


def select(rows, base, visual, cutoff):
    old = evaluate(rows, base, cutoff)
    trials, single = [], []
    keys = list(references())
    refsets = [(k,) for k in keys] + list(combinations(keys, 2))
    for ref, views in product(refsets, [("full",), ("tiles",), ("full", "tiles")]):
        standalone = {i: suppress([p for r in ref for v in views for p in preds[r][v]])
                      for i, preds in visual.items()}
        point = best_threshold(rows, standalone)
        single.append(dict(references=ref, views=views, cutoff=point,
                           **counters(evaluate(rows, standalone, point))))
        for aux, support, iou in product((.035, .08, .15, .30, .50), (.25, .45, .60), (.35, .50)):
            config = dict(references=ref, views=views, aux_cutoff=aux, support=support, iou=iou)
            score = evaluate(rows, combine(base, visual, cutoff, config), cutoff)
            trials.append(dict(**config, **counters(score), kept_every_match=compare(old, score)["passed"]))
    eligible = [t for t in trials if t["kept_every_match"] and t["tp"] > old["tp"]
                and t["precision"] >= old["precision"]]
    selected = max(eligible, key=lambda t: (t["f1"], t["precision"], -len(t["references"]))) if eligible else None
    single.sort(key=lambda t: (t["f1"], t["ap50_95"]), reverse=True)
    write_json(PILOT, dict(baseline=counters(old), standalone=single, trials=trials, selected=selected,
                           holdout_sha256=digest(FRESH) if FRESH.exists() else None,
                           weights_sha256=digest(WEIGHTS), detector_sha256=digest(FINAL)))
    print("Selected", selected, "best standalone", single[0], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "audit", "historical", "fil-rouge"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    train, val, historical = sets()
    frozen, detector = read_json(OLD), read_json(FINAL)
    if args.stage == "select":
        visual = visual_proposals(val, "precision-fresh")
        select(val, baseline(frozen, detector, val, "precision-fresh"), visual, detector["cutoff"])
        return
    protocol, fresh = read_json(PILOT), read_json(FRESH)["rows"]
    if (protocol["holdout_sha256"] != digest(FRESH) or protocol["weights_sha256"] != digest(WEIGHTS)
        or protocol["detector_sha256"] != digest(FINAL)):
        raise ValueError("Test input or model changed since selection.")
    seen = read_json(ART / "supervised_external_manifest.json")["rows"]
    if leakage([*[{**r, "split": "training"} for r in train], *val, *historical, *seen, *fresh]):
        raise ValueError("Grouped experiment partitions overlap.")
    rows, split = {"audit": (fresh, "additive-audit"), "historical": (historical, "test"),
                   "fil-rouge": (train_rows(ROOT), "train")}[args.stage]
    visual = visual_proposals(rows, split)
    ranked = baseline(frozen, detector, rows, split)
    old = evaluate(rows, ranked, detector["cutoff"])
    choice = protocol["selected"]
    standalone = protocol["standalone"][0]
    vp = {i: suppress([p for ref in standalone["references"] for view in standalone["views"]
                       for p in values[ref][view]]) for i, values in visual.items()}
    vp_metrics = evaluate(rows, vp, standalone["cutoff"])
    result = {"baseline": old, "standalone": vp_metrics, "selection_sha256": digest(PILOT)}
    if choice:
        candidate = evaluate(rows, combine(ranked, visual, detector["cutoff"], choice), detector["cutoff"])
        result.update(candidate=candidate, recall_guard=compare(old, candidate))
    write_json(ART / f"visual_prompt_{args.stage}.json", result)
    print(args.stage, "baseline", counters(old), "standalone", counters(vp_metrics),
          "candidate", counters(result["candidate"]) if choice else None, flush=True)


if __name__ == "__main__":
    main()
