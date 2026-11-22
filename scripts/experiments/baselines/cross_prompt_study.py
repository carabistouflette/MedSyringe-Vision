"""Probe cross-model agreement on the development set only."""
from itertools import product

import torch

from medsyringe.detection.additive import additions
from medsyringe.data import write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.baselines.additive_study import proposals
from scripts.experiments.detector_training.optimize_detector import counters
from scripts.experiments.detector_training.refine_detector import ART
from scripts.experiments.transfer.supervised_study import sets
from scripts.experiments.baselines.visual_prompt_study import visual_proposals


def main():
    torch.set_num_threads(4)
    _, rows, _ = sets()
    ranked, ssd, cutoff = proposals(rows, "precision-fresh")
    visual = visual_proposals(rows, "precision-fresh")
    old = evaluate(rows, ranked, cutoff)
    trials = []
    for ssd_min, visual_min, overlap in product((.05,.15,.35,.60), (.015,.03,.06,.12), (.30,.45)):
        predictions = {}
        for row in rows:
            i = row["id"]
            vp = [p for views in visual[i].values() for subset in views.values() for p in subset
                  if p["score"] >= visual_min]
            accepted = [p for p in ssd[i]["both"] if p["score"] >= ssd_min and
                        any(box_iou(p["bbox"], q["bbox"]) >= overlap for q in vp)]
            predictions[i] = additions(ranked[i], accepted, cutoff, support=None,
                                        aux_cutoff=ssd_min, iou=overlap)
        metrics = evaluate(rows, predictions, cutoff)
        trials.append(dict(ssd_min=ssd_min, visual_min=visual_min, overlap=overlap,
                           **counters(metrics), retained=compare(old, metrics)["passed"]))
    write_json(ART / "cross_prompt_study.json", dict(baseline=counters(old), trials=trials,
               scope="Exploratory with previously inspected development data"))
    print("Strongest cross-support setting", max(trials, key=lambda t:(t["f1"],t["recall"])), flush=True)


if __name__ == "__main__":
    main()
