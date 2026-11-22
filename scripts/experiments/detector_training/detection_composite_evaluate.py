"""Compare both training arms on the unchanged public development photographs."""
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from rfdetr import RFDETRSegNano

from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import evaluate
from scripts.experiments.detector_training.detection_composite_ablation import ARMS, ART, SETUP
from scripts.experiments.detector_training.detection_hillclimb_experiment import inputs
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_compare import ranked_region

OUTPUT = ART/"composite_ablation_results.json"


def run():
    if OUTPUT.exists():
        raise ValueError("The matched-budget comparison already exists.")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data = inputs("public-development")
    cases = [(r,source(r)) for r in data]
    outcomes = {}
    for arm in ARMS:
        log = read_json(ART/f"rfdetr_composite_{arm}_run.json")
        path = Path(log["checkpoint"])
        if (log["status"]!="completed" or log["dataset_sha256"]!=digest(SETUP) or
            log["checkpoint_sha256"]!=digest(path)):
            raise ValueError("The warm-started model and dataset are not sealed.")
        with (ART/f"rfdetr_composite_{arm}_metrics.csv").open(newline="") as file:
            row=next(r for r in csv.DictReader(file) if r["val/mAP_50"])
        stats={k.removeprefix("val/"):float(v) for k,v in row.items() if k.startswith("val/") and v}
        model=RFDETRSegNano.from_checkpoint(str(path),device="cpu")
        ranked={}
        times=[]
        for r,img in cases:
            start=perf_counter()
            ranked[r["id"]]=suppress(list(ranked_region(model,img,240)),.45)
            times.append(perf_counter()-start)
        outcomes[arm]=dict(simulator_validation=stats,inference_seconds_median=median(times[1:]),
                           public_development=short(evaluate(data,ranked,.20)), checkpoint_sha256=digest(path),
                           training_seconds=log["seconds"])
        print(arm,outcomes[arm]["public_development"],flush=True)
    write_json(OUTPUT,dict(
        dataset_sha256=digest(SETUP),outcomes=outcomes,
        scope="One seed, one matched CPU epoch. Public development is a distinct photo group. The 0.20 score is a "
              "diagnostic operating point. Neither model was selected using reserve data."))


if __name__=="__main__":
    run()
