"""Evaluate actual augmentation checkpoints on shared real development photographs."""
import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import evaluate
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.public_augmentation_ablation import ARMS, EDGE, PLAN, inputs
from scripts.rfdetr_compare import ranked_region
from scripts.sam3_study import configure_cpu

CACHE = DATA / "proposals/public-augmentation"
RESULT = ART / "public_augmentation_development.json"
FLOOR = .20


def validation():
    _, rows = inputs()
    return {r["id"]: r for r in rows}


def bound(arm, rows):
    file = ART / f"public_augmentation_{arm}.json"
    run = read_json(file)
    if run["status"] != "completed" or digest(run["checkpoint"]) != run["checkpoint_sha256"]:
        raise ValueError(f"Incomplete or changed augmentation run {arm}.")
    if digest(PLAN) != run["design_sha256"]:
        raise ValueError("The matched design has changed.")
    return run["checkpoint"], dict(code=digest(__file__), checkpoint=run["checkpoint_sha256"],
                                   data=run["design_sha256"], input={
                                       k: [r["sha256"], [p["bbox"] for p in r["objects"]]] for k, r in rows.items()})


def collect(arm):
    from rfdetr import RFDETRNano

    configure_cpu()
    rows = validation()
    weights, key = bound(arm, rows)
    path = CACHE / f"{arm}.json"
    state = read_json(path) if path.exists() else dict(binding=key, frames={})
    if state["binding"] != key:
        raise ValueError("The checkpoint, validation images or analysis changed.")
    estimator = RFDETRNano.from_checkpoint(weights, device="cpu")
    if estimator.class_names != ["syringe"] or estimator.model_config.resolution != EDGE:
        raise ValueError("Unexpected class or resolution in the saved detector.")
    for name, row in rows.items():
        if name in state["frames"]:
            continue
        image = source(row)
        start = perf_counter()
        detections = suppress(list(ranked_region(estimator, image, EDGE)), .45)
        state["frames"][name] = dict(ranked=detections, seconds=perf_counter() - start)
        write_json(path, state)
    print("Public augmentation", arm, len(state["frames"]), "fresh inferences", flush=True)


def measure():
    if RESULT.exists():
        raise FileExistsError("The matched augmentation comparison is already measured.")
    all_rows = validation()
    domains = defaultdict(list)
    for row in all_rows.values():
        domains[row.get("domain", "transparent-development")].append(row)
    ledger = {}
    for arm in ARMS:
        results = read_json(CACHE / f"{arm}.json")
        frames = results["frames"]
        if results["binding"] != bound(arm, all_rows)[1] or set(frames) != set(all_rows):
            raise ValueError("An augmentation arm has incomplete real development predictions.")
        forecasts = {k: frame["ranked"] for k, frame in frames.items()}
        ledger[arm] = {
            role: short(evaluate(local, forecasts, FLOOR))
            for role, local in {"all": list(all_rows.values()), **domains}.items()
        }
    write_json(RESULT, dict(evaluation=ledger, score_floor=FLOOR, design_sha256=digest(PLAN),
                            predictions={arm: digest(CACHE / f"{arm}.json") for arm in ARMS},
                            interpretation="Identical public training parents and update budgets. Model comparison "
                                           "uses only real development acquisitions. No Fil Rouge training or "
                                           "validation hyperparameter selection. Deployment requires a further "
                                           "frozen evaluation with no loss of prior matches or counts."))
    print({k: round(v["all"]["f1"], 4) for k, v in ledger.items()}, flush=True)


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
        parser.error("Select a trained arm.")
