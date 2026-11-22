"""Export, replay-check and activate the measured RF-DETR recovery branch."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from medsyringe.detection import auxiliary_detector
from medsyringe.detection.auxiliary_detector import MANIFEST, VerifiedRecovery
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from medsyringe.detection.verification import BACKBONE
from scripts.experiments.detector_training.optimize_detector import FINAL, counters
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_compare import SELECTION, checkpoint_record
from scripts.experiments.transfer.rfdetr_verifier import ART, AUDITS, WORK, baseline_for, inputs
from scripts.experiments.transfer.rfdetr_verifier_refine import FROZEN, protocol_inputs

WEIGHTS = WORK / "verified-recovery.pt"
EVIDENCE = ART / "rfdetr_verified_runtime.json"
AUDIT_DECISION = ART / "rfdetr_refined_release_decision.json"


def locked_settings():
    decision, refined, hybrid = (read_json(p) for p in (AUDIT_DECISION, FROZEN, SELECTION))
    if (not decision["eligible_for_release"] or decision["protocol_sha256"] != digest(FROZEN)
            or refined["protocol"] != protocol_inputs()):
        raise ValueError("The refined candidate lacks a passing frozen study")
    head, = refined["heads"]
    locked = hybrid["locked"]
    if head["encoder"] != "mobile" or head["padding"] != 0 or locked["view"] != "full_and_tiles":
        raise ValueError("The runtime supports the validated tight MobileNet profile with RF-DETR tiles")
    return head, locked


def export():
    from rfdetr import RFDETRSegNano

    _, locked = locked_settings()
    checkpoint = checkpoint_record(locked["run"])["checkpoint"]
    model = RFDETRSegNano.from_checkpoint(checkpoint, device="cpu", segmentation_head=False)
    snapshot = model.inference(compile=True, batch_size=1, compile_backend="torchscript")
    graph = snapshot._native_model.model.inference_model
    temporary = WEIGHTS.with_suffix(".part")
    graph.save(str(temporary))
    temporary.replace(WEIGHTS)
    print("Exported", WEIGHTS, digest(WEIGHTS), flush=True)


def verify():
    head, locked = locked_settings()
    resolution = checkpoint_record(locked["run"])["train"]["resolution"]
    config = dict(weights=str(WEIGHTS), weights_sha256=digest(WEIGHTS), resolution=resolution,
                  options=locked["options"], head={k:head[k] for k in ("weights","bias","cutoff")},
                  encoder_sha256=digest(BACKBONE), runtime_sha256=digest(auxiliary_detector.__file__),
                  base_sha256=digest(FINAL))
    floor = read_json(FINAL)["cutoff"]
    runner = VerifiedRecovery(Path("."), config)
    assessment = {}
    for cohort in AUDITS:
        expected = read_json(ART / f"rfdetr_refined_{cohort}.json")
        if expected["protocol_sha256"] != digest(FROZEN):
            raise ValueError("Runtime comparison requires the frozen audit")
        rows = inputs(cohort)
        baseline = baseline_for(cohort, rows)
        outputs, elapsed = {}, []
        for row in rows:
            image = source(row)
            started = perf_counter()
            outputs[row["id"]] = runner.apply(image, baseline[row["id"]], floor)
            elapsed.append(1000*(perf_counter()-started))
        actual = evaluate(rows, outputs, floor)
        expected_stats = expected["verified"]
        guard = compare(expected_stats, actual)
        original = compare(expected["baseline"], actual)
        parity = (guard["passed"] and original["passed"] and
                  actual["tp"] == expected_stats["tp"] and actual["fp"] == expected_stats["fp"] and
                  sorted(p["id"] for p in actual["false_positives"]) ==
                  sorted(p["id"] for p in expected_stats["false_positives"]))
        assessment[cohort] = dict(parity=parity, metrics=counters(actual),
                                  guard=original, expected_guard=guard, median_added_ms=median(elapsed))
        print(cohort, "parity", parity, counters(actual), flush=True)
    passed = all(c["parity"] for c in assessment.values())
    write_json(EVIDENCE, dict(passed=passed, cohorts=assessment,
                              study_sha256=digest(AUDIT_DECISION), release_sha256=digest(FINAL),
                              runtime_sha256=config["runtime_sha256"], weights_sha256=config["weights_sha256"]))
    if not passed:
        raise ValueError("The executable export does not reproduce the frozen audit")
    write_json(MANIFEST, dict(**config, evidence=str(EVIDENCE), evidence_sha256=digest(EVIDENCE)))
    print("Activated", MANIFEST, flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export","verify"])
    args=parser.parse_args()
    torch.set_num_threads(4)
    (export if args.stage=="export" else verify)()


if __name__=="__main__":
    main()
