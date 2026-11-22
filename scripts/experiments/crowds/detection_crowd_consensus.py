"""Evaluate corroborated RF-DETR additions while retaining every accepted hybrid box."""
import argparse
from itertools import product
from math import sqrt

from medsyringe.detection.additive import duplicates
from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.workers.appearance_text_encoder import TextMarginEncoder
from scripts.experiments.crowds.detection_crowd_training_evaluate import OUTPUT as TRAINING, path as trained_path
from scripts.experiments.detector_audits.detection_deskew_additive import CACHE as NATURAL
from scripts.experiments.detector_audits.detection_deskew_study import CACHE as PARALLEL, OUTPUT as OVERLAP
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, inputs
from scripts.experiments.detector_audits.detection_mosaic_development import MANIFEST as MOSAIC_LABELS
from scripts.experiments.detector_audits.detection_mosaic_study import stored
from scripts.experiments.detector_audits.detection_parallel_development import OUTPUT as PARALLEL_LABELS
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_verifier import expanded
from scripts.experiments.expanded.sam3_study import configure_cpu

CACHE = DATA / "proposals/crowd-supervised-consensus.json"
OUTPUT = ART / "crowd_supervised_consensus_development.json"
SCOPES = ("ordinary", "parallel", "mosaic")
VIEWS = ("suppressed", "deskewed")
ARMS = ("separated", "overlapping")


def baseline(scope):
    if scope == "parallel":
        result = read_json(OVERLAP)
        if digest(PARALLEL) != result["predictions_sha256"]:
            raise ValueError("The original parallel detections changed.")
        return read_json(PARALLEL_LABELS)["rows"], {k: f["baseline"]
                                                  for k, f in read_json(PARALLEL)["frames"].items()}
    if scope == "mosaic":
        return read_json(MOSAIC_LABELS)["rows"], {k: f["boxes"] for k, f in stored("baseline")["frames"].items()}
    rows = inputs("public-development")
    frames = read_json(NATURAL)["frames"]
    return rows, {r["id"]: frames[r["id"]]["baseline"] for r in rows}


def inputs_for(scope):
    references, before = baseline(scope)
    hashes = read_json(TRAINING)["collection_sha256"]
    forecasts = {}
    for arm in ARMS:
        file = trained_path(arm)
        if digest(file) != hashes[arm]:
            raise ValueError("The candidate checkpoint forecasts changed.")
        forecasts[arm] = read_json(file)["frames"][scope]
    if set(before) != {r["id"] for r in references} or not set(before).issubset(forecasts[ARMS[0]]):
        raise ValueError(f"Complete the fixed baseline and both model forecasts for {scope}.")
    hashes = {str(k): digest(k) for k in (TRAINING, __file__)}
    hashes.update(refs={r["id"]: [r["sha256"], [o["bbox"] for o in r["objects"]]] for r in references}, before=before)
    return references, before, forecasts, hashes


def feature_pairs(known, left, right):
    if len(known) < 2:
        return []
    pairs = []
    for first, second in product(left, right):
        lower = min(first["score"], second["score"])
        iou = box_iou(first["bbox"], second["bbox"])
        if lower < .16 or iou < .40:
            continue
        box = [(a + b) / 2 for a, b in zip(first["bbox"], second["bbox"], strict=True)]
        if any(duplicates(other["bbox"], box) for other in known):
            continue
        pairs.append(dict(bbox=box, score=sqrt(first["score"] * second["score"]),
                          floor=lower, support_iou=iou))
    ranked = sorted(pairs, key=lambda p: p["score"] * p["support_iou"], reverse=True)
    return suppress(ranked, .65)[:8]


def collect(scope):
    configure_cpu()
    rows, before, forecasts, binding = inputs_for(scope)
    data = read_json(CACHE) if CACHE.exists() else {}
    record = data.setdefault(scope, dict(binding=binding, frames={}))
    if record["binding"] != binding:
        raise ValueError("The consensus candidates came from different development inputs.")
    encoder = None
    for row in rows:
        identifier = row["id"]
        if identifier in record["frames"]:
            continue
        variants = {}
        for view in VIEWS if scope == "parallel" else VIEWS[:1]:
            a, b = (forecasts[arm][identifier][view] for arm in ARMS)
            variants[view] = feature_pairs(before[identifier], a, b)
        if any(variants.values()):
            if encoder is None:
                encoder = TextMarginEncoder()
            image = source(row)
            for entries in variants.values():
                if entries:
                    patches = [crop(image, expanded(p["bbox"], .08)) for p in entries]
                    scores = encoder.encode(patches)["prompt"]
                    for p, margin in zip(entries, scores[:, 0] - scores[:, 1], strict=True):
                        p["margin"] = float(margin)
        record["frames"][identifier] = variants
        write_json(CACHE, data)
    print("Crowd consensus", scope, len(record["frames"]), "complete", flush=True)


def predict(known, variants, setting):
    view, confidence, minimum_agreement, semantic, cap = setting
    proposals = variants.get(view, variants["suppressed"])
    added = [p for p in proposals if p["floor"] >= confidence and p["support_iou"] >= minimum_agreement
             and p["margin"] >= semantic]
    added.sort(key=lambda p: p["score"], reverse=True)
    return [*known, *(dict(bbox=p["bbox"], score=.1) for p in suppress(added, .45)[:cap])]


def measure():
    if OUTPUT.exists():
        raise ValueError("The crowd-consensus pilot has already been measured.")
    trial = {}
    collected = read_json(CACHE)
    for scope in SCOPES:
        rows, base, _, binding = inputs_for(scope)
        record = collected[scope]
        if record["binding"] != binding or set(record["frames"]) != set(base):
            raise ValueError("Incomplete consensus development inputs.")
        trial[scope] = rows, base, record["frames"], evaluate(rows, base, 0.)
    variants = tuple(product(VIEWS, (.16, .20, .28, .36), (.40, .55, .70), (-.005, .005, .015), (1, 2)))
    records_ = []
    for setting in variants:
        result = {}
        for scope, (rows, base, frames, baseline_) in trial.items():
            result_boxes = {k: predict(p, frames[k], setting) for k, p in base.items()}
            score = evaluate(rows, result_boxes, 0.)
            retained = compare(baseline_, score)
            spoiled = [r["id"] for r in rows if len(base[r["id"]]) == len(r["objects"])
                       and len(result_boxes[r["id"]]) != len(r["objects"])]
            result[scope] = dict(metrics=short(score), retention=retained, spoiled_counts=spoiled,
                                 safe=retained["passed"] and not spoiled and score["fp"] <= baseline_["fp"])
        records_.append(dict(setting=setting, result=result, safe=all(p["safe"] for p in result.values())))
    accepted = [item for item in records_ if item["safe"]]
    accepted.sort(key=lambda row: (sum(v["retention"]["recovered_matches"] for v in row["result"].values()),
                                   row["setting"][-1] == 1, row["setting"][0] == "suppressed"), reverse=True)
    write_json(OUTPUT, dict(variants=len(variants), safe_development=len(accepted), leading=accepted[:12],
                            baseline={k: short(v[3]) for k, v in trial.items()},
                            evidence_sha256=digest(CACHE),
                            scope="Paired supervised heads nominate an additional object only on scenes with at least "
                                  "two accepted hybrid boxes. A separately trained SigLIP 2 model checks the crop. "
                                  "This development selection does not open any real reserve."))
    print("Supervised crowd fusion", len(accepted), "/", len(variants), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    parser.add_argument("--scope", choices=SCOPES)
    args = parser.parse_args()
    if args.action == "measure":
        measure()
    elif args.scope:
        collect(args.scope)
    else:
        parser.error("Choose a fixed development cohort.")
