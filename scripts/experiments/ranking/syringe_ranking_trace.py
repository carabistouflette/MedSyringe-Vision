"""Trace the frozen syringe recovery rules without fitting or changing a decision."""

from collections import Counter

import cv2

from medsyringe.data import digest, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import match
from scripts.experiments.occlusion.occlusion_detector import raster
from scripts.experiments.ranking.overlap_candidate_features import annotate_embedding, candidate_pool, inputs, load
from scripts.experiments.ranking.overlap_ranking import CAP, LOCK, conflict, predict_evidence, sealed, select
from scripts.experiments.transfer.rfdetr_verifier import DinoEncoder
from scripts.experiments.expanded.sam3_study import configure_cpu
from scripts.experiments.identity.syringe_object_features import feature_store, rules
from scripts.experiments.identity.syringe_object_references import WORK
from scripts.experiments.identity.syringe_object_reserve import evidence

OUTPUT = WORK / "ranking-trace.json"
MINIMUM_IOU = .50


def trace_frame(incumbent, ranked, relation, rule, selected):
    count = len(incumbent)
    candidates = [*incumbent, *ranked]
    ranker = rule["ranker"]
    chosen_boxes = {tuple(p["bbox"]) for p in selected}

    def conflicts(i, j):
        affinity = relation.get(tuple(sorted((i, j))))
        return conflict(candidates[i]["bbox"], candidates[j]["bbox"], affinity, rule)

    target_indices = range(count, len(candidates))
    incumbent_conflicts = {i: [j for j in range(count) if conflicts(i, j)] for i in target_indices}
    eligible = [i for i in target_indices if candidates[i][ranker] >= rule["floor"] and not incumbent_conflicts[i]]
    order = sorted(eligible, key=lambda i: (-candidates[i][ranker], i))
    considered = set(order[:CAP])
    chosen = [i for i in order if tuple(candidates[i]["bbox"]) in chosen_boxes]
    trace = []
    for i in target_indices:
        blocking = []
        if candidates[i][ranker] < rule["floor"]:
            stage = "below-rank-cutoff"
        elif incumbent_conflicts[i]:
            stage, blocking = "incumbent-conflict", incumbent_conflicts[i]
        elif i not in considered:
            stage = "candidate-budget"
        elif i in chosen:
            stage = "selected"
        else:
            blocking = [j for j in chosen if conflicts(i, j)]
            stage = "proposal-conflict"
            if not blocking:
                raise ValueError("Selected set did not explain an eligible proposal.")
        trace.append(dict(stage=stage, cutoff=rule["floor"],
                          blockers=[dict(bbox=candidates[j]["bbox"],
                                         learned_distinctness=relation.get(tuple(sorted((i, j)))))
                                    for j in blocking]))
    return trace


def trace_cohort(rows, base, bank, heads, cohort):
    ranked, relation = predict_evidence(rows, bank, heads)
    saved = feature_store(cohort, rows)["routes"]
    baseline_matches = match(rows, base, MINIMUM_IOU)[1]
    stages = {}
    for route, rule in rules().items():
        selected = {key: select(boxes, ranked[key], relation[key], rule) for key, boxes in base.items()}
        if selected != saved[route]:
            raise ValueError(f"The replay did not reproduce the saved {cohort} {route} decisions.")
        output = {key: [*boxes, *selected[key]] for key, boxes in base.items()}
        matched = match(rows, output, MINIMUM_IOU)[1]
        counts, cases = Counter(), {}
        for row in rows:
            key = row["id"]
            reasons = trace_frame(base[key], ranked[key], relation[key], rule, selected[key])
            missed = []
            for i, gold in enumerate(row["objects"]):
                if i in baseline_matches[key]:
                    continue
                candidates = [dict(bbox=p["bbox"], iou=box_iou(gold["bbox"], p["bbox"]),
                                   rf_confidence=p["score"], point_score=p["point"], rank_score=p["rank"],
                                   corroboration=source["corroboration"], view=source["view"], **reason)
                              for p, source, reason in zip(ranked[key], bank[key]["proposed"], reasons, strict=True)]
                candidates.sort(key=lambda p: -p["iou"])
                valid = [p for p in candidates if p["iou"] >= MINIMUM_IOU]
                if i in matched[key]:
                    counts["recovered"] += 1
                    continue
                stage = valid[0]["stage"] if valid else "no-valid-candidate-in-bank"
                counts[stage] += 1
                missed.append(dict(target=i, bbox=gold["bbox"], best_candidate_stage=stage,
                                   valid_candidates=valid, best_candidates=candidates[:3]))
            if missed:
                cases[key] = missed
        stages[route] = dict(rule=rule, counts=counts, missed=cases)
    return stages


def run():
    configure_cpu()
    cv2.setNumThreads(1)
    heads = sealed()["heads"]
    dev, baseline, raw = inputs("development")
    development = trace_cohort(dev, baseline, load("development", dev, baseline, raw), heads, "development")
    rows, base, raw, upstream = evidence()
    encoder, bank = DinoEncoder(), {}
    for row in rows:
        key, photo = row["id"], raster(row)
        original = [dict(bbox=p["bbox"], score=p["score"], corroboration=0., view="released") for p in base[key]]
        requested = [*original, *candidate_pool(raw[key])]
        encoded = annotate_embedding(encoder, photo, requested)
        bank[key] = dict(base=encoded[:len(original)], proposed=encoded[len(original):])
    exposed = trace_cohort(rows, base, bank, heads, "fresh")
    write_json(OUTPUT, dict(ranking_policy=digest(LOCK), frozen_sources=upstream, iou=MINIMUM_IOU,
                           development=development, exposed_diagnostic=exposed,
                           note="A post-evaluation diagnostic. Frozen detection choices reproduced exactly."))
    for role, results in (("development", development), ("exposed_diagnostic", exposed)):
        for name, state in results.items():
            print(role, name, dict(state["counts"]), flush=True)


if __name__ == "__main__":
    run()
