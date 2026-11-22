"""Evaluate appearance verification before a count-preserving syringe replacement."""
import argparse
from itertools import product
from pathlib import Path

import cv2

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.pipeline_extensions import verified_config
from scripts.occlusion_detector import ART, raster
from scripts.overlap_candidate_features import WORK as UPSTREAM_FEATURES, inputs, load
from scripts.pair_relocalization import POLICY as PAIR, check, predict, rank_pairs
from scripts.pair_relocalization import frozen as pair_model
from scripts.pair_relocalization_probe import PROBE, priority
from scripts.relocalization_selector import snapshot
from scripts.sam3_study import configure_cpu
from scripts.syringe_box_quality import frozen as quality_model
from scripts.syringe_object_features import CropSemantics, encode_samples
from scripts.syringe_object_verifier import LOCK as VERIFIER, partition_audit, score_candidates, sealed

WORK = Path("data/replacement-appearance")
REFERENCE = ART / "replacement_appearance_reference.json"
POLICY = ART / "replacement_appearance_policy.json"
RESULT = ART / "replacement_appearance_results.json"
JOINT_FLOORS = (.85, .90, .94, .97, .99)
TEXT_FLOORS = (-.02, -.012, -.006, -.003, 0., .003, .006, .012, .02)
TYPED_FLOORS = (-1., 0., .50, .75, 1.5, 2., 3.)
ARMS = ("targeted-proposal", "targeted-pair", "typed-proposal")


def gate_choices(base, choices, appearance, arm, floor):
    results = {}
    for key, options in choices.items():
        margins = appearance[key]
        gated = []
        for p in options:
            if p["scores"]["joint"] < JOINT_FLOORS[0]:
                gated.append(p)
                continue
            proposed = margins[tuple(p["bbox"])]
            if arm == "targeted-pair":
                incumbent = margins[tuple(base[key][p["slot"]]["bbox"])]
                score = min(proposed["targeted-text"], incumbent["targeted-text"])
            elif arm == "targeted-proposal":
                score = proposed["targeted-text"]
            else:
                score = proposed["typed-hybrid" if arm == "typed-proposal" else "release-veto"]
            # Leave the ownership scores in place even when a candidate is vetoed.
            gated.append(p if score >= floor else dict(p, scores=dict(p["scores"], joint=-1.)))
        results[key] = gated
    return results


def apply(packet, rule):
    base = packet["base"]
    choices = packet["choices"]
    if rule["arm"] != "off":
        margins = {key: {tuple(p["bbox"]): p for p in rows} for key, rows in packet["appearance"].items()}
        choices = gate_choices(base, choices, margins, rule["arm"], rule["cutoff"])
    return predict(base, choices, dict(arm="joint", floor=rule["joint"], link=None))


def screen_examples(row, items):
    """Choose one localized RF proposal per annotated category for diagnostic scoring."""
    exemplar = {}
    references = [(p["bbox"], "syringe") for p in row["objects"]]
    references.extend((p["bbox"], p["kind"]) for p in row.get("lookalikes", []))
    for bbox, kind in references:
        candidate = max(items, key=lambda p: box_iou(bbox, p["bbox"]), default=None)
        if candidate is None:
            continue
        overlap = box_iou(bbox, candidate["bbox"])
        prior = exemplar.get(kind, {}).get("iou", 0.)
        if overlap >= (.50 if kind == "syringe" else .35) and overlap > prior:
            exemplar[kind] = dict(bbox=candidate["bbox"], iou=overlap, kind=kind)
    return list(exemplar.values())


def evidence(rows, base, bank, semantics):
    choices = rank_pairs(base, bank, pair_model()["heads"], quality_model()["models"])
    verifier = sealed()["model"]
    appearance, references = {}, {}
    for row in rows:
        key = row["id"]
        examples = screen_examples(row, bank[key]["proposed"])
        references[key] = examples
        measured = {tuple(p["bbox"]) for p in examples}
        for p in choices[key]:
            if p["scores"]["joint"] >= JOINT_FLOORS[0]:
                measured.update((tuple(p["bbox"]), tuple(base[key][p["slot"]]["bbox"])))
        pixels = {tuple(p["bbox"]):p["features"] for field in ("base", "proposed") for p in bank[key][field]}
        requests = [dict(bbox=list(b), features=pixels[b]) for b in sorted(measured)]
        encoded = encode_samples(raster(row), requests, None, semantics, True)
        scores = score_candidates({key: encoded}, verifier)[key]
        appearance[key] = [dict(bbox=list(b), **p) for b,p in scores.items()]
    return dict(base=base, choices=choices, appearance=appearance, examples=references)


def fixed_rules():
    policy = read_json(PROBE)
    if policy["policy"] != digest(PAIR):
        raise ValueError("The original joint comparison does not match its pair model.")
    joint = policy["selected"]["joint"]["rule"]["floor"]
    rule = verified_config(Path.cwd(), RELEASE)["components"][0]
    if rule["name"] != "semantic-best":
        raise ValueError("Expected the released semantic margin.")
    return {
        "ungated-joint": dict(arm="off", joint=joint, cutoff=None),
        "released-veto": dict(arm="released", joint=joint, cutoff=rule["bounds"][1]),
        "targeted-zero": dict(arm="targeted-proposal", joint=joint, cutoff=0.),
    }


def compare(rows, packet, before, rule):
    filtered = apply(packet, rule)
    guard = check(rows, packet["base"], before, filtered)
    if rule["arm"] == "off":
        missing = {t: [] for t in guard["gained"]}
    else:
        unfiltered = apply(packet, dict(rule, arm="off"))
        original = check(rows, packet["base"], before, unfiltered)
        missing = {t: [entry for entry in gains if entry not in guard["gained"][t]]
                   for t,gains in original["gained"].items()}
    return filtered, dict(**guard, gated_out_gains=missing,
                          preserves_unfiltered_gains=not any(missing.values()))


def select(rows, packet):
    before = snapshot(rows, packet["base"])
    trials, finalists, active = [], {}, {}
    for arm in ARMS:
        cutoffs = TYPED_FLOORS if arm == "typed-proposal" else TEXT_FLOORS
        run = []
        for joint, cutoff in product(JOINT_FLOORS, cutoffs):
            rule = dict(arm=arm, cutoff=cutoff, joint=joint)
            _, verdict = compare(rows, packet, before, rule)
            run.append(dict(rule=rule, **verdict))
        eligible = [t for t in run if t["passes"] and t["preserves_unfiltered_gains"] and any(t["gained"].values())]
        finalists[arm] = max(eligible, key=selection_priority, default=None)
        changing = [t for t in run if t["changed"]]
        active[arm] = max(changing or run, key=selection_priority)
        trials.extend(run)
    winner = max((t for t in finalists.values() if t), key=selection_priority, default=None)
    return dict(trials=trials, finalists=finalists, active=active, winner=winner)


def selection_priority(trial):
    rule = trial["rule"]
    return priority(dict(trial, rule=dict(floor=rule["joint"], link=rule["cutoff"])))


def bindings():
    paths = (Path(__file__), REFERENCE, RELEASE, PAIR, PROBE, VERIFIER, UPSTREAM_FEATURES/"features-development.json",
             WORK/"semantic-development.json", Path("scripts/replacement_appearance_reserve.py"))
    return {str(p): digest(p) for p in paths}


def freeze():
    from scripts.replacement_appearance_reserve import signature

    if POLICY.exists():
        raise FileExistsError(POLICY)
    dev, _, _ = inputs("development")
    fit, _, _ = inputs("fit")
    sizes, overlaps = partition_audit(fit, dev, read_json(REFERENCE)["rows"])
    cache = read_json(WORK/"semantic-development.json")
    expected = signature((UPSTREAM_FEATURES/"features-development.json",))
    if cache["binding"] != expected:
        raise ValueError("The development semantic cache does not match its inputs.")
    packet = cache["packet"]
    chosen = select(dev, packet)
    control = fixed_rules()
    before = snapshot(dev, packet["base"])
    write_json(POLICY, dict(binding=bindings(), sizes=sizes, overlaps=overlaps, selection=chosen, controls=control,
                            development={k:compare(dev, packet, before, r)[1] for k,r in control.items()}))
    print("Development winner", chosen["winner"], flush=True)
    for arm, t in chosen["active"].items():
        print(arm, t["rule"], t["changed"], t["passes"], flush=True)


def frozen():
    state = read_json(POLICY)
    if state["binding"] != bindings():
        raise ValueError("Frozen appearance selection evidence changed.")
    return state


def example_scores(packet):
    result = []
    for key, items in packet["examples"].items():
        for p in items:
            margins = next(s for s in packet["appearance"][key] if s["bbox"] == p["bbox"])
            result.append(dict(id=key, **p, scores={k:v for k,v in margins.items() if k != "bbox"}))
    return result


def score():
    from scripts.replacement_appearance_reserve import cohorts

    state = frozen()
    selected = state["selection"]
    rules = {**state["controls"], **{arm:t["rule"] for arm,t in selected["active"].items()}}
    if selected["winner"]:
        rules["selected"] = selected["winner"]["rule"]
    result = {}
    for role, (rows, packet) in cohorts():
        base = packet["base"]
        before = snapshot(rows, base)
        variants = {}
        for arm, rule in rules.items():
            predictions, guard = compare(rows, packet, before, rule)
            variants[arm] = dict(guard=guard, metrics=evaluate(rows, predictions, 0.),
                                 changes={k:[p for p in values if "prior_bbox" in p] for k,values in predictions.items()
                                          if any("prior_bbox" in p for p in values)})
        result[role] = dict(baseline=evaluate(rows, base, 0.), variants=variants, examples=example_scores(packet))
        print(role, {k:(v["guard"]["changed"],v["metrics"]["tp"],v["guard"]["passes"])
                     for k,v in variants.items()}, flush=True)
    qualified = bool(selected["winner"] and all(
        r["variants"]["selected"]["guard"]["passes"] and r["variants"]["selected"]["guard"]["preserves_unfiltered_gains"]
        for r in result.values()) and any(result["reserve"]["variants"]["selected"]["guard"]["gained"].values()))
    write_json(RESULT, dict(policy=digest(POLICY), qualified=qualified, cohorts=result))
    print("Qualified", qualified, flush=True)


def collect(roles):
    from scripts.replacement_appearance_reserve import batches, store

    if roles != ["development"]:
        frozen()
    configure_cpu()
    cv2.setNumThreads(1)
    semantics = CropSemantics()
    for role, (rows, base, bank, files) in batches(roles):
        packet = evidence(rows, base, bank, semantics)
        store(role, packet, files)
        print(role, len(rows), sum(map(len,packet["appearance"].values())), "scored crops", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score", "development", "evaluation"))
    stages = dict(freeze=freeze, score=score, development=lambda: collect(["development"]),
                  evaluation=lambda: collect(["reserve","exposed"]))
    stages[cli.parse_args().stage]()
