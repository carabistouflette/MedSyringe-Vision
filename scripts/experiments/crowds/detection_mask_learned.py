"""Transfer appearance heads trained on reviewed fit crops to class-agnostic proposals."""
import argparse
from itertools import product
from pathlib import Path

from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_audits.detection_appearance_extension import FIT_CACHE, feature_heads, fit_snapshot
from scripts.experiments.detector_training.detection_context_data import RELEASE as CONTEXT
from scripts.experiments.detector_audits.detection_crop_semantics import VisualSemantics, feature_cache, read_features, scores
from scripts.experiments.transfer.detection_device_scene_data import MANIFEST as DEVICES
from scripts.experiments.detector_audits.detection_guided_geometry import references
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, inputs
from scripts.experiments.detector_training.detection_hillclimb_rank import minimal
from scripts.experiments.crowds.detection_mask_verification import CACHE as CANDIDATES, OUTPUT as TEXT_RESULTS
from scripts.experiments.detector_audits.detection_semantics_audit import crop_frame
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

FEATURES = DATA / "proposals/mask-inherited-fit-features.json"
OUTPUT = ART / "mask_learned_development.json"
LIMIT_PER_IMAGE = 10


def binding():
    return {str(p): digest(p) for p in (
        Path("scripts/experiments/crowds/detection_mask_learned.py"), CANDIDATES, TEXT_RESULTS,
        FIT_CACHE, feature_cache("fit"), CONTEXT, DEVICES)}


def collect():
    configure_cpu()
    prev = read_json(TEXT_RESULTS)
    if digest(CANDIDATES) != prev["features_sha256"]:
        raise ValueError("The previously measured candidate matrix changed.")
    original = read_json(CANDIDATES)["scopes"]
    key = binding()
    saved = read_json(FEATURES) if FEATURES.exists() else dict(inputs=key, inputs_by_scope={})
    if saved["inputs"] != key:
        raise ValueError("The input proposal features no longer match their source experiment.")
    encoder = None
    for scope in DEVELOPMENT:
        frames = saved["inputs_by_scope"].setdefault(scope, {})
        for row in inputs(scope):
            identifier = row["id"]
            if identifier in frames:
                continue
            shortlist = sorted(original[scope][identifier], key=lambda p: (
                p["primary"] or p["open_votes"] > 0, p["margin"], p["score"]), reverse=True)[:LIMIT_PER_IMAGE]
            if shortlist:
                if encoder is None:
                    encoder = VisualSemantics()
                features = crop_frame(encoder, source(row), {"floor": 0., "ranked": shortlist})
            else:
                features = dict(candidates=[], values={})
            frames[identifier] = features
            write_json(FEATURES, saved)
            print("Learned mask verification", scope, len(frames), flush=True)


def train():
    main, previous = fit_snapshot()
    rows = [{**r, "split": "fit"} for r in (*main, *previous)]
    protected = [r for path in (CONTEXT, DEVICES) for r in read_json(path)["rows"]]
    if leakage([*rows, *protected]):
        raise ValueError("A learned crop training image overlaps a reserved real source.")
    groups = {r["id"]: r["group"] for r in rows}
    src = read_json(FIT_CACHE)
    if any(digest(f) != h for f, h in src["key"]["files"].items()):
        raise ValueError("The supervised training feature cache changed.")
    cached = {**read_features("fit"), **src["frames"]}
    if set(cached) != set(groups):
        raise ValueError("Every learned crop must have a reviewed source with a known fit role.")
    return feature_heads(cached, groups), len(rows), len(set(groups.values()))


def select(original, features, values, policy):
    logit, sem, witness, scope = policy
    if original and scope == "empty":
        return original
    candidate = [(p, s) for p, s in zip(features, values, strict=True)
                 if s >= logit and p["margin"] >= sem and
                 (witness == "none" or p["primary"] or p["open_votes"])]
    top = max(candidate, key=lambda item: item[1], default=None)
    return [*original, dict(bbox=top[0]["bbox"], score=.08)] if top else original


def measure():
    if OUTPUT.exists():
        raise ValueError("The controlled learned comparison was already measured.")
    saved = read_json(FEATURES)
    if saved["inputs"] != binding():
        raise ValueError("A fitted crop input changed.")
    layouts = saved["inputs_by_scope"]
    trained, images, authors = train()
    base = {scope: references(scope) for scope in DEVELOPMENT}
    refs = {scope: inputs(scope) for scope in DEVELOPMENT}
    if any(set(layouts[k]) != set(base[k]) for k in DEVELOPMENT):
        raise ValueError("The measurement requires every protected development scene.")
    original = {scope: minimal(refs[scope], base[scope], 0.) for scope in DEVELOPMENT}
    configurations = tuple(product((.2, .6, 1., 1.5, 2.), (-.005, .005, .025),
                                   ("none", "any"), ("empty", "any")))
    qualified = []
    for name, model in trained.items():
        logits = {scope: {k: scores(f, model["name"], model["head"]) if f["candidates"] else []
                          for k, f in layouts[scope].items()} for scope in DEVELOPMENT}
        for policy in configurations:
            result = {}
            for scope in DEVELOPMENT:
                predicted = {
                    k: select(ranked, layouts[scope][k]["candidates"], logits[scope][k], policy)
                    for k, ranked in base[scope].items()}
                measured = minimal(refs[scope], predicted, 0.)
                guard = compare(original[scope], measured)
                if not guard["passed"] or measured["fp"] > original[scope]["fp"]:
                    break
                result[scope] = dict(tp=measured["tp"], fp=measured["fp"], fn=measured["fn"],
                                     recovered=guard["recovered_matches"])
            if len(result) == len(DEVELOPMENT):
                qualified.append(dict(model=name, policy=policy, results=result))
    qualified.sort(key=lambda t: (min(m["recovered"] for m in t["results"].values()),
                                  sum(m["recovered"] for m in t["results"].values())), reverse=True)
    ruler = qualified[0] if qualified else None
    write_json(OUTPUT, dict(fit_images=images, fit_groups=authors, trials=len(trained)*len(configurations),
                            safe=len(qualified), leading=qualified[:10],
                            selected_checkpoint=trained[ruler["model"]] if ruler else None,
                            demonstration_sha256=digest(FEATURES),
                            scope="Frozen crop embeddings, reviewer-labeled fit crops and development-only "
                                  "detector decisions. An external qualification is required for activation."))
    print("Learned mask verifier", ruler, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("collect", "measure"))
    (collect if p.parse_args().action == "collect" else measure)()
