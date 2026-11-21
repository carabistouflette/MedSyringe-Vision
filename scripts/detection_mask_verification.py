"""Require corroborating appearance evidence for new class-agnostic mask boxes."""
import argparse
from itertools import product
from pathlib import Path

from medsyringe.additive import overlap_fraction
from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.recall_guard import compare
from scripts.appearance_text_encoder import TextMarginEncoder
from scripts.detection_guided_geometry import references
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, inputs, loaded
from scripts.detection_hillclimb_rank import minimal
from scripts.detection_mask_retrieval import STUDY, path
from scripts.detection_open_models import OPEN_MODELS, open_views
from scripts.precision_study import source
from scripts.rfdetr_verifier import expanded
from scripts.sam3_study import configure_cpu

OUTPUT = ART / "mask_verification_development.json"
CACHE = DATA / "proposals/mask-verification-development.json"
LIMIT_PER_IMAGE = 48


def duplicate(box, known):
    return any(box_iou(box, item["bbox"]) >= .42 or overlap_fraction(box, item["bbox"]) >= .80
               for item in known)


def candidates(frame, accepted):
    options = [p for p in frame["grids"]["16"] if p["stability"] >= .80 and not duplicate(p["bbox"], accepted)]
    return options[:LIMIT_PER_IMAGE]


def support(box, auxiliary, baseline):
    corroborates = [any(box_iou(box, p["bbox"]) >= .40 for p in choices) for choices in auxiliary]
    old = max((box_iou(box, p["bbox"]) for p in baseline if p.get("original_score", p["score"]) >= .23),
              default=0.)
    return dict(primary=old >= .40, open_votes=sum(corroborates))


def binding():
    return dict(study=digest(STUDY), implementation=digest(Path(__file__)),
                proposals={scope: digest(path(scope)) for scope in DEVELOPMENT})


def collect():
    configure_cpu()
    proof = binding()
    saved = read_json(CACHE) if CACHE.exists() else dict(inputs=proof, scopes={})
    if saved["inputs"] != proof:
        raise ValueError("Proposal or feature settings changed.")
    encoder = None
    for scope in DEVELOPMENT:
        chosen = saved["scopes"].setdefault(scope, {})
        published = references(scope)
        base = loaded(scope, "baseline")
        extra = [open_views(scope, m)["wide"] for m in OPEN_MODELS]
        retrieval = read_json(path(scope))["frames"]
        for row in inputs(scope):
            key = row["id"]
            if key in chosen:
                continue
            present = candidates(retrieval[key], published[key])
            scored = []
            if present:
                if encoder is None:
                    encoder = TextMarginEncoder()
                picture = source(row)
                chips = [crop(picture, expanded(p["bbox"], .08)) for p in present]
                margins = []
                for start in range(0, len(chips), 8):
                    features = encoder.encode(chips[start:start+8])["prompt"]
                    margins.extend((features[:, 0]-features[:, 1]).tolist())
                for proposal, value in zip(present, margins, strict=True):
                    witnesses = support(proposal["bbox"], [m[key] for m in extra], base[key]["ranked"])
                    scored.append({**proposal, **witnesses, "margin": value})
            chosen[key] = scored
            write_json(CACHE, saved)
            print("Mask appearance verification", scope, len(chosen), "/", len(published), flush=True)


def proposal_filter(row, settings):
    minimum_score, margin, ratio, agreement, _, minimum_stability = settings
    if row["score"] < minimum_score or row["margin"] < margin or row["elongation"] < ratio:
        return False
    if row["stability"] < minimum_stability:
        return False
    if agreement == "primary":
        return row["primary"]
    if agreement == "external":
        return row["open_votes"] >= 1
    if agreement == "two":
        return row["open_votes"] + row["primary"] >= 2
    return True


def detect(accepted, alternatives, policy):
    mode = policy[4]
    if (mode == "empty" and accepted) or (mode == "multiple" and len(accepted) < 2):
        return accepted
    possible = [p for p in alternatives if proposal_filter(p, policy)]
    best = max(possible, key=lambda p: (p["margin"], p["score"]), default=None)
    return [*accepted, dict(bbox=best["bbox"], score=.08)] if best else accepted


def measure():
    if OUTPUT.exists():
        raise ValueError("Development selection has already been run.")
    saved = read_json(CACHE)
    before = {scope: references(scope) for scope in DEVELOPMENT}
    data = {scope: inputs(scope) for scope in DEVELOPMENT}
    if saved["inputs"] != binding() or any(set(saved["scopes"][k]) != set(before[k]) for k in DEVELOPMENT):
        raise ValueError("The candidate appearance records are incomplete or changed.")
    baseline = {scope: minimal(data[scope], before[scope], 0.) for scope in DEVELOPMENT}
    space = list(product((.55, .70, .80), (-.005, .005, .015, .03), (1., 2., 4.),
                         ("none", "primary", "external", "two"), ("empty", "any", "multiple"), (.80, .90)))
    qualified = []
    for settings in space:
        results = {}
        for scope in DEVELOPMENT:
            after = {key: detect(accepted, saved["scopes"][scope][key], settings)
                     for key, accepted in before[scope].items()}
            new = minimal(data[scope], after, 0.)
            guard = compare(baseline[scope], new)
            if new["fp"] > baseline[scope]["fp"] or not guard["passed"]:
                break
            results[scope] = dict(tp=new["tp"], fp=new["fp"], fn=new["fn"],
                                  new_matches=guard["recovered_matches"])
        if len(results) == len(DEVELOPMENT):
            qualified.append(dict(settings=settings, outcomes=results))
    qualified.sort(key=lambda t: (min(r["new_matches"] for r in t["outcomes"].values()),
                                  sum(r["new_matches"] for r in t["outcomes"].values())), reverse=True)
    write_json(OUTPUT, dict(variants=len(space), retained=len(qualified), leading=qualified[:12],
                            features_sha256=digest(CACHE),
                            shortlisted={k: sum(len(v) for v in frames.values()) for k, frames in saved["scopes"].items()},
                            scope="Development only. One new automatically accepted box at most under the listed conditions."))
    print("Mask verification", len(qualified), "/", len(space), qualified[:1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    (collect if parser.parse_args().action == "collect" else measure)()
