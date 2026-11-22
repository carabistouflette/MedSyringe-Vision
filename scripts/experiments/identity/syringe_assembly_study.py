"""Calibrate assembly-aware additions and measure thin-object proposal coverage."""
import argparse
from itertools import product
from statistics import median

import cv2

from medsyringe.detection.additive import duplicates
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.experiments.crowds.native_candidate_reserve import FOLDER as NATIVE
from scripts.experiments.crowds.native_candidate_verifier import RESULT as PRIOR, compare, simple_metrics
from scripts.experiments.occlusion.occlusion_detector import ART, raster
from scripts.experiments.occlusion.occlusion_proposals import oracle
from scripts.experiments.occlusion.occlusion_relocalization import prepare_reference
from scripts.experiments.occlusion.occlusion_relocalization_reserve import DATA as RELOCAL
from scripts.experiments.identity.syringe_assembly_model import MODEL, descriptor, features, model, pair_is_near, probabilities
from scripts.experiments.identity.thin_syringe_tiles import MODES, WORK, inputs, observations

POLICY = ART / "syringe_assembly_selection.json"
OUTPUT = ART / "syringe_assembly_results.json"
RULES = ("overlap", "box-veto", "pixel-veto", "pixel-identity")
FLOORS = (.15, .20, .28, .36, .45, .55, .65, .75, .85, .95, 1.01)
VETOES = (.55, .70, .85, .95)


def pair_records(row, pools, incumbent, heads):
    rgb = raster(row)
    boxes = {tuple(p["bbox"]) for p in incumbent}
    boxes.update(tuple(p["bbox"]) for pool in pools.values() for p in pool)
    descriptive = {b: descriptor(rgb, b) for b in sorted(boxes)}
    found = {}
    for a in descriptive:
        for b in descriptive:
            if a == b or not pair_is_near(a, b, rgb.size):
                continue
            values = features(a, b, rgb.size, descriptive[a], descriptive[b])
            found[a, b] = {arm: probabilities(head, values) for arm, head in heads.items()}
    return found


def blocked(existing, proposed, rule, evidence):
    a, b = existing["bbox"], proposed["bbox"]
    conventional = duplicates(a, b)
    if rule["method"] == "overlap" or box_iou(a, b) >= .85:
        return conventional
    record = evidence.get((tuple(a), tuple(b)))
    if record is None:
        return conventional
    arm = "boxes" if rule["method"] == "box-veto" else "pixels"
    values = record[arm]
    if values["same"] + values["fragment"] >= rule["veto"]:
        return True
    return conventional and not (rule["method"] == "pixel-identity" and values["distinct"] >= .90)


def fuse(incumbent, proposals, rule, evidence):
    found = list(incumbent)
    for item in sorted(proposals, key=lambda p: -p["score"]):
        if item["score"] < rule["floor"]:
            break
        if not any(blocked(p, item, rule, evidence) for p in found):
            found.append(dict(bbox=item["bbox"], score=.002 + .001*item["score"],
                              source="assembly study", detector_score=item["score"]))
    return found


def run_policies(rows, baseline, data, rules):
    heads = model()
    cv2.setNumThreads(1)
    pair_bank = {r["id"]: pair_records(r, data[r["id"]]["pools"], baseline[r["id"]], heads)
                 for r in rows if r["id"] in data}
    for rule in rules:
        predictions = {r["id"]: fuse(baseline[r["id"]], data[r["id"]]["pools"][rule["view"]],
                                     rule, pair_bank[r["id"]]) for r in rows}
        yield rule, predictions


def choose():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    rows, baseline, prev, _ = prepare_reference()
    view_data = observations("development", rows)
    before = simple_metrics(rows, baseline)
    grid = (dict(view=view, method=method, floor=floor, veto=veto)
            for view, method, floor in product(MODES, RULES, FLOORS)
            for veto in ((None,) if method == "overlap" else VETOES))
    trials = [dict(rule=rule, **compare(rows, before, prediction))
              for rule, prediction in run_policies(rows, baseline, view_data, grid)]
    finalists = {}
    for view, method in product(MODES, RULES):
        viable = [t for t in trials if t["safe"] and t["rule"]["method"] == method and t["rule"]["view"] == view]
        finalists[f"{view}/{method}"] = max(
            viable, key=lambda t: (t["gain"], t["exact_count"], -t["rule"]["floor"], -(t["rule"]["veto"] or 0.)))
    positive = [(name, data) for name, data in finalists.items() if data["gain"] > 0]
    winner = max(positive, default=None, key=lambda v: (v[1]["gain"], v[1]["exact_count"],
                 v[1]["rule"]["view"] == "standard", v[1]["rule"]["method"] == "overlap"))
    write_json(POLICY, dict(code_sha256=digest(__file__), model_sha256=digest(MODEL),
                            development_sha256=digest(WORK / "tiles-development.json"),
                            release_archives=prev, contenders=finalists, trials=trials,
                            selected=winner[0] if winner else None,
                            development_baseline=before,
                            acceptance="Calibrate on only the 60 development images. Preserve matched identities, "
                                       "per-image false positives and per-image count error. Require prospective gain."))
    print("Development selection", winner, flush=True)


def selected_policy():
    seal = read_json(POLICY)
    if (seal["code_sha256"] != digest(__file__) or seal["model_sha256"] != digest(MODEL)
            or digest(WORK / "tiles-development.json") != seal["development_sha256"]
            or any(digest(path) != checksum for path, checksum in seal["release_archives"].items())):
        raise ValueError("A frozen operating point changed.")
    return seal


def summarize(rows, baseline, views, finalists):
    before = simple_metrics(rows, baseline)
    results = {}
    for name, (rule, pred) in zip(finalists, run_policies(
        rows, baseline, views, (r["rule"] for r in finalists.values())), strict=True):
        results[name] = dict(metrics=evaluate(rows, pred, 0.), delta=compare(rows, before, pred))
    ceilings = {r["id"]: {name: oracle(r["objects"], views[r["id"]]["pools"][name])["matched"]
                          for name in MODES} for r in rows}
    totals = {name: sum(value[name] for value in ceilings.values()) for name in MODES}
    cost = {name: median(f["extra_seconds"][name] for f in views.values())
            for name in ("fine-squares", "strips")}
    return dict(baseline=evaluate(rows, baseline, 0.), arms=results, oracle_by_image=ceilings, oracle_total=totals,
                median_added_cpu_seconds=cost)


def replay_fragment():
    """Test the new veto against the exact prior hybrid failure, without retuning on it."""
    rows, _ = inputs("history")
    old = read_json(PRIOR)
    case = read_json(ART / "native_candidate_verifier_prospective.json")["prospective"]
    archived_hybrid = old["evidence"]["prior-relocalization-reserve"]
    mode = "tight-context-rf-identity"
    proposals = {**archived_hybrid["arms"][mode]["supplemental"], **case["arms"][mode]["supplemental"]}
    baseline = {**read_json(RELOCAL / "released.json")["predictions"], **read_json(NATIVE / "released.json")["outputs"]}
    output = {}
    rule = selected_policy()["contenders"]["standard/pixel-veto"]["rule"]
    heads = model()
    for row in rows:
        key = row["id"]
        if key not in proposals:
            continue
        released = baseline[key]
        records = pair_records(row, {"existing additions": proposals[key]}, released, heads)
        accepted = list(released)
        audit = []
        for p in proposals[key]:
            veto = any(blocked(q, p, rule, records) for q in accepted)
            audit.append(dict(bbox=p["bbox"], vetoed=veto,
                              against=[records.get((tuple(q["bbox"]), tuple(p["bbox"]))) for q in accepted]))
            if not veto:
                accepted.append(p)
        target = {key: accepted}
        historical = {key: [*released, *proposals[key]]}
        output[key] = dict(rule=rule, pairs=audit, before=compare([row], simple_metrics([row], {key: released}), historical),
                           after=compare([row], simple_metrics([row], {key: released}), target))
    return output


def score():
    from scripts.experiments.identity.syringe_assembly_reserve import REFERENCE, RELEASE, references

    policy = selected_policy()
    old_rows, _ = inputs("history")
    old_baseline = {**read_json(RELOCAL / "released.json")["predictions"], **read_json(NATIVE / "released.json")["outputs"]}
    previous = summarize(old_rows, old_baseline, observations("history", old_rows), policy["contenders"])
    rows = references()
    release = read_json(RELEASE)
    if release["binding"] != {"reference": digest(REFERENCE), "policy": digest(POLICY)}:
        raise ValueError("Reference or released forecast changed.")
    result = summarize(rows, release["outputs"], observations("prospective", rows), policy["contenders"])
    winner = policy["selected"]
    passes = bool(winner and result["arms"][winner]["delta"]["safe"] and result["arms"][winner]["delta"]["gain"] > 0
                  and previous["arms"][winner]["delta"]["safe"])
    write_json(OUTPUT, dict(policy=digest(POLICY), reference=digest(REFERENCE), released=digest(RELEASE),
                            retrospective=previous, prospective=result, fragment_replay=replay_fragment(),
                            selected=winner, passes_release_gate=passes))
    print("New reserve coverage", result["oracle_total"], "release", passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("choose", "score"))
    {"choose": choose, "score": score}[parser.parse_args().stage]()
