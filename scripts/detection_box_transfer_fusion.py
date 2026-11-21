"""Measure whether the trained branches add useful, independently corroborated boxes."""
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
from scripts.detection_real_box_evaluate import OUTPUT as TRAINING, path as raw_path
from scripts.detection_real_box_transfer import ARMS
from scripts.precision_study import source
from scripts.rfdetr_verifier import expanded
from scripts.sam3_study import configure_cpu

OUTPUT = ART / "reviewed_box_transfer_fusion.json"
CACHE = DATA / "proposals/reviewed-box-transfer-fusion.json"
VIEWS = ("whole", "wide")


def trial_inputs():
    record = read_json(TRAINING)["matched_arms"]
    raw = {}
    for arm in ARMS:
        asset = raw_path(arm)
        if digest(asset) != record[arm]["collection_sha256"]:
            raise ValueError("The proposals differ from the completed training comparison.")
        raw[arm] = read_json(asset)["frames"]
    return raw


def collect():
    configure_cpu()
    raw = trial_inputs()
    proof = dict(selection=digest(TRAINING), code=digest(Path(__file__)))
    saved = read_json(CACHE) if CACHE.exists() else dict(key=proof, scopes={})
    if saved["key"] != proof:
        raise ValueError("The combined proposal inputs changed.")
    encoder = None
    for scope in DEVELOPMENT:
        rows, accepted = inputs(scope), references(scope)
        baseline = loaded(scope, "baseline")
        features = saved["scopes"].setdefault(scope, {})
        for row in rows:
            name = row["id"]
            if name in features:
                continue
            records = {}
            all_boxes = {}
            for arm in ARMS:
                for view in VIEWS:
                    viable = [p for p in raw[arm][name][view] if p["score"] >= .18]
                    viable = [p for p in viable if not duplicates(p, accepted[name])]
                    records[f"{arm}/{view}"] = viable[:8]
                    for p in viable[:8]:
                        all_boxes.setdefault(tuple(p["bbox"]), p)
            if all_boxes:
                if encoder is None:
                    encoder = TextMarginEncoder()
                image = source(row)
                crops = [crop(image, expanded(p["bbox"], .08)) for p in all_boxes.values()]
                margins = []
                for start in range(0, len(crops), 8):
                    vectors = encoder.encode(crops[start:start+8])["prompt"]
                    margins.extend((vectors[:, 0]-vectors[:, 1]).tolist())
                for box, margin in zip(all_boxes, margins, strict=True):
                    observations = baseline[name]["ranked"]
                    all_boxes[box] = dict(
                        margin=margin, before=max((box_iou(box, p["bbox"]) for p in observations
                                                  if p["score"] >= .23), default=0.),
                        other_models={arm: max((p["score"] for p in raw[arm][name]["wide"]
                                                if box_iou(box, p["bbox"]) >= .42), default=0.)
                                      for arm in ARMS})
            features[name] = {k: [{**p, **all_boxes[tuple(p["bbox"])]} for p in entries]
                              for k, entries in records.items()}
            write_json(CACHE, saved)
            print("Supervised corroboration", scope, len(features), "/", len(rows), flush=True)


def duplicates(item, accepted):
    box = item["bbox"]
    return any(box_iou(box, obj["bbox"]) >= .42 or overlap_fraction(box, obj["bbox"]) >= .80 for obj in accepted)


def detect(old, new, settings):
    arm, confidence, semantic, witness, region = settings
    if old and region == "empty":
        return old
    candidate = []
    other, = [name for name in ARMS if name != arm.split("/")[0]]
    for proposal in new[arm]:
        if proposal["score"] < confidence or proposal["margin"] < semantic:
            continue
        if witness == "prior" and proposal["before"] < .38:
            continue
        if witness == "supervised" and proposal["other_models"][other] < .23:
            continue
        candidate.append(dict(bbox=proposal["bbox"], score=.10))
    return [*old, *suppress(candidate, .42)[:1]]


def measure():
    if OUTPUT.exists():
        raise ValueError("The supervised fusion development result already exists.")
    saved = read_json(CACHE)
    if saved["key"] != dict(selection=digest(TRAINING), code=digest(Path(__file__))):
        raise ValueError("Mismatched collection inputs.")
    rows = {scope: inputs(scope) for scope in DEVELOPMENT}
    prior = {scope: references(scope) for scope in DEVELOPMENT}
    old = {scope: minimal(rows[scope], prior[scope], 0.) for scope in DEVELOPMENT}
    if any(set(saved["scopes"][k]) != set(prior[k]) for k in DEVELOPMENT):
        raise ValueError("Incomplete proposal verification.")
    series = list(product((f"{a}/{v}" for a, v in product(ARMS, VIEWS)), (.18, .24, .28, .32),
                          (-.005, .005, .018), ("prior", "supervised", "none"), ("empty", "any")))
    qualified = []
    for setting in series:
        tested = {}
        for scope in DEVELOPMENT:
            altered = {key: detect(known, saved["scopes"][scope][key], setting)
                       for key, known in prior[scope].items()}
            new = minimal(rows[scope], altered, 0.)
            guard = compare(old[scope], new)
            if new["fp"] > old[scope]["fp"] or not guard["passed"]:
                break
            tested[scope] = dict(tp=new["tp"], fp=new["fp"], fn=new["fn"],
                                 gained=guard["recovered_matches"])
        if len(tested) == len(DEVELOPMENT):
            qualified.append(dict(setting=setting, scopes=tested))
    qualified.sort(key=lambda k: (min(v["gained"] for v in k["scopes"].values()),
                                  sum(v["gained"] for v in k["scopes"].values())), reverse=True)
    write_json(OUTPUT, dict(variants=len(series), safe_on_development=len(qualified), leading=qualified[:8],
                            records_sha256=digest(CACHE), scores="One additional box at most. Compare two trained "
                            "auxiliaries and two views, SigLIP crop agreement and separate detector witnesses."))
    print("Supervised fusion", len(qualified), "/", len(series), qualified[:1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("collect", "measure"))
    (collect if p.parse_args().action == "collect" else measure)()
