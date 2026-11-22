"""Compare visual part geometry and frozen DINOv2 features on the same SAM proposals."""
import argparse
from base64 import b64decode, b64encode
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import match
from medsyringe.runtime.grounding_backend import analyze
from medsyringe.runtime.hardware import configure_device
from medsyringe.measurement.part_geometry import PARTS, PROMPTS, geometry
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.crowds.crowded_sam_resolution import MASK_EDGE, mask_iou
from scripts.experiments.crowds.low_count_prospective import WORK as TRAY_CACHE
from scripts.experiments.crowds.low_count_recovery import accept_additions, brief, crowd
from scripts.experiments.parts.part_verifier_data import OUT as REFERENCES, WORK
from scripts.experiments.transfer.rfdetr_verifier import DINOV2_INFO, DinoEncoder, expanded
from scripts.experiments.expanded.sam3_study import LOCK, Segmenter, configure_cpu

FREEZE = crowd.ART / "part_verifier_freeze.json"
RESULT = crowd.ART / "part_verifier_results.json"
CODE = (Path(__file__), Path("medsyringe/measurement/part_geometry.py"))
BODY_PROMPTS = ("syringe", "prefilled syringe")
SEED = 607


def signature():
    return dict(reference=digest(REFERENCES), code={str(p): digest(p) for p in CODE},
                sam=digest(LOCK), appearance=digest(DINOV2_INFO))


def source(role):
    rows = read_json(REFERENCES)["roles"][role]
    if role == "fit":
        return [r for r in rows if not r["objects"] or len(r["objects"]) > 1
                or any(o.get("parts") for o in r["objects"]) or r["id"].isdigit()]
    return rows


def infer(model, image):
    import torch

    with torch.inference_mode():
        pixels = model.processor(images=image, return_tensors="pt",
                                 size={"height": model.edge, "width": model.edge}).pixel_values.to(model.device)
        features = model.model.get_vision_features(pixel_values=pixels)
        results = {}
        for label, prompt in [*((label, label) for label in BODY_PROMPTS), *PROMPTS.items()]:
            tokens = model.processor(text=prompt, return_tensors="pt").to(model.device)
            transposed = model.processor.post_process_instance_segmentation(
                model.model(vision_embeds=features, **tokens), threshold=.18,
                target_sizes=[(MASK_EDGE, MASK_EDGE)])[0]
            candidates = []
            boxes = transposed["boxes"].cpu().tolist()
            scores = transposed["scores"].cpu().tolist()
            silhouettes = transposed["masks"].cpu().numpy()
            for box, score, silhouette in zip(boxes, scores, silhouettes, strict=True):
                rect = [max(0., min(1., k / MASK_EDGE)) for k in box]
                if valid_box(rect) and silhouette.any():
                    candidate = dict(bbox=rect, score=score)
                    if label in BODY_PROMPTS:
                        candidate["mask"] = b64encode(np.packbits(silhouette).tobytes()).decode("ascii")
                    candidates.append(candidate)
            results[label] = sorted(candidates, key=lambda item: -item["score"])[:25]
    return results


def describe(image, observations, encoder):
    selected = []
    collection = sorted((p for key in BODY_PROMPTS for p in observations[key]), key=lambda p: -p["score"])
    for candidate in collection:
        if candidate["score"] < .40:
            continue
        raw = np.frombuffer(b64decode(candidate["mask"]), dtype=np.uint8)
        if any(mask_iou(raw, seen) >= .45 for _, seen in selected):
            continue
        selected.append((candidate, raw))
    candidates = []
    for p, raw in selected[:24]:
        y, x = np.where(np.unpackbits(raw).reshape(MASK_EDGE, MASK_EDGE))
        points = np.column_stack((x * image.width/MASK_EDGE, y * image.height/MASK_EDGE))
        parts, matched = geometry(p["bbox"], observations, points, image.width, image.height)
        candidates.append(dict(bbox=p["bbox"], score=p["score"], features=[p["score"], *parts], parts=matched))
    embeddings = encoder.encode([crop(image, expanded(p["bbox"], .07)) for p in candidates])
    for p, embedding in zip(candidates, embeddings, strict=True):
        p["embedding"] = embedding.tolist()
    return candidates


def collect(role):
    if role != "fit":
        load()
    configure_cpu()
    path = WORK / f"part-features-{role}.json"
    sig = signature()
    state = read_json(path) if path.exists() else dict(signature=sig, frames={})
    if state["signature"] != sig:
        raise ValueError("Feature evidence changed.")
    pending = [row for row in source(role) if row["id"] not in state["frames"]]
    if not pending:
        return
    model, encoder = Segmenter(756, "xpu"), DinoEncoder()
    for row in pending:
        if digest(row["image"]) != row["sha256"]:
            raise ValueError("Reference bytes changed.")
        image = read_image(row["image"])
        observation = infer(model, image)
        state["frames"][row["id"]] = dict(proposals=describe(image, observation, encoder),
                                          prompt_parts={key: observation[key] for key in PARTS})
        write_json(path, state)
        print(role, row["id"], len(state["frames"][row["id"]]["proposals"]), flush=True)


def records(role):
    item = read_json(WORK / f"part-features-{role}.json")
    if item["signature"] != signature() or set(item["frames"]) != {r["id"] for r in source(role)}:
        raise ValueError("The complete frozen feature cache is required.")
    return item["frames"]


def features(proposal, arm):
    if arm == "parts":
        return proposal["features"]
    visual = (4. * np.asarray(proposal["embedding"])).tolist()
    if arm == "appearance":
        return visual
    return [*proposal["features"], *visual]


def fit():
    if FREEZE.exists():
        raise FileExistsError(FREEZE)
    examples = []
    bank = records("fit")
    for row in source("fit"):
        sid = row["id"]
        assignments, _ = match([row], {sid: bank[sid]["proposals"]}, .5)
        examples.extend((proposal, target is not None, row["group"]) for _, proposal, target, _ in assignments)
    y = np.array([value for _, value, _ in examples])
    groups = np.array([group for _, _, group in examples])
    if min(len(set(groups[y])), len(set(groups[~y]))) < 4:
        raise ValueError("Each class needs four distinct fitting groups.")
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(y, y, groups))
    heads = {}
    for arm in ("parts", "appearance", "combined"):
        x = np.asarray([features(p, arm) for p, _, _ in examples])
        oof = np.zeros(len(y))
        for train, valid in folds:
            model = LogisticRegression(C=.5, class_weight="balanced", max_iter=600)
            oof[valid] = model.fit(x[train], y[train]).predict_proba(x[valid])[:, 1]
        floor = float(np.nextafter(oof[~y].max(), np.inf))
        model = LogisticRegression(C=.5, class_weight="balanced", max_iter=600).fit(x, y)
        heads[arm] = dict(weight=model.coef_[0].tolist(), bias=float(model.intercept_[0]), floor=floor,
                          fit_oof_true=int(np.count_nonzero((oof >= floor) & y)),
                          fit_oof_false=int(np.count_nonzero((oof >= floor) & ~y)))
    write_json(FREEZE, dict(signature=signature(), heads=heads, fitting_photos=len(bank),
                           positive=int(y.sum()), negative=int((~y).sum()), groups=len(set(groups)),
                           fit_feature_cache=digest(WORK / "part-features-fit.json"),
                           criteria="Fresh exact count and recall improve with no added false positives, "
                                    "no baseline matches lost, and prior observations retained."))
    print({k: v["fit_oof_true"] for k, v in heads.items()}, flush=True)


def load():
    policy = read_json(FREEZE)
    if policy["signature"] != signature() or policy["fit_feature_cache"] != digest(WORK / "part-features-fit.json"):
        raise ValueError("The sealed model or its input changed.")
    return policy["heads"]


def accept(p, head, arm):
    z = np.asarray(features(p, arm)) @ head["weight"] + head["bias"]
    return float(1./(1.+np.exp(-z))) >= head["floor"]


def offer(baseline, proposals, criterion):
    if len(baseline) == 1 and proposals:
        peer = max(proposals, key=lambda p: box_iou(baseline[0]["bbox"], p["bbox"]))
        if box_iou(baseline[0]["bbox"], peer["bbox"]) >= .40:
            proposals = [p for p in proposals if p is not peer]
    return accept_additions(baseline, proposals, criterion)


def released():
    load()
    configure_cpu()
    root = Path.cwd()
    engine = MedSyringeEngine(root)
    configure_device(engine, root, "xpu")
    objects = {}
    try:
        for row in [*source("fresh"), *source("transfer")]:
            result = analyze(engine, read_image(row["image"]), False)
            checks = result["detection_checks"]
            if checks["status"] != "available" or checks["primary_backend"] != "xpu":
                raise ValueError("The released Intel inference path was unavailable.")
            objects[row["id"]] = result["objects"]
            print(row["id"], len(result["objects"]), flush=True)
    finally:
        engine.close()
    write_json(WORK / "released.json", dict(signature=signature(), frames=objects))


def landmark_audit(rows, bank):
    values = {key: dict(reviewed=0, proposal=0, within_proposal=0, center_within_10pct=0) for key in PARTS}
    for row in rows:
        for obj in row["objects"]:
            for key, point in obj.get("parts", {}).items():
                if point is None:
                    continue
                ledger = values[key]
                ledger["reviewed"] += 1
                valid = [p for p in bank[row["id"]]["prompt_parts"][key]
                         if p["score"] >= .30 and box_iou(p["bbox"], obj["bbox"]) >= .012]
                if not valid:
                    continue
                ledger["proposal"] += 1
                l, t, r, b = valid[0]["bbox"]
                ledger["within_proposal"] += l <= point[0] <= r and t <= point[1] <= b
                body = obj["bbox"]
                diag = np.linalg.norm(np.array(body[2:])-body[:2])
                ledger["center_within_10pct"] += np.linalg.norm(np.array([(l+r)/2, (t+b)/2])-point) <= .10*diag
    return values


def measure():
    heads = load()
    baseline = {i: v["objects"] for i, v in read_json(crowd.CACHE / "released.json")["frames"].items()}
    baseline.update(read_json(TRAY_CACHE / "released.json")["outputs"])
    fresh_release = read_json(WORK / "released.json")
    if fresh_release["signature"] != signature():
        raise ValueError("Fresh results were produced on another reference.")
    outcomes = {}
    for role in ("fresh", "transfer", "regression"):
        rows, bank = source(role), records(role)
        previous = baseline if role == "regression" else fresh_release["frames"]
        known = {r["id"]: previous[r["id"]] for r in rows}
        candidates = {key: v["proposals"] for key, v in bank.items()}
        arms = {"unverified": lambda p: p["score"] >= .50,
                "geometry_rule": lambda p: p["features"][6] >= .30 or
                (p["features"][5] >= .38 and p["features"][1] >= .38),
                **{arm: (lambda p, key=arm: accept(p, heads[key], key)) for arm in heads}}
        before = brief(rows, known)
        staged = {arm: {key: offer(existing, candidates[key], rule) for key, existing in known.items()}
                  for arm, rule in arms.items()}
        trials = {arm: brief(rows, branch) for arm, branch in staged.items()}
        outcomes[role] = dict(baseline=before, trials=trials,
                              retention={k: compare(before, t) for k, t in trials.items()},
                              counts={r["id"]: {"expected": len(r["objects"]), "released": len(known[r["id"]]),
                                               **{arm: len(branch[r["id"]]) for arm, branch in staged.items()}}
                                      for r in rows})
        print(role, {k: (v["tp"], v["fp"], v["exact_count"]) for k, v in {"baseline": before, **trials}.items()}, flush=True)
    write_json(RESULT, dict(freeze=digest(FREEZE), outcomes=outcomes,
                            part_audit={role: landmark_audit(source(role), records(role)) for role in ("fit", "transfer", "fresh")},
                            scope="Fresh screened public acquisitions and separate new photos by previously seen contributors. "
                                  "The prior tray, its two negatives, and the prior crowded photos are regression."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("collect", "fit", "released", "measure"))
    parser.add_argument("--role", choices=("fit", "fresh", "transfer", "regression"), default="fit")
    args = parser.parse_args()
    if args.job == "collect":
        collect(args.role)
    else:
        {"fit": fit, "released": released, "measure": measure}[args.job]()
