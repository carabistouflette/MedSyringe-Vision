"""Fit an additive proposal verifier and replay it against protected references."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medsyringe.data import digest, leakage, read_image, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate, match
from medsyringe.recall_guard import compare
from scripts import crowd_next_audit as crowd
from scripts.crowded_sam_resolution import CrowdedMasks, MASK_EDGE, mask_iou, masks
from scripts.expanded_detection_study import short
from scripts.sam3_study import LOCK, configure_cpu

DATASET = Path("data/instance-review-dataset/manifest.json")
CACHE = crowd.CACHE.parent / "low-count"
FREEZE = crowd.ART / "low_count_recovery_freeze.json"
REPORT = crowd.ART / "low_count_recovery_results.json"
PROMPT = "syringe"
SEED = 607
FEATURES = ("score", "other_prompt_mask_iou", "other_prompt_score", "barrel_prompt_mask_iou",
            "axial_ratio", "principal_component_fraction", "bbox_mask_fill", "connected_fraction", "area")


def source_rows(role):
    if role == "regression":
        return crowd.inputs()[0]
    return [r for r in read_json(DATASET)["rows"]
            if r["split"] == f"public-{role}" and r["rights_status"] == "individual source verified"]


def image_path(row):
    return Path(row["image"]) if row["image"].startswith("data/") else Path("data") / row["image"]


def signature(role):
    return dict(code=digest(__file__), checkpoint=digest(LOCK), resolution=756,
                source=digest(crowd.REFERENCE if role == "regression" else DATASET),
                rasters={r["id"]: r["sha256"] for r in source_rows(role)})


def collect(role):
    if role != "fit":
        load_policy()
    configure_cpu()
    path = CACHE / f"{role}.json"
    binding = signature(role)
    state = read_json(path) if path.exists() else dict(binding=binding, frames={})
    if state["binding"] != binding:
        raise ValueError("A cached proposal input changed.")
    pending = [r for r in source_rows(role) if r["id"] not in state["frames"]]
    if not pending:
        return
    model = CrowdedMasks(756, "xpu")
    for row in pending:
        asset = image_path(row)
        if digest(asset) != row["sha256"]:
            raise ValueError(f"Photo changed: {row['id']}")
        state["frames"][row["id"]] = dict(model.candidates(read_image(asset)))
        write_json(path, state)
        print(role, row["id"], {q: len(p) for q, p in state["frames"][row["id"]].items()}, flush=True)


def cached(role):
    path = CACHE / f"{role}.json"
    state = read_json(path)
    if state["binding"] != signature(role) or set(state["frames"]) != set(state["binding"]["rasters"]):
        raise ValueError(f"Missing or stale proposals on {role}.")
    return state["frames"]


def disjoint(raw, minimum=.18):
    primary = masks(raw[PROMPT])
    others = {name: masks(raw[name]) for name in ("transparent plastic syringe", "syringe barrel")}
    chosen = []
    for proposal in primary:
        if proposal["score"] < minimum:
            continue
        if any(mask_iou(proposal["pixels"], old["pixels"]) >= .45 for old in chosen):
            continue
        own = proposal["pixels"]
        best = {name: max(((mask_iou(own, other["pixels"]), other["score"]) for other in parts),
                          default=(0., 0.)) for name, parts in others.items()}
        silhouette = np.unpackbits(own).reshape(MASK_EDGE, MASK_EDGE).astype(np.uint8)
        y, x = np.nonzero(silhouette)
        if len(x) < 4:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(np.stack((x, y))))
        slender = np.sqrt((eigenvalues[-1] + 1.) / (eigenvalues[0] + 1.))
        count, _, components, _ = cv2.connectedComponentsWithStats(silhouette, connectivity=8)
        largest = max(components[1:count, cv2.CC_STAT_AREA], default=0)
        line = [proposal["score"], *best["transparent plastic syringe"], best["syringe barrel"][0],
                float(np.log(slender)), float(eigenvalues[-1] / max(1., eigenvalues.sum())),
                len(x) / ((np.ptp(x) + 1) * (np.ptp(y) + 1)), largest / len(x), len(x) / MASK_EDGE**2]
        chosen.append({**proposal, "features": [float(value) for value in line]})
    return chosen


def labelled(rows, bank):
    features, labels, groups = [], [], []
    for row in rows:
        objects = {row["id"]: disjoint(bank[row["id"]])}
        assignments, _ = match([row], objects, .5)
        for _, proposal, target, _ in assignments:
            features.append(proposal["features"])
            labels.append(target is not None)
            groups.append(row["group"])
    return np.asarray(features), np.asarray(labels), np.asarray(groups)


def candidate_policy(x, y, group):
    if min(Counter(y).values()) < 4 or len(set(group[y])) < 4 or len(set(group[~y])) < 4:
        raise ValueError("Need positive and negative proposals from at least four acquisition groups each.")
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(x, y, group))
    choices = []
    for columns in (list(range(4)), list(range(len(FEATURES)))):
        for strength in (.1, 1.):
            raw = np.zeros(len(y))
            for train, valid in folds:
                if not len(np.unique(y[train])) == 2:
                    raise ValueError("A grouped fold has no proposal class coverage.")
                estimator = make_pipeline(StandardScaler(), LogisticRegression(C=strength, max_iter=500))
                raw[valid] = estimator.fit(x[train][:, columns], y[train]).predict_proba(x[valid][:, columns])[:, 1]
            floor = float(np.nextafter(raw[~y].max(), np.inf))
            choices.append(dict(columns=columns, c=strength, floor=floor, tp=int(np.sum((raw >= floor) & y)),
                                fp=int(np.sum((raw >= floor) & ~y)), samples=len(y)))
    selected = max(choices, key=lambda q: (q["tp"], -len(q["columns"]), -q["c"]))
    fitted = make_pipeline(StandardScaler(), LogisticRegression(C=selected["c"], max_iter=500))
    fitted.fit(x[:, selected["columns"]], y)
    scale, head = fitted.steps[0][1], fitted.steps[1][1]
    return dict(**selected, mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                weight=head.coef_[0].tolist(), bias=float(head.intercept_[0])), choices


def fit():
    if FREEZE.exists():
        raise FileExistsError(FREEZE)
    three = [r for role in ("fit", "development", "reserve") for r in source_rows(role)]
    if leakage(three):
        raise ValueError("The fitting roles share an acquisition or duplicate.")
    lookup = read_json(DATASET)["hard_holdout_sha256"]
    if lookup != digest(crowd.REFERENCE):
        raise ValueError("The separate crowd cohort has changed.")
    x, y, group = labelled(source_rows("fit"), cached("fit"))
    head, controls = candidate_policy(x, y, group)
    write_json(FREEZE, dict(code=digest(__file__), dataset=digest(DATASET), masks=digest(CACHE / "fit.json"),
                           checkpoint=digest(LOCK), reference=lookup, features=FEATURES, seed=SEED,
                           fit_photos=len(source_rows("fit")), positive_proposals=int(y.sum()),
                           negative_proposals=int((~y).sum()), groups=len(set(group)),
                           selected=head, grouped_controls=controls,
                           evaluation_gate="On fresh crowded acquisitions, increase exact count and recall, "
                                           "lose zero baseline matches and add zero false positives. "
                                           "Preserve existing regression controls."))
    print("Fixed proposal verifier", head["tp"], "/", int(y.sum()), "grouped out-of-fold true positives", flush=True)


def load_policy():
    state = read_json(FREEZE)
    if (state["code"] != digest(__file__) or state["dataset"] != digest(DATASET)
            or state["masks"] != digest(CACHE / "fit.json") or state["checkpoint"] != digest(LOCK)):
        raise ValueError("Frozen verifier evidence changed.")
    return state["selected"]


def verification_score(head, proposal):
    x = np.asarray(proposal["features"])[head["columns"]]
    z = ((x - head["mean"]) / head["scale"]) @ head["weight"] + head["bias"]
    return float(1. / (1. + np.exp(-z)))


def accept_additions(baseline, proposals, criterion):
    if len(baseline) >= 2:
        return list(baseline)
    additions = []
    for item in proposals:
        if not criterion(item):
            continue
        if any(box_iou(p["bbox"], item["bbox"]) >= .67 for p in baseline):
            continue
        score = min((p["score"] for p in baseline), default=item["score"])
        additions.append(dict(bbox=item["bbox"], score=min(item["score"], score), source="sam recovery"))
    return [*baseline, *additions]


def brief(rows, predictions):
    value = evaluate(rows, predictions, 0.)
    return {**short(value), "false_positives": value["false_positives"], "false_negatives": value["false_negatives"]}


def measure():
    head = load_policy()
    arm = dict(
        unverified=lambda p: p["score"] >= .50,
        corroborated=lambda p: p["score"] >= .50 and p["features"][1] >= .55 and p["features"][2] >= .50,
        fitted=lambda p: verification_score(head, p) >= head["floor"],
    )
    scopes = {}
    for role in ("development", "reserve", "regression"):
        bank = cached(role)
        rows = source_rows(role)
        proposals = {key: disjoint(raw) for key, raw in bank.items()}
        if role == "regression":
            crowd.require_freeze()
            released = read_json(crowd.CACHE / "released.json")
            if released["freeze"] != digest(crowd.FREEZE):
                raise ValueError("Protected released results changed.")
            before = {key: item["objects"] for key, item in released["frames"].items()}
        else:
            # These tiny samples check proposal precision without pretending to be engine comparisons.
            before = {row["id"]: [] for row in rows}
        observed = {"baseline": before, **{k: {i: accept_additions(p, proposals[i], f) for i, p in before.items()}
                                         for k, f in arm.items()}}
        cohort = {"all": rows}
        if role == "regression":
            cohort["crowded with 0 or 1 starting prediction"] = [
                r for r in rows if len(r["objects"]) >= 2 and len(before[r["id"]]) < 2]
            cohort["empty"] = [r for r in rows if not r["objects"]]
        scopes[role] = {}
        for name, cases in cohort.items():
            ids = {r["id"] for r in cases}
            details = {k: brief(cases, {i: p for i, p in v.items() if i in ids}) for k, v in observed.items()}
            base = details["baseline"]
            scopes[role][name] = dict(outcomes=details,
                                     retention={k: compare(base, result) for k, result in details.items()})
        print(role, {k: {f: v[f] for f in ("tp", "fp", "fn", "exact_count")}
                     for k, v in scopes[role]["all"]["outcomes"].items()}, flush=True)
    write_json(REPORT, dict(freeze=digest(FREEZE), cohorts=scopes,
                            scope="Development and small independent reserve check individual proposals. "
                                  "The crowd cohort is an already observed regression benchmark. "
                                  "Release requires new prospective crowded acquisitions."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("collect", "fit", "measure"))
    parser.add_argument("--role", choices=("fit", "development", "reserve", "regression"), default="fit")
    args = parser.parse_args()
    if args.job == "collect":
        collect(args.role)
    elif args.job == "fit":
        fit()
    else:
        measure()
