"""Add visually confirmed training crops to the candidate-only appearance head."""
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.evaluate import match
from medsyringe.detection.recall_guard import compare
from scripts.experiments.parts.part_verifier_study import (
    FREEZE, REFERENCES, WORK, DinoEncoder, brief, crowd, expanded, load, offer, records, source,
)

POLICY = crowd.ART / "part_reference_crop_freeze.json"
REPORT = crowd.ART / "part_reference_crop_results.json"
PASTEUR = [[.018, .055, .983, .388], [.12, .675, .891, .933]]


def labelled_candidates():
    training = records("fit")
    examples = []
    for row in source("fit"):
        key = row["id"]
        matches, _ = match([row], {key: training[key]["proposals"]}, .5)
        examples.extend((p["embedding"], target is not None, row["group"]) for _, p, target, _ in matches)
    return examples


def verified_crops():
    photos, classes, groups = [], [], []
    for row in read_json(REFERENCES)["roles"]["fit"]:
        negatives = PASTEUR if row["id"] == "pasteur-fit" else [o["bbox"] for o in row.get("lookalikes", [])]
        boxes = [*[(obj["bbox"], True) for obj in row["objects"]], *((box, False) for box in negatives)]
        if boxes:
            picture = read_image(row["image"])
            for box, expected in boxes:
                photos.append(crop(picture, expanded(box, .07)))
                classes.append(expected)
                groups.append(row["group"])
    return 4. * DinoEncoder().encode(photos), np.asarray(classes), np.asarray(groups)


def train():
    load()
    if POLICY.exists():
        raise FileExistsError(POLICY)
    observations = labelled_candidates()
    x = 4. * np.asarray([p for p, _, _ in observations])
    y = np.asarray([target for _, target, _ in observations])
    groups = np.asarray([group for _, _, group in observations])
    gx, gy, gg = verified_crops()

    def estimator(mask, gold):
        xtrain, ytrain = np.concatenate([x[mask], gx[gold]]), np.r_[y[mask], gy[gold]]
        training_groups = np.r_[groups[mask], gg[gold]]
        counts = Counter(training_groups)
        weights = np.array([1./counts[g] for g in training_groups])
        weights *= len(weights)/weights.sum()
        head = LogisticRegression(C=.5, class_weight="balanced", max_iter=600)
        return head.fit(xtrain, ytrain, sample_weight=weights)

    oof = np.zeros(len(y))
    for train, validation in StratifiedGroupKFold(4, shuffle=True, random_state=607).split(x, y, groups):
        unseen = set(groups[validation])
        kept = np.array([group not in unseen for group in gg])
        oof[validation] = estimator(train, kept).predict_proba(x[validation])[:, 1]
    floor = float(np.nextafter(oof[~y].max(), np.inf))
    head = estimator(np.arange(len(y)), np.ones(len(gy), bool))
    write_json(POLICY, dict(freeze=digest(FREEZE), code=digest(__file__), added_crop_positives=int(gy.sum()),
                            added_crop_negatives=int((~gy).sum()), floor=floor,
                            oof_true=int(np.sum((oof >= floor) & y)), oof_false=int(np.sum((oof >= floor) & ~y)),
                            weights=head.coef_[0].tolist(), bias=float(head.intercept_[0])))
    print("Ground-truth crops", int(gy.sum()), int((~gy).sum()), "out-of-fold true", int(np.sum((oof >= floor) & y)), flush=True)


def accepts(policy, p):
    score = (4.*np.asarray(p["embedding"])) @ policy["weights"] + policy["bias"]
    return float(1./(1.+np.exp(-score))) >= policy["floor"]


def measure():
    state = read_json(POLICY)
    if state["freeze"] != digest(FREEZE) or state["code"] != digest(__file__):
        raise ValueError("The appearance ablation was modified after fitting.")
    values = read_json(WORK / "released.json")["frames"]
    output = {}
    for role in ("fresh", "transfer", "regression"):
        rows, bank = source(role), records(role)
        if role == "regression":
            values = {i: d["objects"] for i, d in read_json(crowd.CACHE / "released.json")["frames"].items()}
            values.update(read_json("data/low-count-prospective/released.json")["outputs"])
        before = {row["id"]: values[row["id"]] for row in rows}
        selected = {key: offer(base, bank[key]["proposals"], lambda p: accepts(state, p)) for key, base in before.items()}
        baseline, verified = brief(rows, before), brief(rows, selected)
        output[role] = dict(baseline=baseline, verified=verified, retention=compare(baseline, verified))
        print(role, {k: (baseline[k], verified[k]) for k in ("tp", "fp", "exact_count")}, flush=True)
    write_json(REPORT, dict(policy=digest(POLICY), outcomes=output,
                            scope="Exploratory reference-crop ablation on previously consulted proposal banks. "
                                  "Only original training-role annotations enter fitting."))


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument("job", choices=("fit", "measure"))
    (train if parse.parse_args().job == "fit" else measure)()
