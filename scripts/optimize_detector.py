"""Train proposal heads, select combinations and score locked holdouts."""
import argparse
from itertools import combinations

from joblib import dump, load
import numpy as np
from PIL import ImageEnhance

from medsyringe.data import digest, download, read_image, read_json, train_rows, write_json
from medsyringe.detectors import model_hashes, suppress
from medsyringe.evaluate import best_threshold, evaluate, match
from medsyringe.proposal_learning import VARIANTS, adjust_scores, describe_proposals, fit_variant, labels_for
from medsyringe.refinement import view_image
from scripts.refine_detector import ART, HOLDOUT, NEW, OLD, ROOT, cache, predict

FINAL = ART / "detector_release.json"
WEIGHTS = ART / "proposal_heads.joblib"
SCORES = ("precision", "recall", "f1", "ap50", "ap50_95", "exact_count", "tp", "fp", "fn")


def validate_local_inputs(frozen):
    if model_hashes(ROOT, frozen["backbone"]) != frozen["bundle_sha256"]:
        raise ValueError("Backbone weights changed.")
    if digest(ART / "annotations.json") != frozen["draft_annotations_sha256"]:
        raise ValueError("Training annotations differ from the proposal-cache reference.")


def prepare_training(rows, frozen):
    inputs = []
    for variant in ("original", "augmented"):
        stored = read_json(ROOT / "data/cache" / f"dino-train-{variant}.json")
        config = stored["config"]
        if (config["weights"] != frozen["bundle_sha256"] or config["variant"] != variant or
            dict(config["rows"]) != {r["id"]: r["derived_sha256"] for r in rows}):
            raise ValueError("Training proposal cache does not match the training partition.")
        inputs.append(stored["predictions"])
    frames = []
    for r in rows:
        path = ROOT / "data" / r["image"]
        if digest(path) != r["derived_sha256"]:
            raise ValueError(f"Prepared image does not match the manifest: {r['id']}")
        image = read_image(path)
        truth = [ob["bbox"] for ob in r["objects"]]
        augmented = ImageEnhance.Brightness(view_image(image, "mirror")).enhance(
            0.78 if int(r["id"][-1]) % 2 else 1.20)
        for j, im in enumerate((image, augmented)):
            candidates = inputs[j][r["id"]]["proposals"]
            geometry, texture = describe_proposals(im, candidates)
            boxes = [[1-c, b, 1-a, d] for a,b,c,d in truth] if j else truth
            frames.append(dict(burst=r["objects"][0]["burst"], augmented=bool(j),
                               geometry=geometry, texture=texture, labels=labels_for(candidates, boxes)))
    return frames


def counters(metrics):
    return {k: metrics[k] for k in SCORES}


def describe(rows, predictions):
    return {r["id"]: describe_proposals(read_image(ROOT / "data" / r["image"]), predictions[r["id"]])
            for r in rows}


def rescore(predictions, features, models, weight, overlap):
    if not models:
        return predictions
    return {i: suppress(adjust_scores(p, features[i], models, weight), overlap)
            for i, p in predictions.items()}


def select(frozen):
    if FINAL.exists():
        raise ValueError("Release selection is already frozen.")
    stage = read_json(NEW)
    if digest(HOLDOUT) != stage["holdout_manifest_sha256"]:
        raise ValueError("Holdout references differ from the preregistered snapshot.")
    rows = read_json(ART / "validation_manifest.json")["rows"]
    frames = prepare_training(train_rows(ROOT), frozen)
    fitted = {}
    folder = ROOT / "data/trained_heads"
    folder.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        model = fit_variant(name, frames)
        fitted[name] = model
        dump(model, folder / f"{name}.joblib", compress=3)
        print("trained", name, model["metrics"]["training_proposals"], flush=True)
    searched = read_json(ART / "v2_search.json")["trials"]
    entries = cache(frozen, "validation", rows, sorted({v for c in searched for v in c["views"]}))
    results = [{**c, "heads": [], "weight": 0.0} for c in searched]
    top_configs = searched[:4] + [next(c for c in searched if len(c["views"]) == 1)]
    for config in top_configs:
        views, strategy, overlap = (config[k] for k in ("views", "strategy", "overlap"))
        predictions = predict(entries, views, strategy, overlap)
        descriptors = describe(rows, predictions)
        pool = []
        for heads in [(name,) for name in fitted]:
            for weight in (0.25, 0.5):
                rescored = rescore(predictions, descriptors, [fitted[n] for n in heads], weight, overlap)
                cutoff = best_threshold(rows, rescored)
                item = dict(views=views, strategy=strategy, overlap=overlap, heads=list(heads),
                            weight=weight, cutoff=cutoff, **counters(evaluate(rows, rescored, cutoff)))
                pool.append(item)
        results.extend(pool)
        ranks = sorted(pool, key=rank, reverse=True)
        first = ranks[0]["heads"][0]
        second = next(x["heads"][0] for x in ranks if
                      fitted[x["heads"][0]]["metrics"]["family"] != fitted[first]["metrics"]["family"])
        rescored = rescore(predictions, descriptors, [fitted[first], fitted[second]], .375, overlap)
        cutoff = best_threshold(rows, rescored)
        results.append(dict(views=views, strategy=strategy, overlap=overlap, heads=[first, second],
                            weight=.375, cutoff=cutoff, **counters(evaluate(rows, rescored, cutoff))))
    results.sort(key=rank, reverse=True)
    selected = results[0]
    asset_sha = None
    if selected["heads"]:
        dump([fitted[n] for n in selected["heads"]], WEIGHTS, compress=3)
        asset_sha = digest(WEIGHTS)
    report = {**stage, **selected, "id": "optimized-detector-v3",
              "proposal_learning_sha256": digest("medsyringe/proposal_learning.py"),
              "proposal_heads_sha256": asset_sha}
    write_json(ART / "detector_training_trials.json", dict(
        candidate_count=len(results), models={k: v["metrics"] for k,v in fitted.items()},
        training_annotations_sha256=digest(ART / "annotations.json"), candidates=results,
        ranking=["validation F1", "recall", "AP50–95", "fewer views", "fewer learned heads"]))
    write_json(FINAL, report)
    print("Release", selected, flush=True)


def rank(item):
    return (item["f1"], item["recall"], item["ap50_95"], -len(item["views"]), -len(item["heads"]))


def image_counts(rows, predictions, threshold):
    accepted = {i: [p for p in values if p["score"] >= threshold]
                for i, values in predictions.items()}
    _, used = match(rows, accepted, .5)
    return {r["id"]: [len(used[r["id"]]), len(accepted[r["id"]])-len(used[r["id"]]),
                      len(r["objects"])-len(used[r["id"]])] for r in rows}


def paired_interval(rows, baseline, optimized):
    by_group = {}
    for row in rows:
        group = row["group"]
        by_group.setdefault(group, np.zeros((2, 3), dtype=int))
        by_group[group] += np.array([baseline[row["id"]], optimized[row["id"]]])
    values = np.array(list(by_group.values()))
    rng = np.random.default_rng(607)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].sum(axis=1)
    tp, fp, fn = np.moveaxis(draws, -1, 0)
    denom = 2*tp+fp+fn
    f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    valid = (denom > 0).all(axis=1)
    differences = f1[valid, 1]-f1[valid, 0]
    return dict(author_groups=len(values), paired_bootstrap_f1_delta_95=np.percentile(
        differences, [2.5, 97.5]).tolist(), seed=607, draws=10000)


def verify(frozen, split):
    config = read_json(FINAL)
    if digest(HOLDOUT) != config["holdout_manifest_sha256"]:
        raise ValueError("Holdout references changed since selection.")
    if split == "holdout":
        rows = read_json(HOLDOUT)["rows"]
        for row in rows:
            download(row["image_url"], ROOT / "data" / row["image"])
    elif split == "test":
        rows = read_json(ART / "test_manifest.json")["rows"]
    else:
        rows = train_rows(ROOT)
    entries = cache(frozen, split, rows, config["views"])
    predictions = predict(entries, config["views"], config["strategy"], config["overlap"])
    if config["heads"]:
        if digest(WEIGHTS) != config["proposal_heads_sha256"]:
            raise ValueError("Proposal checkpoint checksum mismatch.")
        predictions = rescore(predictions, describe(rows, predictions), load(WEIGHTS),
                              config["weight"], config["overlap"])
    before = predict(entries, ["original"], "max", .6)
    baseline = evaluate(rows, before, frozen["threshold"])
    result = evaluate(rows, predictions, config["cutoff"])
    view_only = read_json(NEW)
    if set(view_only["views"]).issubset(config["views"]):
        ablation = predict(entries, view_only["views"], view_only["strategy"], view_only["overlap"])
        result["view_only"] = counters(evaluate(rows, ablation, view_only["cutoff"]))
    median_seconds = sorted(sum(e[v]["seconds"] for v in config["views"]) for e in entries.values())
    result.update(baseline=baseline, config_sha256=digest(FINAL),
                  median_proposal_seconds=median_seconds[len(median_seconds)//2],
                  scope={"holdout": "New independent authors, first scored after release selection",
                         "test": "Historical external regression", "train": "Development regression"}[split])
    counts = image_counts(rows, predictions, config["cutoff"])
    previous = image_counts(rows, before, frozen["threshold"])
    result["per_image_counts"] = {key: {"baseline": previous[key], "optimized": value}
                                  for key, value in counts.items()}
    if split == "holdout":
        result["uncertainty"] = paired_interval(rows, previous, counts)
    write_json(ART / f"optimized_{split}.json", result)
    print(split, "baseline", counters(baseline), "optimized", counters(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "holdout", "test", "train"])
    args = parser.parse_args()
    frozen = read_json(OLD)
    validate_local_inputs(frozen)
    if args.stage == "select":
        select(frozen)
    else:
        verify(frozen, args.stage)


if __name__ == "__main__":
    main()
