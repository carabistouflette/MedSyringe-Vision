"""Compare multi-view detectors locally and freeze a unified operating point."""
import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

from medsyringe.data import digest, download, leakage, read_image, read_json, train_rows, write_json
from medsyringe.detectors import Grounder, model_hashes
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.refinement import VIEWS, fuse, view_proposals

ROOT = Path(".")
ART = ROOT / "artifacts"
OLD = ART / "freeze.json"
NEW = ART / "detector_v2.json"
HOLDOUT = ART / "v2_holdout_manifest.json"


def cache(frozen, split, rows, views):
    path = ROOT / "data/cache" / f"detector-v2-{split}.json"
    key = dict(weights=frozen["bundle_sha256"], views_code=digest("medsyringe/refinement.py"),
               old_code=frozen["code_sha256"], rows={r["id"]: r.get("derived_sha256", r["sha256"])
                                                    for r in rows})
    saved = read_json(path) if path.exists() else {}
    entries = saved["entries"] if saved.get("key") == key else {}
    old_path = ROOT / "data/cache" / f"dino-{split}-original.json"
    old = read_json(old_path) if old_path.exists() else {}
    old_config = old.get("config", {})
    reusable = (old_config.get("weights") == key["weights"] and
                old_config.get("runtime") == {k: v for k, v in key["old_code"].items()
                                             if k in ("data.py", "detectors.py")} and
                dict(old_config.get("rows", [])) == key["rows"])
    detector = None
    for i, row in enumerate(rows, 1):
        identifier = row["id"]
        present = entries.setdefault(identifier, {})
        missing = [v for v in views if v not in present]
        if not missing:
            continue
        if reusable and "original" in missing and identifier in old["predictions"]:
            present["original"] = old["predictions"][identifier]
            missing.remove("original")
        if missing:
            if detector is None:
                detector = Grounder(ROOT, frozen["backbone"])
            im_path = ROOT / "data" / row["image"]
            if digest(im_path) != key["rows"][identifier]:
                raise ValueError(f"Image hash mismatch on {identifier}")
            image = read_image(im_path)
            for view in missing:
                started = perf_counter()
                proposals = view_proposals(detector, image, view)
                present[view] = dict(proposals=proposals, seconds=perf_counter()-started)
        if i % 5 == 0 or i == len(rows):
            write_json(path, dict(key=key, entries=entries))
            print(split, i, "/", len(rows), flush=True)
    return entries


def predict(entries, views, strategy, overlap):
    return {
        identifier: fuse({v: candidate[v]["proposals"] for v in views},
                         strategy=strategy, overlap=overlap)
        for identifier, candidate in entries.items()
    }


def select(frozen):
    if NEW.exists():
        raise ValueError("This selection is already frozen.")
    rows = read_json(ART / "validation_manifest.json")["rows"]
    published = read_json(HOLDOUT)["rows"]
    prior = read_json(ART / "test_manifest.json")["rows"]
    if leakage([*train_rows(ROOT), *rows, *prior, *published]):
        raise ValueError("Cross-partition collision in the holdout design.")
    entries = cache(frozen, "validation", rows, VIEWS)
    configurations = [(("original",), "max", .6)]
    for n in range(1, len(VIEWS)):
        for rest in combinations(VIEWS[1:], n):
            for strategy in ("max", "consensus"):
                for overlap in (.6, .75):
                    configurations.append((("original", *rest), strategy, overlap))
    trials = []
    for views, strategy, overlap in configurations:
        predictions = predict(entries, views, strategy, overlap)
        threshold = best_threshold(rows, predictions)
        metrics = evaluate(rows, predictions, threshold)
        trials.append(dict(views=views, strategy=strategy, overlap=overlap, cutoff=threshold,
                           **{k: metrics[k] for k in ("precision", "recall", "f1", "ap50",
                                                      "ap50_95", "exact_count", "tp", "fp", "fn")}))
    trials.sort(key=lambda t: (t["f1"], t["recall"], t["ap50_95"], -len(t["views"])), reverse=True)
    chosen = trials[0]
    selection = dict(id="multiview-v2", original_freeze_sha256=digest(OLD),
                     refinement_sha256=digest("medsyringe/refinement.py"),
                     validation_sha256=digest(ART / "validation_manifest.json"),
                     holdout_manifest_sha256=digest(HOLDOUT), **chosen)
    write_json(ART / "v2_search.json", dict(
        scope="Common validation set only. No former test predictions used for selection.",
        selection_order=["f1", "recall", "ap50_95", "fewer views"], trials=trials))
    print("Selected", chosen, flush=True)
    write_json(NEW, selection)


def verify(frozen, split):
    config = read_json(NEW)
    manifest = ART / f"{split}_manifest.json"
    if split == "holdout":
        if config["holdout_manifest_sha256"] != digest(HOLDOUT):
            raise ValueError("Fresh holdout references changed since selection.")
        rows = read_json(HOLDOUT)["rows"]
        for row in rows:
            download(row["image_url"], ROOT / "data" / row["image"])
    else:
        rows = train_rows(ROOT) if split == "train" else read_json(manifest)["rows"]
    entries = cache(frozen, split, rows, config["views"])
    predictions = predict(entries, config["views"], config["strategy"], config["overlap"])
    report = evaluate(rows, predictions, config["cutoff"])
    baseline = evaluate(rows, predict(entries, ("original",), "max", .6), frozen["threshold"])
    report.update(config_sha256=digest(NEW), baseline=baseline,
                  scope={
                      "train": "Development regression against draft references",
                      "test": "Previously inspected external test; historical regression",
                      "holdout": "Newly labeled author-separated external holdout, first scored after selection",
                  }[split])
    times = [1000 * sum(entry[view]["seconds"] for view in config["views"])
             for entry in entries.values()]
    report["median_proposal_ms"] = sorted(times)[len(times)//2]
    write_json(ART / f"v2_{split}.json", report)
    print(split, {k: report[k] for k in ("precision", "recall", "f1", "ap50", "ap50_95",
                                         "exact_count", "tp", "fp", "fn")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "train", "test", "holdout"])
    opts = parser.parse_args()
    frozen = read_json(OLD)
    if model_hashes(ROOT, frozen["backbone"]) != frozen["bundle_sha256"]:
        raise ValueError("Backbone weights do not match the frozen bundle.")
    if opts.stage == "select":
        select(frozen)
    else:
        verify(frozen, opts.stage)


if __name__ == "__main__":
    main()
