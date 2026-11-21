"""Render publication figures and image-level error summaries from measured artifacts."""
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from medsyringe.data import read_image, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import match

PALETTE = ("#14584d", "#65a997", "#db9c42")
LABELS = {"dino": "Grounding DINO Tiny", "owl": "OWLv2 Base",
          "dino-trained": "DINO + local re-ranker", "dino-augmented": "DINO + augmented re-ranker"}


def test_uncertainty(rows, predictions, threshold):
    groups = defaultdict(lambda: np.zeros(3, dtype=int))
    group_of = {row["id"]: row["group"] for row in rows}
    for row in rows:
        groups[row["group"]][2] += len(row["objects"])
    accepted = {key: [p for p in values if p["score"] >= threshold] for key, values in predictions.items()}
    for identifier, _, target, _ in match(rows, accepted, 0.5)[0]:
        groups[group_of[identifier]][0 if target is not None else 1] += 1
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(607)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].sum(axis=1)
    tp, fp, actual = samples.T
    recall = tp[actual>0] / actual[actual>0]
    precision = tp[tp+fp>0] / (tp+fp)[tp+fp>0]
    return dict(method=("Percentile bootstrap by author group. A zero-false-positive sample "
                        "gives a degenerate empirical precision distribution."),
                seed=607, groups=len(values), draws=10000,
                recall_95=np.percentile(recall, [2.5, 97.5]).tolist(),
                precision_empirical_95=(np.percentile(precision, [2.5, 97.5]).tolist()
                                        if np.any(values[:, 1]) else None))


def style(axes):
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=.22)
    axes.set_axisbelow(True)


def validation_figure(results, target):
    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    x = np.arange(len(results))
    ax.bar(x-.17, [r["ap50"] for r in results.values()], .34, color=PALETTE[0], label="AP@50")
    ax.bar(x+.17, [r["ap50_95"] for r in results.values()], .34, color=PALETTE[1], label="AP@50–95")
    ax.set(xticks=x, xticklabels=[LABELS[m].replace(" + ", "\n+ ") for m in results], ylim=(0, 1),
           title="Detector comparison on Wikimedia Commons validation", ylabel="Average precision")
    ax.legend(frameon=False)
    style(ax)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def test_figure(result, target):
    fig, (ax, other) = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained")
    keys = ("single", "multiple", "annotated-occlusion", "no-annotated-occlusion")
    slices = [result["slices"][k] for k in keys]
    positions = range(len(keys))
    ax.bar(positions, [r["recall"] for r in slices], color=PALETTE[0], width=.58)
    for i, r in enumerate(slices):
        ax.text(i, r["recall"]+.035, f"{r['tp']}/{r['objects']}", ha="center")
    ax.set(xticks=list(positions), xticklabels=["Single", "Multiple", "Occlusion", "No occlusion"],
           ylim=(0, 1.19), ylabel="Recall", title="Recall by scene condition")
    style(ax)
    xs = [float(k) for k in result["ap_by_iou"]]
    other.plot(xs, list(result["ap_by_iou"].values()), marker="o", color=PALETTE[0])
    other.set(xlabel="IoU threshold", ylabel="Average precision", ylim=(0, 1),
              title="Localization across IoU thresholds")
    style(other)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def attribute_figure(results, target):
    fig, ax = plt.subplots(figsize=(7.8, 4.0), layout="constrained")
    x = np.arange(len(results))
    correct = np.array([r["correct"] for r in results.values()])
    wrong = np.array([r["answered"]-r["correct"] for r in results.values()])
    unknown = np.array([r["total"]-r["answered"] for r in results.values()])
    ax.bar(x, correct, .58, color=PALETTE[0], label="Correct")
    ax.bar(x, wrong, .58, bottom=correct, color=PALETTE[2], label="Incorrect")
    ax.bar(x, unknown, .58, bottom=correct+wrong, color="#cdd8d4", label="Abstained")
    ax.set(xticks=x, xticklabels=[m.replace("-", "\n").title() for m in results], ylim=(0, 82),
           ylabel="Draft labeled frames", title="Same-session leave-burst-out attribute diagnostic")
    ax.legend(frameon=False, ncol=3)
    style(ax)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def draw_scene(image, boxes, predicted, name):
    tile = Image.new("RGB", (380, 320), "#ffffff")
    image.thumbnail((360, 275))
    x, y = (380-image.width)//2, (285-image.height)//2
    tile.paste(image, (x, y))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default(size=14)
    for regions, color, width in ((boxes, "#e3a410", 4), (predicted, "#2ffabb", 2)):
        for l, t, r, b in regions:
            draw.rectangle((x+l*image.width, y+t*image.height, x+r*image.width, y+b*image.height),
                           outline=color, width=width)
    draw.text((12, 294), name, fill="#172426", font=font)
    return tile


def error_contact(rows, predictions, threshold, target):
    cols = 3
    canvas = Image.new("RGB", (380*cols, 320*((len(rows)+cols-1)//cols)), "#f0f1ee")
    for i, row in enumerate(rows):
        image = read_image(Path("data") / row["image"])
        truth = [obj["bbox"] for obj in row["objects"]]
        hits = [p["bbox"] for p in predictions[row["id"]] if p["score"] >= threshold]
        tile = draw_scene(image, truth, hits, f"{row['id']} · gold {len(truth)} · accepted {len(hits)}")
        canvas.paste(tile, ((i % cols)*380, (i//cols)*320))
    canvas.save(target)


def optimization_figure(artifacts, target):
    holdout = read_json(artifacts / "optimized_holdout.json")
    labels = ["Precision", "Recall", "F1", "AP@50", "AP@50–95"]
    keys = ("precision", "recall", "f1", "ap50", "ap50_95")
    before = [holdout["baseline"][k] for k in keys]
    after = [holdout[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
    x = np.arange(len(keys))
    first = ax.bar(x-.17, before, .34, color="#a6b4b2", label="Single-view baseline")
    second = ax.bar(x+.17, after, .34, color=PALETTE[0], label="Unified detector")
    ax.bar_label(first, fmt="%.2f", padding=4, fontsize=9)
    ax.bar_label(second, fmt="%.2f", padding=4, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.16),
           title="New external holdout · 24 images, 20 reference boxes")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def main():
    documents = Path("artifacts")
    figures = documents / "figures"
    figures.mkdir(exist_ok=True)
    results = read_json(documents / "validation_results.json")
    test = read_json(documents / "test_results.json")
    rows = read_json(documents / "test_manifest.json")["rows"]
    predictions = read_json(documents / "test_predictions.json")
    threshold = test["threshold"]
    uncertainty = test_uncertainty(rows, predictions, threshold)
    errors = []
    for miss in test["false_negatives"]:
        candidates = predictions[miss["id"]]
        matches = [p for p in candidates if box_iou(miss["box"], p["bbox"]) >= 0.50]
        candidate = max(matches, key=lambda p: p["score"], default=None)
        errors.append({**miss, "best_candidate_at_iou_50": candidate,
                       "best_available_iou": max((box_iou(miss["box"], p["bbox"]) for p in candidates),
                                                 default=0)})
    overlap = read_json(documents / "dataset_summary.json")["draft_top_proposal_agreement"]
    write_json(documents / "error_analysis.json", dict(test_uncertainty=uncertainty,
        missed_boxes=errors, train_draft_vs_proposal=dict(median_iou=median(x["top_proposal_iou"] for x in overlap),
            boxes_at_75=sum(x["top_proposal_iou"] >= .75 for x in overlap), n=len(overlap))))
    validation_figure(results, figures / "validation_models.png")
    test_figure(test, figures / "external_test.png")
    attribute_figure(read_json(documents / "attribute_diagnostics.json"), figures / "attribute_diagnostics.png")
    local_figures = Path("data/figures")
    local_figures.mkdir(exist_ok=True)
    error_contact(rows, predictions, threshold, local_figures / "external_test_overlays.jpg")
    if (documents / "optimized_holdout.json").is_file():
        optimization_figure(documents, figures / "unified_detector_holdout.png")
    print("Reports and figures generated.")


if __name__ == "__main__":
    main()
