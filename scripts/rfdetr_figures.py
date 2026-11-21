"""Plot measured training curves and locked operating-point comparisons."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medsyringe.data import read_json
from scripts.rfdetr_compare import RUNS
from scripts.rfdetr_transfer import ART


def curve(run):
    with (ART / f"rfdetr_{run}_metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    return [(int(row["epoch"])+1, float(row["val/mAP_50_95"]))
            for row in rows if row["val/mAP_50_95"]]


def main():
    plots = ART / "figures"
    plots.mkdir(exist_ok=True)
    result = read_json(ART / "rfdetr_release_decision.json")["cohorts"]
    reserve = read_json(ART / "rfdetr_evaluation_fresh.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    names = {"public": "Zenodo masks", "direct": "Direct box adaptation",
             "transfer": "Mask-to-box transfer"}
    for run in RUNS:
        x, y = zip(*curve(run))
        axes[0].plot(x, y, "-o", label=names[run])
    axes[0].set(title="Within-source validation during training", xlabel="Epoch",
                ylabel="Box AP at IoU 0.50–0.95", xticks=(1, 2, 3), ylim=(0, 1))
    axes[0].legend(fontsize=8)

    values = [reserve["baseline"], *(reserve["standalone"][r] for r in RUNS), reserve["candidate"]]
    models = ["v3", "Zenodo", "Direct", "Transfer", "Hybrid"]
    for i, (key, label) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))):
        axes[1].bar(np.arange(len(models)) + (i-1)*.24, [v[key] for v in values], width=.24, label=label)
    axes[1].set(title="Reserved Zenodo clips", ylabel="Score at locked operating point",
                xticks=range(len(models)), xticklabels=models, ylim=(0, 1))
    axes[1].legend(fontsize=8)

    cohorts = ("fresh", "fil_rouge", "historical", "previous_external")
    labels = ["Zenodo\nreserve", "Fil rouge\ndevelopment", "Open Images\nregression", "Commons\nregression"]
    for i, (key, label) in enumerate((("tp", "True positives"), ("fp", "False positives"), ("fn", "False negatives"))):
        values = [result[c]["candidate"][key]-result[c]["baseline"][key] for c in cohorts]
        axes[2].bar(np.arange(len(cohorts)) + (i-1)*.24, values, width=.24, label=label)
        for j, count in enumerate(values):
            axes[2].text(j+(i-1)*.24, count, f"{count:+d}", ha="center", va="bottom")
    axes[2].set(title="Hybrid change relative to v3", ylabel="Change in detection count",
                xticks=range(len(cohorts)), xticklabels=labels)
    axes[2].axhline(0, color="gray", linewidth=.7)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    fig.savefig(plots / "rfdetr_transfer.png", dpi=180)


if __name__ == "__main__":
    main()
