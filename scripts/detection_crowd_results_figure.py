"""Draw the completed orientation and supervised crowd development comparisons."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from medsyringe.data import read_json
from scripts.detection_crowd_training_evaluate import OUTPUT as SUPERVISED
from scripts.detection_deskew_additive import OUTPUT as NATURAL
from scripts.detection_deskew_study import OUTPUT as OVERLAP

OUTPUT = Path("artifacts/figures/dense_scene_development_comparisons.png")


def draw_panel(axis, title, subtitle, entries):
    axis.set_title(title, loc="left", pad=25, fontweight="bold", fontsize=12)
    axis.text(0, 1.035, subtitle, transform=axis.transAxes, color="#4b5563", fontsize=9)
    for index, (label, result) in enumerate(entries):
        hit = result["tp"]
        false = result["fp"]
        axis.barh(index, hit, height=.55, color="#256863")
        if false:
            axis.barh(index, false, left=hit, height=.55, color="#b76527")
        axis.text(hit + false + 1.1, index, f"{hit} matched / {false} false",
                  va="center", fontsize=8, color="#253349")
    axis.set_yticks(range(len(entries)), [label for label, _ in entries], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0, max(r["tp"] + r["fp"] for _, r in entries) + 20)
    axis.tick_params(length=0)
    axis.tick_params(axis="x", labelsize=8)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#e6eaf1")
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)


def main():
    orientation = read_json(OVERLAP)["outcomes"]
    trained = read_json(SUPERVISED)["outcomes"]
    natural = read_json(NATURAL)["natural"]["transparent"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), layout="constrained")
    fig.suptitle("Orientation and crowd detection development", fontsize=17, fontweight="bold", x=.02, ha="left")
    draw_panel(axes[0], "Released hybrid", "Synthetic overlap stress · 54 visible instances", [
        ("Original image", orientation["baseline"]), ("Dominant-axis view", orientation["estimated"]),
        ("Known angle diagnostic", orientation["oracle"]),
    ])
    draw_panel(axes[1], "RF-DETR paired training", "Same 54 synthetic instances · fixed score 0.20", [
        ("Separated · original", trained["separated"]["parallel"]["all"]["suppressed"]),
        ("Separated · dominant axis", trained["separated"]["parallel"]["all"]["deskewed"]),
        ("Overlapping · original", trained["overlapping"]["parallel"]["all"]["suppressed"]),
        ("Overlapping · dominant axis", trained["overlapping"]["parallel"]["all"]["deskewed"]),
    ])
    draw_panel(axes[2], "Recovery on empty outputs", "Native transparent-device development · 20 visible instances", [
        ("Current hybrid", natural["baseline"]), ("Conditional orientation", natural["prospective"]),
    ])
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
