"""Render publication-ready detection measurements from the saved study records."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medsyringe.data import read_json

ART = Path("artifacts/detection_hillclimb")
COLORS = {"Baseline":"#7b8a96", "Candidate":"#167a71"}


def bar_groups(axes, groups, title):
    offsets = np.arange(3)
    keys = ("precision", "recall", "f1")
    for i, (name, stats) in enumerate(groups.items()):
        values = [stats[k] for k in keys]
        bars = axes.bar(offsets + (i-.5)*.31, values, width=.28, color=COLORS[name], label=name)
        axes.bar_label(bars, [f"{v:.0%}" for v in values], padding=3, fontsize=8)
    axes.set_title(title,loc="left",fontweight="semibold",fontsize=10)
    axes.set_ylim(0,1.12)
    axes.set_yticks([0,.25,.5,.75,1.], ["0%","25%","50%","75%","100%"])
    axes.set_xticks(offsets,[k.title() if k!="f1" else "F1" for k in keys])
    axes.grid(visible=True,axis="y",color="#efefef")
    axes.set_axisbelow(True)
    axes.spines[["right","top"]].set_visible(False)


def audit_bars(proof, destination):
    title = {"public-audit":"External public photographs", "previous-reserve":"Historical public regression",
             "dental":"Dental-instrument challenge"}
    records = proof["outcomes"]
    fig, plots = plt.subplots(1,len(records),figsize=(11,4.15),layout="constrained")
    for ax, (scope, cohort) in zip(np.atleast_1d(plots),records.items(),strict=True):
        baseline = cohort["previous"]
        bar_groups(ax,{"Baseline":baseline,"Candidate":cohort["candidate"]},title[scope])
        ax.set_xlabel(f"{baseline['images']} images · {baseline['objects']} reference syringes")
    plots = np.atleast_1d(plots)
    plots[-1].legend(loc="upper right",frameon=False,fontsize=8,ncol=2)
    fig.suptitle("Frozen crop policy at IoU ≥ 0.50",x=.01,ha="left",fontweight="semibold",fontsize=15)
    fig.savefig(destination,dpi=210,facecolor="white")
    plt.close(fig)


def comparison(proof, destination):
    names = ("public-development", "phlebotomy-development")
    titles = ("Public photographs", "Simulated phlebotomy")
    items = proof["standalone"]
    fig, plots = plt.subplots(1,2,figsize=(12,4.7),layout="constrained",sharey=True,sharex=True)
    for ax, scope, title in zip(plots,names,titles,strict=True):
        for name, views in items.items():
            values = {label:variant["scopes"][scope] for label,variant in views.items()}
            best_view, result = max(values.items(),key=lambda pair:pair[1]["f1"])
            ax.scatter(result["recall"],result["precision"],label=f"{name} · {best_view}",s=34)
        baseline = proof["baseline"][scope]
        ax.scatter(baseline["recall"],baseline["precision"],s=80,marker="*",color="#1c2224",label="Released hybrid")
        ax.set_title(title,loc="left",fontsize=11,fontweight="semibold")
        ax.set_xlabel("Recall")
        ax.set_xlim(-.03,1.03)
        ax.set_ylim(-.03,1.03)
        ax.grid(visible=True,color="#eeeeee")
        ax.set_axisbelow(True)
        ax.spines[["right","top"]].set_visible(False)
        ax.legend(fontsize=6.5,loc="upper right",frameon=False,ncol=1)
    plots[0].set_ylabel("Precision")
    fig.suptitle("Model and view comparison on development data",x=.01,ha="left",fontweight="semibold",fontsize=15)
    fig.savefig(destination,dpi=210,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artifacts/figures"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    audit = ART/"crop_semantics_audit.json"
    select = ART/"detector_selection.json"
    if audit.exists():
        audit_bars(read_json(audit),args.output/"detection_generalization_audit.png")
    if select.exists():
        comparison(read_json(select),args.output/"detection_generalization_development.png")
