"""Render publication-ready detection measurements from the saved study records."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from medsyringe.data import digest, read_image, read_json

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


def deployed_precision(destination):
    paths = ART/"precision_only_confirmation_audit.json", ART/"medbin_detector_audit.json"
    cohorts = read_json(paths[0])["outcomes"]
    medbin = read_json(paths[1])["precision_gate"]
    records = (("Earlier Commons", cohorts["commons-regression"]),
               ("Fresh Commons", cohorts["fresh"]),
               ("Native MedBin", medbin))
    fig,axes=plt.subplots(1,3,figsize=(11.7,4.4),layout="constrained",sharey=True)
    for ax,(title,record) in zip(axes,records,strict=True):
        prior=record["previous"]
        bar_groups(ax,{"Baseline":prior,"Candidate":record["candidate"]},title)
        ax.set_xlabel(f"{prior['images']} images · {prior['objects']} reference syringes")
    axes[0].set_ylabel("Instance detection at IoU ≥ 0.50")
    axes[-1].legend(["Earlier detector","Released gate"],loc="upper right",frameon=False,fontsize=8,ncol=2)
    fig.suptitle("Independent and regression results for the deployed appearance check",
                 x=.02,ha="left",fontweight="semibold",fontsize=14)
    fig.savefig(destination,dpi=215,facecolor="white")
    plt.close(fig)


def native_count_examples(bundle, destination):
    cohort = read_json(bundle/"cohorts/native-device.json")
    refs = {r["id"]: r for r in cohort["rows"]}
    observed = cohort["observations"]
    crowded = [r for r in refs.values() if len(r["objects"]) > 1]
    worst = max(crowded, key=lambda r: abs(len(r["objects"])-len(observed[r["id"]]["actual"])))
    exact = next(r for r in crowded if len(r["objects"]) == len(observed[r["id"]]["actual"]))
    figure, panels = plt.subplots(2,2,figsize=(12,7.5),layout="constrained")
    for row, axes in zip((exact,worst),panels,strict=True):
        path = bundle/"data"/row["image"]
        if digest(path) != row["sha256"]:
            raise ValueError("A figure source differs from the frozen native study.")
        raster = read_image(path)
        for ax, boxes, label, description, color in zip(
            axes,(row["objects"],observed[row["id"]]["actual"]),("Frozen reference","Automatic result"),
            ("visible instances","detections"),("#f8cc5c","#38cec2"),strict=True):
            ax.imshow(raster)
            for box in boxes:
                x,y,right,bottom = box["bbox"]
                ax.add_patch(Rectangle((x*raster.width,y*raster.height),(right-x)*raster.width,
                                       (bottom-y)*raster.height, fill=False, edgecolor=color, linewidth=1.4))
            ax.set_title(f"{label} · {len(boxes)} {description}",fontsize=10,loc="left")
            ax.axis("off")
    figure.legend(handles=[Patch(facecolor="#f8cc5c",label="Reference extent"),
                           Patch(facecolor="#38cec2",label="Detected extent")],
                  loc="outside lower center",ncols=2,frameon=False)
    figure.suptitle("An exact count and an undercount in the native device audit",
                    x=.02,ha="left",fontweight="semibold",fontsize=14)
    figure.savefig(destination,dpi=200,facecolor="white")
    plt.close(figure)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artifacts/figures"))
    parser.add_argument("--native-evidence",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    audit = ART/"crop_semantics_audit.json"
    select = ART/"detector_selection.json"
    if audit.exists():
        audit_bars(read_json(audit),args.output/"detection_generalization_audit.png")
    if select.exists():
        comparison(read_json(select),args.output/"detection_generalization_development.png")
    if (ART/"medbin_detector_audit.json").exists():
        deployed_precision(args.output/"deployed_precision_results.png")
    if args.native_evidence:
        native_count_examples(args.native_evidence,args.output/"native_device_count_examples.png")
