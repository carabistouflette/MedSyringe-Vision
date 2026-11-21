"""Calibrate an automatic SAM 3 fallback and audit it on a frozen reserve."""
import argparse
from itertools import chain, product
from pathlib import Path
from time import perf_counter

from medsyringe.additive import additions, overlapping_tiles
from medsyringe.auxiliary_detector import load_verified_recovery
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.optimize_detector import FINAL, counters
from scripts.precision_study import source
from scripts.rfdetr_compare import baseline_predictions
from scripts.rfdetr_verifier import baseline_for, inputs
from scripts.sam3_study import (ART, LOCK, MODELS, RELEASE, Segmenter, configure_cpu,
                                crop_predictions)
from scripts.zoom_study import regions

RESERVE = ART / "sam3_reserve.json"
FROZEN = ART / "sam3_fallback_selection.json"
VIEWS = ("full", "full_and_guided", "full_and_tiles")
CONTROLS = ("fil_rouge", "previous_external", "pen_reserve", "historical", "reserve")


def cohort_base(cohort):
    if cohort == "calibration":
        return [(name, inputs(name)) for name in ("calibration","syringe_reserve","pen_reserve")]
    if cohort == "reserve":
        return [("reserve",read_json(RESERVE)["rows"])]
    return [(cohort, inputs(cohort))]


def prepared_path(cohort):
    return MODELS/f"fallback-base-{cohort}.json"


def prepare(cohort):
    configure_cpu()
    runner = load_verified_recovery(Path("."), FINAL)
    if not runner:
        raise ValueError("The validated base release is required.")
    rows, base = [], {}
    floor = read_json(FINAL)["cutoff"]
    for name, subset in cohort_base(cohort):
        proposals = (baseline_predictions(subset,"sam3-reserve") if name=="reserve" else
                     baseline_for(name,subset))
        base.update({r["id"]:runner.apply(source(r),proposals[r["id"]],floor) for r in subset})
        rows.extend(subset)
    if len(rows)!=len(base):
        raise ValueError("Dataset identifiers are not unique.")
    write_json(prepared_path(cohort),dict(rows=rows,base=base,release_sha256=digest(RELEASE)))
    print(cohort,len(rows),"images",sum(not any(p["score"]>=floor for p in ps) for ps in base.values()),
          "empty baseline outputs",flush=True)


def get_prepared(cohort):
    payload=read_json(prepared_path(cohort))
    if payload["release_sha256"] != digest(RELEASE):
        raise ValueError("The active base release changed.")
    return payload["rows"],payload["base"]


def infer(cohort,edge,device,views):
    rows,base=get_prepared(cohort)
    floor=read_json(FINAL)["cutoff"]
    key=dict(prepared=digest(prepared_path(cohort)),checkpoint=digest(LOCK),
             code=digest(__file__),backend=digest(Path("scripts/sam3_study.py")),edge=edge,device=device)
    path=MODELS/f"fallback-{cohort}-{device}-{edge}.json"
    previous=read_json(path) if path.exists() else {}
    frames=previous["frames"] if previous.get("key")==key else {}
    segmenter=None
    for row in rows:
        identifier=row["id"]
        if any(p["score"]>=floor for p in base[identifier]):
            continue
        current=frames.setdefault(identifier,dict(proposals={},seconds={}))
        missing=[v for v in views if v not in current["proposals"]]
        if not missing:
            continue
        if segmenter is None:
            segmenter=Segmenter(edge,device)
        image=source(row)
        full=(0,0,*image.size)
        windows=dict(full=[full],full_and_guided=[full,*regions(base[identifier],image.size,floor)],
                     full_and_tiles=[full,*overlapping_tiles(image.size,.66)])
        local={}
        for view in missing:
            started=perf_counter()
            for window in windows[view]:
                if window not in local:
                    local[window]=list(crop_predictions(segmenter.predict,image,[window]))
            current["proposals"][view]=suppress(list(chain.from_iterable(local[w] for w in windows[view])),.45)
            current["seconds"][view]=perf_counter()-started
        write_json(path,dict(key=key,frames=frames))
        print(cohort,identifier,"seconds",round(sum(current["seconds"].values()),2),flush=True)
    return rows,base,frames


def combine(base,frames,floor,view,aux_cutoff,support):
    return {key:additions(ranked,frames[key]["proposals"][view],floor,support=support,
                          aux_cutoff=aux_cutoff,iou=.30) if key in frames else ranked
            for key,ranked in base.items()}


def hashes():
    paths=(RESERVE, LOCK, RELEASE, Path(__file__), Path("scripts/sam3_study.py"),prepared_path("calibration"))
    return {path.name:digest(path) for path in paths}


def select(edge,device,modes):
    rows,base,frames=infer("calibration",edge,device,modes)
    floor=read_json(FINAL)["cutoff"]
    baseline=evaluate(rows,base,floor)
    candidates=[]
    for view,threshold,support in product(modes,(.35,.5,.65,.8,.9), (None,.30)):
        options=dict(view=view,aux_cutoff=threshold,support=support)
        predictions=combine(base,frames,floor,**options)
        result=evaluate(rows,predictions,floor)
        guard=compare(baseline,result)
        candidates.append(dict(options=options,**counters(result),guard=guard,
                               eligible=guard["passed"] and result["fp"]<=baseline["fp"] and result["tp"]>baseline["tp"]))
    candidates.sort(key=lambda c:(c["eligible"],c["f1"],c["recall"],c["options"]["view"]=="full"),
                    reverse=True)
    selected=next((c for c in candidates if c["eligible"]),None)
    write_json(FROZEN,dict(selected=selected,edge=edge,device=device,inputs=hashes(),
                           trials=candidates,baseline=counters(baseline)))
    print("Calibration",counters(baseline),"selected",selected,flush=True)


def audit(cohort):
    record=read_json(FROZEN)
    if not record["selected"] or record["inputs"] != hashes():
        raise ValueError("The study requires an eligible unchanged frozen calibration.")
    options=record["selected"]["options"]
    rows,base,frames=infer(cohort,record["edge"],record["device"],(options["view"],))
    floor=read_json(FINAL)["cutoff"]
    baseline=evaluate(rows,base,floor)
    candidate=evaluate(rows,combine(base,frames,floor,**options),floor)
    guard=compare(baseline,candidate)
    write_json(ART/f"sam3_audit_{cohort}.json",dict(
        baseline=baseline,candidate=candidate,guard=guard,activated_images=len(frames),
        frozen_sha256=digest(FROZEN),processing_seconds=sum(f["seconds"][options["view"]] for f in frames.values())))
    print(cohort,counters(baseline),counters(candidate),guard,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","select","audit"))
    parser.add_argument("--cohort",choices=("calibration",*CONTROLS))
    parser.add_argument("--edge",type=int,choices=(504,672,1008),default=504)
    parser.add_argument("--device",choices=("cpu","xpu"),default="cpu")
    parser.add_argument("--views",nargs="+",choices=VIEWS,default=["full","full_and_guided"])
    args=parser.parse_args()
    if args.stage in ("prepare","audit") and not args.cohort:
        parser.error("A cohort is required.")
    if args.stage=="prepare":
        if args.cohort=="reserve" and not (FROZEN.exists() and read_json(FROZEN)["selected"]):
            parser.error("Freeze an eligible calibration before opening the reserve.")
        prepare(args.cohort)
    else:
        configure_cpu()
        if args.stage=="select":
            select(args.edge,args.device,args.views)
        else:
            audit(args.cohort)


if __name__=="__main__":
    main()
