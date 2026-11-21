"""Compare local full-frame, tiled, and corroborated syringe detection."""
import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.additive import duplicates, overlapping_tiles, project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_hillclimb_corpus import ART, DATA, LOCK
from scripts.detection_hillclimb_train import INITIAL, INITIAL_RECORD
from scripts.expanded_detection_study import short
from scripts.expanded_proposals import rows_for
from scripts.precision_study import source

PUBLIC = ART / "public_manifest.json"
TRAINED = ART / "rfdetr_whole-frame_training.json"
FLOOR = Path("artifacts/detector_release.json")
EXISTING = Path("artifacts/expanded/crop_verifier_release.json")
SELECTED = ART / "detector_selection.json"
AUDIT = ART / "detector_audit.json"
DEVELOPMENT = ("phlebotomy-development","public-development")
HELD_OUT = ("public-audit","previous-reserve")
BACKENDS = ("baseline","original-rf","adapted-rf")


def manifest(name):
    if name == "phlebotomy-development":
        return LOCK
    if name.startswith("public-"):
        return PUBLIC
    raise ValueError(f"No standalone manifest for {name}.")


def inputs(name):
    if name == "previous-reserve":
        return rows_for("reserve")
    role = "development" if name == "phlebotomy-development" else name
    return [r for r in read_json(manifest(name))["rows"] if r["split"] == role]


def cache(scope, backend):
    return DATA / "proposals" / f"{scope}-{backend}.json"


def trainer(backend):
    if backend == "original-rf":
        settings = read_json(INITIAL_RECORD)
        checkpoint = INITIAL
    else:
        settings = read_json(TRAINED)
        checkpoint = Path(settings["checkpoint"])
    if settings["status"] != "completed" or digest(checkpoint) != settings["checkpoint_sha256"]:
        raise ValueError("The checkpoint does not match the training record.")
    return checkpoint, settings["train"]["resolution"]


def collection_key(scope, backend):
    rows = inputs(scope)
    label_hash = digest(manifest(scope)) if scope != "previous-reserve" else digest("artifacts/sam3_reserve.json")
    weights_hash = digest(EXISTING) if backend == "baseline" else digest(trainer(backend)[0])
    return dict(label_sha256=label_hash, rows={r["id"]: r.get("derived_sha256",r["sha256"]) for r in rows},
                backend=backend, weights_sha256=weights_hash, code_sha256=digest(__file__))


def tiles(image):
    yield from overlapping_tiles(image.size,.66)
    width,height = image.size
    w,h = round(width*.47),round(height*.47)
    for x,y in product((0,(width-w)//2,width-w),(0,(height-h)//2,height-h)):
        yield x,y,x+w,y+h


def collect(scope, backend, device):
    from scripts.sam3_study import configure_cpu

    if scope in HELD_OUT and not SELECTED.exists():
        raise ValueError("Freeze a development decision before opening a reserved inference cache.")
    configure_cpu()
    destination = cache(scope,backend)
    binding = collection_key(scope,backend)
    previous = read_json(destination) if destination.exists() else dict(key=binding, frames={})
    if previous["key"] != binding:
        raise ValueError("The existing proposal cache differs from its inputs.")
    rows = inputs(scope)
    if not rows:
        raise ValueError(f"No reference rows for {scope}.")
    frames = previous["frames"]
    if len(frames) == len(rows):
        return
    engine = None
    try:
        if backend == "baseline":
            from medsyringe.grounding_backend import enable_intel
            from medsyringe.production import ProductionEngine

            engine = ProductionEngine(Path("."))
            if device == "xpu":
                enable_intel(engine,Path("."))
                if engine.identity.get("primary_backend") != "xpu":
                    raise ValueError("The requested XPU backend did not activate.")
        else:
            from rfdetr import RFDETRSegNano
            from scripts.rfdetr_compare import ranked_region

            checkpoint, edge = trainer(backend)
            model = RFDETRSegNano.from_checkpoint(str(checkpoint),device="cpu")
            if model.class_names != ["syringe"]:
                raise ValueError("Unexpected trained class list.")
        for row in rows:
            name = row["id"]
            if name in frames:
                continue
            image = source(row)
            tick = perf_counter()
            if engine is not None:
                ranked, floor, _ = engine.expanded.detect(image)
                result = dict(ranked=ranked, floor=floor, device=device)
            else:
                full = suppress(list(ranked_region(model,image,edge)),.45)
                windows = []
                for index, region in enumerate(tiles(image)):
                    for value in ranked_region(model,image.crop(region),edge):
                        projected = project(value["bbox"], region,image.size)
                        if projected and value["score"] >= .008:
                            windows.append({**value,"bbox":projected,"view":index})
                result = dict(full=full, windows=windows)
            result["ms"] = (perf_counter()-tick)*1000
            frames[name] = result
            if len(frames)%4 == 0 or len(frames) == len(rows):
                write_json(destination,previous)
                print(scope,backend,len(frames),"/",len(rows),flush=True)
    finally:
        if engine is not None:
            engine.close()


def loaded(scope, backend):
    payload = read_json(cache(scope,backend))
    if payload["key"] != collection_key(scope,backend):
        raise ValueError("Proposals are from another code, checkpoint or dataset version.")
    if set(payload["frames"]) != {r["id"] for r in inputs(scope)}:
        raise ValueError("The proposal collection is incomplete.")
    return payload["frames"]


def proposals(frame, view):
    if view == "whole":
        return frame["full"]
    windows = frame["windows"]
    if view == "wide":
        return suppress([*frame["full"], *(p for p in windows if p["view"] < 4)], .40)
    if view == "mixed":
        return suppress([*frame["full"],*windows], .40)
    ordered = suppress([*frame["full"],*windows], .40)
    stable = []
    for proposal in ordered:
        box = proposal["bbox"]
        supporting = {p["view"] for p in windows if p["score"] >= .025 and box_iou(box,p["bbox"]) >= .48}
        global_match = any(p["score"] >= .02 and box_iou(box,p["bbox"]) >= .48 for p in frame["full"])
        if len(supporting) + global_match >= 2:
            stable.append(proposal)
    return stable


def hybrid(primary, additional, policy):
    floor = primary["floor"]
    ranked = primary["ranked"]
    accepted = [p for p in ranked if p["score"] >= floor]
    if accepted and policy["scope"] == "empty":
        return ranked
    refer = [p for p in ranked if .26 <= p["score"] < floor and "original_score" not in p]
    vetoed = [p for p in ranked if "crop_score" in p]
    added = []
    for proposal in proposals(additional,policy["view"]):
        if proposal["score"] < policy["threshold"]:
            continue
        box = proposal["bbox"]
        if any(duplicates(box,p["bbox"]) for p in (*accepted,*vetoed,*added)):
            continue
        if policy["require_grounding"] and not any(box_iou(box,p["bbox"]) >= .38 for p in refer):
            continue
        added.append({**proposal,"score":floor,"source":"hillclimb RF-DETR",
                      "auxiliary_score":proposal["score"]})
    return [*accepted,*added,*(p for p in ranked if p["score"] < floor)]


def model_eval(rows, base, rf, policy):
    floor = read_json(FLOOR)["cutoff"]
    prediction = {r["id"]: hybrid(base[r["id"]],rf[r["id"]],policy) for r in rows}
    metrics = evaluate(rows,prediction,floor)
    return metrics


def statistics(name, base, variants, policy):
    rows = inputs(name)
    previous = evaluate(rows,{key: value["ranked"] for key,value in base.items()},read_json(FLOOR)["cutoff"])
    after = model_eval(rows,base,variants,policy)
    return dict(baseline=short(previous), candidate=short(after), guard=compare(previous,after),
                misses=after["false_negatives"], false_positives=after["false_positives"])


def select():
    if SELECTED.exists():
        raise ValueError("Selection has already been fixed.")
    model_names = BACKENDS[1:]
    blobs = {g: {m: loaded(g,m) for m in BACKENDS} for g in DEVELOPMENT}
    thresholds = (.025,.06,.12,.23,.40,.60,.82)
    trials = []
    # Cache per-view candidates because model proposals themselves do not
    # depend on the operating point or the proposed rescue gate.
    views = ("whole","wide","mixed","stable")
    controls = {g: evaluate(inputs(g),{k:v["ranked"] for k,v in blobs[g]["baseline"].items()},
                           read_json(FLOOR)["cutoff"]) for g in DEVELOPMENT}
    counts = {g: inputs(g) for g in DEVELOPMENT}
    for model,view,limit,require,scene_scope in product(model_names,views,thresholds,(False,True),("all","empty")):
        option = dict(model=model,view=view,threshold=limit,require_grounding=require,scope=scene_scope)
        outcomes = {}
        for g in DEVELOPMENT:
            val = model_eval(counts[g],blobs[g]["baseline"],blobs[g][model],option)
            guard = compare(controls[g],val)
            outcomes[g] = dict(result=short(val), guard=guard, extra_fp=val["fp"]-controls[g]["fp"])
        trials.append(dict(policy=option,outcome=outcomes))
    def key(trial):
        states = trial["outcome"]
        return (states["public-development"]["guard"]["recovered_matches"],
                sum(g["guard"]["recovered_matches"] for g in states.values()),
                -sum(g["extra_fp"] for g in states.values()), trial["policy"]["threshold"])
    candidates = [t for t in trials if all(o["guard"]["passed"] and o["extra_fp"] <= 0 for o in t["outcome"].values())
                  and sum(o["guard"]["recovered_matches"] for o in t["outcome"].values()) > 0]
    chosen = max(candidates,key=key,default=None)
    shadow = chosen or max(trials,key=lambda t:(-sum(o["extra_fp"] for o in t["outcome"].values()),*key(t)))
    write_json(SELECTED,dict(selected=chosen, shadow=shadow, trials=trials,
              baseline={g:short(v) for g,v in controls.items()},
              inputs={str(cache(g,m)):digest(cache(g,m)) for g in DEVELOPMENT for m in BACKENDS}))
    print("Selection", chosen, "diagnostic fallback", shadow if chosen is None else None,flush=True)


def audit():
    selection = read_json(SELECTED)
    if any(digest(path) != checksum for path,checksum in selection["inputs"].items()):
        raise ValueError("Calibration evidence changed after selection.")
    policy = selection["shadow"]["policy"]
    cases = {name: statistics(name,loaded(name,"baseline"),loaded(name,policy["model"]),policy) for name in HELD_OUT}
    passed = bool(selection["selected"] and all(r["guard"]["passed"] and
                  r["candidate"]["fp"] <= r["baseline"]["fp"] for r in cases.values()) and
                  cases["public-audit"]["guard"]["recovered_matches"])
    timings = {}
    for name in HELD_OUT:
        timings[name] = {m: median(v["ms"] for v in loaded(name,m).values())
                         for m in ("baseline",policy["model"])}
    write_json(AUDIT,dict(passed=passed,cases=cases,selection_sha256=digest(SELECTED),
                          median_collection_ms=timings))
    print("Audit",passed,{k:(v["baseline"],v["candidate"],v["guard"]) for k,v in cases.items()},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","select","audit"))
    parser.add_argument("--scope",choices=(*DEVELOPMENT,*HELD_OUT))
    parser.add_argument("--backend",choices=BACKENDS)
    parser.add_argument("--device",choices=("cpu","xpu"),default="cpu")
    args = parser.parse_args()
    if args.action == "collect":
        if not args.scope or not args.backend:
            parser.error("Collection needs both --scope and --backend.")
        collect(args.scope,args.backend,args.device)
    else:
        (select if args.action == "select" else audit)()
