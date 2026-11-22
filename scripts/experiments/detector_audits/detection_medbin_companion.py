"""Check the preregistered OpenMMLab witness on the medical-waste reserve."""
import argparse
from pathlib import Path

from medsyringe.detection.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from scripts.workers.appearance_text_encoder import TextMarginEncoder
from scripts.experiments.detector_audits.detection_auxiliary_choice import LOCK as DEVELOPMENT
from scripts.experiments.detector_training.detection_consensus import MODELS, bound, classify, features
from scripts.experiments.detector_audits.detection_crop_semantics import VisualSemantics
from scripts.experiments.detector_audits.detection_empty_scene_audit import LOCK as RECOVERY, is_empty, metrics
from scripts.experiments.detector_audits.detection_empty_scene_screened import eligible_frames
from scripts.experiments.detector_audits.detection_exemplar_comparison import TERMS, OpenMMLabDetector, grounder_lock
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, FLOOR
from scripts.experiments.detector_audits.detection_medbin_audit import LOCK as AUDIT, bound as audit_frames, frozen
from scripts.experiments.detector_audits.detection_semantics_audit import apply_rules, crop_frame
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

PLAN = ART/"medbin_companion_preregistration.json"
OUTPUT = ART/"medbin_companion_results.json"
CACHE = DATA/"proposals/medbin-held-out-openmmlab.json"


def get_inputs():
    policy = read_json(PLAN)
    if policy["development_sha256"]!=digest(DEVELOPMENT) or policy["medbin_freeze_sha256"]!=digest(AUDIT):
        raise ValueError("The prospective witness or the independent data changed.")
    choice=policy["gate"]
    if choice["arm"]!="title-wide" or choice["partner"]!="old-pair":
        raise ValueError("The witness collector needs the registered single-label whole/central branch.")
    rows=frozen()
    baseline={k:p["result"] for k,p in audit_frames("baseline")["frames"].items()}
    if set(baseline)!={r["id"] for r in rows}:
        raise ValueError("The frozen audit engine did not finish collecting the full benchmark.")
    return rows,baseline,choice


def collect():
    configure_cpu()
    rows,old,_=get_inputs()
    key=dict(plan_sha256=digest(PLAN),models=grounder_lock()["files"],file_sha256=digest(Path(__file__)))
    saved=read_json(CACHE) if CACHE.exists() else dict(key=key,frames={})
    if saved["key"]!=key:
        raise ValueError("The witness collection differs from its settings.")
    model=None
    for row in rows:
        identifier=row["id"]
        if not is_empty(old[identifier]) or identifier in saved["frames"]:
            continue
        if model is None:
            model=OpenMMLabDetector("xpu")
        picture=source(row)
        proposed=model.run(picture,TERMS[:1])
        edge=min(picture.size)
        x,y=round((picture.width-edge)/2),round((picture.height-edge)/2)
        if x or y:
            window=(x,y,x+edge,y+edge)
            focused=model.run(picture.crop(window),TERMS[:1])
            proposed=suppress([*proposed,*[{**q,"bbox":box} for q in focused
                                 if (box:=project(q["bbox"],window,picture.size)) is not None]],.43)
        saved["frames"][identifier]=proposed
        write_json(CACHE,saved)
        print("MedBin third detector",len(saved["frames"]),flush=True)
    if not CACHE.exists():
        write_json(CACHE,saved)


def measure():
    configure_cpu()
    if OUTPUT.exists():
        raise ValueError("The companion has already been measured.")
    refs,base,choice=get_inputs()
    present=read_json(CACHE)
    expected={k for k,v in base.items() if is_empty(v)}
    if set(present["frames"])!=expected or present["key"]["plan_sha256"]!=digest(PLAN):
        raise ValueError("Run the registered witness on every applicable scene.")
    aux={m:{k:v["result"] for k,v in audit_frames(m)["frames"].items()} for m in MODELS}
    if any(set(y)!=expected for y in aux.values()):
        raise ValueError("Both original witness detectors must have completed.")
    original=read_json(RECOVERY)
    floor=read_json(FLOOR)["cutoff"]
    old,new={},{}
    texts=TextMarginEncoder()
    extractor=None
    for row in refs:
        identifier=row["id"]
        frame=base[identifier]
        picture=source(row)
        if identifier not in expected:
            retained=[p for p in frame["ranked"] if p["score"]>=frame["floor"]]
            results=crop_frame(texts,picture,dict(ranked=retained,floor=frame["floor"]))
            old[identifier]=new[identifier]=apply_rules(frame,results,original["control"])[0]
            continue
        if extractor is None:
            extractor=VisualSemantics()
        supporting={m:out[identifier] for m,out in aux.items()}
        patches,_=eligible_frames(picture,frame,supporting,choice["settings"],extractor)
        h=original["heads"]
        recorded=features(patches,"semantic-best",h["joint"],h["shape"],supporting)
        witness=present["frames"][identifier]
        old[identifier]=frame["ranked"]
        new[identifier]=classify(frame,[r for r in recorded if bound(r[0]["bbox"],witness)>=choice["minimum"]],
                                 choice["settings"])
    result=metrics(refs,old,new,floor)
    qualified=bool(result["guard"]["passed"] and result["guard"]["recovered_matches"] and
                   not result["new_false_alarm_scenes"] and result["candidate"]["fp"]<=result["previous"]["fp"])
    write_json(OUTPUT,dict(passed=qualified,plan_sha256=digest(PLAN),outcome=result))
    print("MedBin corroboration with a third model",qualified,result["previous"],result["candidate"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","measure"))
    (collect if parser.parse_args().action=="collect" else measure)()
