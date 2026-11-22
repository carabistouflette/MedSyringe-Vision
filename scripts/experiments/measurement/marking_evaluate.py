"""Evaluate selective OCR on burst-disjoint crops and inverted controls."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
from pathlib import Path

from medsyringe.measurement.capacity_ocr import resolve
from medsyringe.data import digest, read_json, write_json

ART=Path("artifacts")
DATA=Path("data/ocr-v6")
SELECTION=ART/"capacity_ocr_selection.json"
CONFIGS=("quarter_turns","rectified","targeted","guided")


def records(kind,variant="",guided=False):
    name=f"markings-{kind}{variant}{'-guided' if guided else ''}.json"
    path=DATA/name
    return read_json(path)["frames"] if path.exists() else {}


def read_case(key,original,extra,options):
    tokens=original.get(key,{}).get("tokens",[])
    mode=options["mode"]
    if mode=="guided":
        tokens=[*tokens,*extra.get(key,{}).get("tokens",[])]
    elif mode in ("rectified","quarter_turns"):
        tokens=[t for t in tokens if not t.get("source") and
                (mode!="quarter_turns" or t["family"]=="raw")]
    return resolve(tokens,cutoff=options["cutoff"],repeats=options["repeats"],
                   scale_cutoff=options["scale_cutoff"])


def stats(rows,original,extra,options):
    findings={}
    caps,types=Counter(),Counter()
    for row in rows:
        identifier=row["id"]
        response=read_case(identifier,original,extra,options)
        capacity=response["capacity"]
        scale=response["scale"]
        target=int(row["scale"].split("-")[0]) if row.get("scale") else None
        caps["answered"]+=capacity is not None
        caps["correct"]+=bool(capacity and capacity["value"]==target)
        types["answered"]+=scale is not None
        types["correct"]+=bool(scale and scale["value"]==row["scale"])
        findings[identifier]=dict(capacity=capacity,scale=scale,status=response["status"],
                                  expected_capacity=target,expected_scale=row["scale"])
    caps["wrong"]=caps["answered"]-caps["correct"]
    types["wrong"]=types["answered"]-types["correct"]
    return dict(n=len(rows),capacity=dict(caps),type=dict(types),findings=findings)


def cohorts(rows):
    members=defaultdict(list)
    for row in rows:
        members[row["burst"]].append(row)
    chosen=[]
    for label in ("20-scale","60-scale"):
        groups=sorted((k for k,v in members.items() if all(r["scale"]==label for r in v)),
                      key=lambda k:sha256(f"607:{k}".encode()).digest())
        chosen.extend(groups[:max(1,len(groups)//3)])
    return [r for r in rows if r["burst"] not in chosen],[r for r in rows if r["burst"] in chosen]


def evaluate_controls(rows,options):
    targets=[r for r in rows if r.get("capacity_visible")]
    out={}
    for name,variant in (("tiny",""),("tiny_inverted","-tiny-180"),("medium","-medium-0")):
        original=records("reference",variant)
        extra=records("reference",variant,guided=True)
        if not all(r["id"] in original for r in targets):
            continue
        backend=options if name!="medium" or extra else {**options,"mode":"targeted"}
        result=stats(targets,original,extra,backend)
        result["visible_numeric_hits"]=sum(any(t["text"]==r["capacity_visible"]
             for t in original[r["id"]]["tokens"]+extra.get(r["id"],{}).get("tokens",[])) for r in targets)
        out[name]=result
    return out


def select():
    manifest=ART/"marking_reference_manifest.json"
    rows=read_json(manifest)["files"]
    original,extra=records("reference"),records("reference",guided=True)
    if len(rows)!=len(original) or len(rows)!=len(extra):
        raise ValueError("The complete reference readings are required.")
    dev,test=cohorts(rows)
    trials=[]
    for mode,threshold,repeats,scale in product(CONFIGS,(.74,.80,.86,.92),(1,2),(.80,.88)):
        settings=dict(mode=mode,cutoff=threshold,repeats=repeats,scale_cutoff=scale)
        result=stats(dev,original,extra,settings)
        trials.append(dict(options=settings,capacity=result["capacity"],type=result["type"]))
    eligible=[r for r in trials if not r["capacity"]["wrong"] and not r["type"]["wrong"]
              and r["capacity"]["answered"] and r["type"]["answered"]]
    if not eligible:
        raise ValueError("No error-free supported classification was found on development crops.")
    final=max(eligible,key=lambda r:(r["capacity"]["answered"]+r["type"]["answered"],
                                    r["capacity"]["answered"], r["options"]["cutoff"]))
    lock=dict(options=final["options"],calibration_ids=[r["id"] for r in dev],
              heldout_ids=[r["id"] for r in test],selection=final,trials=trials,
              reference_manifest_sha256=digest(manifest),reader_sha256=digest("medsyringe/measurement/capacity_ocr.py"),
              refiner_sha256=digest("medsyringe/measurement/ocr_refine.py"),
              scope="Burst-disjoint development and verification, all from a single acquisition session.")
    write_json(SELECTION,lock)
    truth=stats(test,original,extra,lock["options"])
    write_json(ART/"capacity_ocr_reference_results.json",dict(
        heldout=truth,all=stats(rows,original,extra,lock["options"]),
        orientation_controls=evaluate_controls(rows,lock["options"]),selection_sha256=digest(SELECTION)))
    print("Selected",final,"heldout",truth["n"],truth["capacity"],truth["type"],flush=True)


def detected():
    config=read_json(SELECTION)
    rows=read_json(ART/"marking_detected_manifest.json")["files"]
    original,extra=records("detected"),records("detected",guided=True)
    if any(r["image"] and (r["id"] not in original or r["id"] not in extra) for r in rows):
        raise ValueError("The complete localization-to-OCR evaluation is required.")
    final=stats(rows,original,extra,config["options"])
    heldout=stats([r for r in rows if r["id"] in config["heldout_ids"]],original,extra,config["options"])
    write_json(ART/"capacity_ocr_detected_results.json",dict(
        all=final,heldout=heldout,selection_sha256=digest(SELECTION)))
    print("Detected",final["n"],final["capacity"],final["type"],"heldout",heldout["n"],
          heldout["capacity"],heldout["type"],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select","detected"))
    args=parser.parse_args()
    (select if args.stage=="select" else detected)()


if __name__=="__main__":
    main()
