"""Check end-to-end local detection and timing on a specified packaged runtime."""
import argparse
from pathlib import Path
from subprocess import check_output
from time import perf_counter

from medsyringe.data import digest, read_image, write_json
from medsyringe.grounding_backend import analyze, enable_intel
from medsyringe.pipeline import MedSyringeEngine
from scripts.sam3_study import configure_cpu


def probe(args):
    configure_cpu()
    beginning = perf_counter()
    engine = MedSyringeEngine(args.root)
    revision = check_output(["git","-C",str(args.root),"rev-parse","HEAD"],text=True).strip()
    runs=[]
    try:
        if args.device=="xpu":
            enable_intel(engine,args.root)
            if engine.identity.get("primary_backend")!="xpu":
                raise ValueError("The packaged Intel path did not activate.")
        else:
            for worker in (engine.sam3,engine.expanded.worker,
                           engine.crop_verifier.support_worker if engine.crop_verifier else None):
                if worker is not None:
                    worker.options = {**worker.options,"device":"cpu"}
        setup=perf_counter()-beginning
        images=[(p,read_image(p),count,digest(p)) for p,count in zip(args.image,args.expected,strict=True)]
        for iteration in range(args.repeats):
            for path,picture,expected,checksum in images:
                start=perf_counter()
                result=analyze(engine,picture,args.ocr)
                check=result["detection_checks"]
                case=dict(source=path.name,sha256=checksum,expected=expected,actual=result["accepted_count"],
                          boxes=[p["bbox"] for p in result["objects"]],seconds=perf_counter()-start,
                          discovery_round=iteration+1,appearance_release=result["model"].get("appearance_detection_sha256"),
                          verification=check,passed=result["accepted_count"]==expected and check["status"]=="available")
                runs.append(case)
                write_json(args.output,dict(completed=False,revision=revision,device=args.device,runs=runs,load_seconds=setup))
                print(args.device,path.name,case["actual"],round(case["seconds"],2),check["status"],flush=True)
    finally:
        engine.close()
    write_json(args.output,dict(completed=True,passed=all(r["passed"] for r in runs),revision=revision,
                               device=args.device,native_assets=True,ocr=args.ocr,runs=runs,load_seconds=setup))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--device",choices=("cpu","xpu"),required=True)
    parser.add_argument("--image",action="append",type=Path,required=True)
    parser.add_argument("--expected",action="append",type=int,required=True)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--ocr",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if len(args.image)!=len(args.expected) or args.repeats<1:
        parser.error("Supply one expected count per image and a positive repetition count.")
    probe(args)
