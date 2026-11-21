"""Package the released precision study's exact public rasters and replay its metrics."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.appearance_runtime_parity import old_records
from scripts.appearance_worker import explain
from scripts.detection_hillclimb_experiment import ART, FLOOR
from scripts.detection_hillclimb_rank import minimal
from scripts.detection_precision_confirmation import CAPTURE, RESULT as METRICS
from scripts.detection_precision_negative_audit import RESULT as NEGATIVE_METRICS
from scripts.expanded_detection_study import short
from scripts.runtime_bundle import confined, copy_bytes, git_revision

PUBLIC = ("fresh","commons-regression","openimages","lookalikes","independent-negatives")
MANIFEST = "precision-evidence.json"


def expected_metrics():
    result = read_json(METRICS)["outcomes"]
    result["independent-negatives"] = read_json(NEGATIVE_METRICS)
    return result


def pack(destination,include_capture):
    if destination.exists():
        raise FileExistsError(destination)
    release = read_json(RELEASE)
    expected = expected_metrics()
    floor = read_json(FLOOR)["cutoff"]
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=destination.parent,prefix="medsyringe-evidence-") as work:
        staging = Path(work)/"evidence"
        staging.mkdir()
        roles = {}
        for name in (*PUBLIC,*CAPTURE) if include_capture else PUBLIC:
            rows,before,after = old_records(name)
            assets = {}
            if name not in CAPTURE:
                for row in rows:
                    local = Path("data")/confined(row["image"])
                    checksum = row.get("derived_sha256", row["sha256"])
                    if digest(local)!=checksum:
                        raise ValueError(f"Public raster {row['id']} differs from its benchmark version.")
                    if not row.get("license") or not row.get("source_url"):
                        raise ValueError(f"Public raster {row['id']} lacks attribution metadata.")
                    assets[str(local)] = checksum
                    if not (staging/local).exists():
                        copy_bytes(local,staging/local)
            relative = Path("cohorts")/f"{name}.json"
            write_json(staging/relative,dict(
                rows=rows, raster_sha256=assets, floor=floor,
                previous={k:v["ranked"] for k,v in before.items()},candidate=after,
                expected={k:expected[name][k] for k in ("previous","candidate")},
            ))
            roles[name] = dict(path=str(relative),sha256=digest(staging/relative))
        evidence = {}
        for path in (RELEASE,METRICS,NEGATIVE_METRICS,ART/"precision_negative_manifest.json",
                     ART/"consensus_audit_manifest.json"):
            copy_bytes(path,staging/path)
            evidence[str(path)] = digest(path)
        write_json(staging/MANIFEST,dict(
            git_revision=git_revision(Path.cwd()), release=release["id"], cohorts=roles, evidence=evidence,
            scope="Portable replay of released precision metrics from stored ranked inference. The public raster "
                  "bytes and per-image attribution are included. Optional capture metadata contains only labels and "
                  "ranked boxes. It contains no original private photographs or observation database."))
        replay(staging)
        staging.rename(destination)
    print("Evidence archive created",destination,flush=True)


def replay(bundle, originals=None):
    manifest = read_json(bundle/MANIFEST)
    for path,checksum in manifest["evidence"].items():
        if digest(bundle/confined(path))!=checksum:
            raise ValueError(f"Changed release evidence at {path}.")
    findings = {}
    for name,record in manifest["cohorts"].items():
        path = bundle/confined(record["path"])
        if digest(path)!=record["sha256"]:
            raise ValueError(f"Changed cohort {name}.")
        cohort = read_json(path)
        rows = cohort["rows"]
        if len({x["id"] for x in rows})!=len(rows) or leakage(rows):
            raise ValueError(f"Broken reference integrity in {name}.")
        for raster,check in cohort["raster_sha256"].items():
            if digest(bundle/confined(raster))!=check:
                raise ValueError(f"Changed source raster {raster}.")
        if originals is not None and name in CAPTURE:
            for row in rows:
                if digest(originals/confined(row["file"]))!=row["sha256"]:
                    raise ValueError(f"Changed original capture file {row['file']}.")
        computed = {k:evaluate(rows,cohort[k],cohort["floor"]) for k in ("previous","candidate")}
        summary = {k:short(value) for k,value in computed.items()}
        if summary!=cohort["expected"]:
            raise ValueError(f"Metrics from {name} differ from the released records.")
        check = compare(computed["previous"],computed["candidate"])
        if not check["passed"]:
            raise ValueError(f"Instance retention failed in {name}.")
        reach = None
        if name in PUBLIC and summary["candidate"]["objects"]:
            floor = cohort["floor"]
            reviewed = {key:explain(dict(ranked=ranked,floor=floor),cohort["candidate"][key])
                        for key,ranked in cohort["previous"].items()}
            reach = {}
            for limit in (2,5):
                accepted = {}
                for key,ranked in reviewed.items():
                    chosen = [p for p in ranked if p["score"]>=floor]
                    pending = [p for p in ranked if p["score"]<floor][:limit]
                    accepted[key] = [{**p,"score":1.0} for p in (*chosen,*pending)]
                reach[f"top_{limit}"] = minimal(rows,accepted,1.0)["tp"]
        findings[name] = dict(metrics=summary,instance_guard=check,review_reach=reach)
        print(name,"match",summary["candidate"]["tp"],"TP",summary["candidate"]["fp"],"FP",flush=True)
    return findings


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("pack","replay"))
    parser.add_argument("directory",type=Path)
    parser.add_argument("--include-capture-metadata",action="store_true")
    parser.add_argument("--original-capture",type=Path)
    args = parser.parse_args()
    if args.action=="pack":
        pack(args.directory.resolve(),args.include_capture_metadata)
    else:
        replay(args.directory.resolve(),args.original_capture)
