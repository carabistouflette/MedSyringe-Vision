"""Check selected same-session draft box disagreements without rewriting references."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.detection_hillclimb_experiment import ART, FLOOR
from scripts.expanded_detection_study import load, short

REVIEWED = {
    "IMG_3284.jpeg": ([.206,.231,.631,.823], "The draft included substantial surrounding hand and background."),
    "IMG_3212.jpeg": ([.070,.424,.867,.766], "The draft was vertically displaced from the syringe."),
    "IMG_3236.jpeg": ([.411,.235,.517,.780], "The draft included fingers outside the visible transparent piston."),
    "IMG_3242.jpeg": ([.141,.544,1.,.716], "The draft extended above and below the horizontally clipped syringe."),
    "IMG_3190.jpeg": ([.429,.456,.517,.765], "The draft included the surrounding hand below the plunger."),
}


def audit(source):
    floor = read_json(FLOOR)["cutoff"]
    reports = {}
    entries = []
    for split in ("fit", "calibration", "test"):
        rows,frames = load(split)
        capture = [row for row in rows if "file" in row]
        before = {r["id"]:frames[r["id"]]["prior"] for r in capture}
        amended = []
        for row in capture:
            path = row["file"]
            if path not in REVIEWED:
                amended.append(row)
                continue
            image_hash = digest(source / path)
            if image_hash != row["sha256"] or len(row["objects"]) != 1:
                raise ValueError(f"The manually inspected image or its reference changed: {path}")
            exact,reason = REVIEWED[path]
            if not valid_box(exact):
                raise ValueError(f"Invalid reviewed reference for {path}")
            original = row["objects"][0]
            accepted = [p for p in before[row["id"]] if p["score"]>=floor]
            best = max(accepted,key=lambda p:box_iou(p["bbox"],exact))
            entries.append(dict(id=row["id"],file=path,sha256=image_hash,partition=split,
                                draft=original["bbox"],alternative=exact,reason=reason,model_box=best["bbox"],
                                draft_iou=box_iou(original["bbox"],best["bbox"]),
                                alternative_iou=box_iou(exact,best["bbox"])))
            amended.append({**row,"objects":[{**original,"bbox":exact}]})
        reports[split] = dict(draft=short(evaluate(capture,before,floor)),
                              alternative=short(evaluate(amended,before,floor)))
    if {e["file"] for e in entries} != set(REVIEWED):
        raise ValueError("The targeted visual check did not resolve every selected image.")
    write_json(ART/"fil_rouge_box_review.json",dict(entries=entries,results=reports,
               description="Post-hoc sensitivity check on selected localization disagreements. Five sources were inspected "
                           "at full size with an independent coordinate grid and their original bytes verified. A prior "
                           "model replay selected the cases. No dataset or model bytes changed. No claim of blind annotation, "
                           "independent validation or complete reference review is made.",
               reference_sha256=digest("artifacts/expanded/annotations.json"),code_sha256=digest(__file__)))
    print({name:{k:(v["tp"],v["fp"],v["fn"]) for k,v in record.items()} for name,record in reports.items()},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    audit(parser.parse_args().source)
