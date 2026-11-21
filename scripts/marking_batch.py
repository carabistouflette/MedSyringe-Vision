"""Run PP-OCRv6 locally and cache unfiltered orientation readings."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

from medsyringe.capacity_ocr import DETECTOR, MODEL_ROOT, RECOGNIZER, SceneOCR
from medsyringe.ocr_refine import guided_read

ART=Path("artifacts")
DATA=Path("data/ocr-v6")
MODELS=(DETECTOR,RECOGNIZER)


def digest(path):
    with Path(path).open("rb") as file:
        return file_digest(file,"sha256").hexdigest()


def save(path,data):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2)+"\n")
    temp.replace(path)


def model_hashes(backend="tiny"):
    names = [model.replace("tiny",backend) for model in MODELS]
    return {str(p.relative_to(MODEL_ROOT)):digest(p) for model in names
            for p in sorted((MODEL_ROOT/model).glob("*")) if p.is_file()}


def create_reader(backend):
    if backend=="tiny":
        return SceneOCR()
    from paddleocr import PaddleOCR, TextRecognition

    reader=SceneOCR.__new__(SceneOCR)
    reader.model_root=MODEL_ROOT
    det,rec=(name.replace("tiny","medium") for name in MODELS)
    options=dict(device="cpu",cpu_threads=2,enable_mkldnn=False)
    reader.ocr=PaddleOCR(text_detection_model_name=det,text_detection_model_dir=str(MODEL_ROOT/det),
                         text_recognition_model_name=rec,text_recognition_model_dir=str(MODEL_ROOT/rec),
                         use_doc_orientation_classify=False,use_doc_unwarping=False,
                         use_textline_orientation=False,**options)
    reader.recognizer=TextRecognition(model_name=rec,model_dir=str(MODEL_ROOT/rec),**options)
    return reader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("reference","detected"))
    parser.add_argument("--only-visible",action="store_true")
    parser.add_argument("--backend",choices=("tiny","medium"),default="tiny")
    parser.add_argument("--rotate",type=int,choices=(0,180),default=0)
    parser.add_argument("--guided",action="store_true")
    args=parser.parse_args()
    manifest=ART/f"marking_{args.cohort}_manifest.json"
    rows=json.loads(manifest.read_text())["files"]
    if args.only_visible:
        rows=[r for r in rows if r.get("capacity_visible")]
    variant = f"-{args.backend}-{args.rotate}" if args.backend!="tiny" or args.rotate else ""
    output=DATA/f"markings-{args.cohort}{variant}.json"
    raw_output=output
    if args.guided:
        output=output.with_name(output.stem+"-guided.json")
    key=dict(manifest=digest(manifest),models=model_hashes(args.backend),
             reader_sha256=digest("medsyringe/capacity_ocr.py"))
    if args.guided:
        key.update(refiner_sha256=digest("medsyringe/ocr_refine.py"),raw=digest(raw_output))
        original=json.loads(raw_output.read_text())["frames"]
    old=json.loads(output.read_text()) if output.exists() else {}
    saved=old["frames"] if old.get("key")==key else {}
    reader=None
    for index,row in enumerate(rows,1):
        identifier=row["id"]
        if identifier in saved or not row["image"]:
            continue
        if digest(row["image"])!=row["sha256"]:
            raise ValueError("The source crop changed.")
        if reader is None:
            reader=create_reader(args.backend)
        started=perf_counter()
        with Image.open(row["image"]) as image:
            oriented=image if args.rotate==0 else ImageOps.flip(ImageOps.mirror(image))
            if args.guided:
                tokens=guided_read(reader,oriented,original[identifier]["tokens"])
            else:
                tokens=reader.read(oriented)
        saved[identifier]=dict(tokens=tokens,seconds=perf_counter()-started)
        save(output,dict(key=key,frames=saved))
        print(args.cohort,index,identifier,round(saved[identifier]["seconds"],2),
              "tokens",[(t["text"],round(t["score"],2)) for t in tokens if t["score"]>.7][:25],
              flush=True)


if __name__=="__main__":
    main()
