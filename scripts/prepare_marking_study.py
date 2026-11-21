"""Prepare separate original-resolution reference or detector crops for OCR."""
import argparse
from pathlib import Path

from PIL import ImageOps

from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, train_rows, write_json
from scripts.precision_study import source
from scripts.ocr_diagnostic import VISIBLE_TOKENS

ART = Path("artifacts")
ROOT = Path("data/ocr-v6")


def prepare(predicted=None):
    records = []
    for row in train_rows(Path(".")):
        identifier = row["id"]
        truth = row["objects"][0]
        box = truth["bbox"]
        if predicted is not None:
            candidates = predicted[identifier]
            if not candidates:
                records.append(dict(id=identifier, scale=truth["scale"], burst=truth["burst"],
                                    capacity_visible=VISIBLE_TOKENS.get(identifier), image=None))
                continue
            box = max(candidates, key=lambda p: p["score"])["bbox"]
        canvas = crop(source(row),box)
        canvas.info.clear()
        canvas = ImageOps.contain(canvas,(1900,1900))
        relative = ROOT/("reference" if predicted is None else "detected")/f"{identifier}.png"
        relative.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(relative,compress_level=1)
        records.append(dict(id=identifier,scale=truth["scale"],burst=truth["burst"],bbox=box,
                            capacity_visible=VISIBLE_TOKENS.get(identifier),image=str(relative),
                            sha256=digest(relative)))
    name = "reference" if predicted is None else "detected"
    write_json(ART/f"marking_{name}_manifest.json",dict(
        scope="Single-session draft scale references. Explicit unit/number visibility was manually confirmed on the separate nine-image subset.",
        files=records))
    print(name,len(records),"images",sum(r["image"] is not None for r in records),"retained crops",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path)
    args=parser.parse_args()
    prepare(read_json(args.predictions) if args.predictions else None)


if __name__=="__main__":
    main()
