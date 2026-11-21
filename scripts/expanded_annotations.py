"""Validate reviewed scene boxes and publish same-coordinate COCO references."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.expanded_dataset import ART

BOXES = ART / "new_boxes.tsv"
ANNOTATIONS = ART / "annotations.json"


def build():
    inventory = read_json(ART / "inventory.json")["rows"]
    previous = read_json("artifacts/annotations.json")
    boxes = {}
    for line in BOXES.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        frame, *coords = line.split()
        box = [int(c)/100 for c in coords]
        identifier = f"IMG_{frame}"
        if identifier in boxes or len(box) != 4 or not valid_box(box):
            raise ValueError(f"Invalid or repeated annotation for {identifier}.")
        boxes[identifier] = box
    expected = {r["id"] for r in inventory if "previous_sha256" not in r}
    if set(boxes) != expected:
        raise ValueError(f"Unexpected annotation coverage: {set(boxes) ^ expected}")
    labeled = []
    by_split = Counter()
    for row in inventory:
        identifier = row["id"]
        if identifier in expected:
            box = boxes[identifier]
            provenance = "Visible extent independently marked on the oriented JPEG"
        else:
            old = previous[identifier]["objects"]
            if len(old) != 1:
                raise ValueError(f"Box transfer requires one verified re-export: {identifier}.")
            box = old[0]["bbox"]
            provenance = "Draft extent transferred from perceptually matched HEIC"
        flags = ["Frame-edge clipping"] if min(box) == 0 or max(box) == 1 else []
        labeled.append({**row, "objects": [{"bbox": box}], "status": "reviewed draft",
                        "provenance": provenance, "flags": flags})
        by_split[row["split"]] += 1
    write_json(ANNOTATIONS, dict(rows=labeled, counts=dict(by_split),
                                 inventory_sha256=digest(ART / "inventory.json"), boxes_sha256=digest(BOXES)))
    for split in by_split:
        images, annotations = [], []
        for index, row in enumerate((r for r in labeled if r["split"] == split), 1):
            width, height = row["stored_size"]
            if row["orientation"] in {5, 6, 7, 8}:
                width, height = height, width
            images.append(dict(id=index, file_name=row["file"], width=width, height=height))
            left, top, right, bottom = row["objects"][0]["bbox"]
            box = [left*width, top*height, (right-left)*width, (bottom-top)*height]
            annotations.append(dict(id=index, image_id=index, category_id=1, bbox=box,
                                    area=box[2]*box[3], iscrowd=0))
        write_json(ART / f"coco-{split}.json", dict(images=images, annotations=annotations,
                   categories=[dict(id=1, name="syringe")], info={"coordinate_space": "EXIF-oriented JPEGs"}))
    print("Box reference count by partition", dict(by_split), flush=True)
    return labeled


def review(source, rows, target):
    target.mkdir(exist_ok=True, parents=True)
    for start in range(0, len(rows), 25):
        canvas = Image.new("RGB", (1100, 1125), "#f7f7f7")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(rows[start:start+25]):
            with Image.open(source / row["file"]) as original:
                original.draft("RGB", (440, 440))
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((204, 201))
            x, y = index % 5*220 + (220-image.width)//2, index // 5*225
            canvas.paste(image, (x, y))
            left, top, right, bottom = row["objects"][0]["bbox"]
            draw.rectangle((x+left*image.width, y+top*image.height, x+right*image.width-1,
                            y+bottom*image.height-1), outline="#43e9f6", width=2)
            draw.text((index % 5*220+7, y+204), row["id"], fill="#292929")
        canvas.save(target / f"boxes-{start//25+1}.jpg", quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    rows = build()
    if args.review:
        review(args.source, rows, args.review)


if __name__ == "__main__":
    main()
