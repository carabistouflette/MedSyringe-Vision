"""Generate reviewable instance suggestions without editing the source images."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detection.detectors import suppress
from scripts.experiments.public_data.public_annotation_queue import CHECKPOINT, EDGE, configure_cpu, ranked_region
from scripts.experiments.public_data.public_catalog_exposure_audit import project_links, protected_projects
from scripts.experiments.public_data.public_training_intake import close, raster_hashes, references

CATALOG = Path("data/public-syringe-dataset/manifest.json")
OUT = Path("data/instance-next-batch")
TILE = (245, 220)


def eligible(catalog):
    groups = protected_projects()
    rows = read_json(catalog)["rows"]
    return [r for r in rows if (r.get("positive_review_sha256") or r.get("visual_status") == "positive")
            and not project_links(r, groups)]


def collect(catalog, out):
    from rfdetr import RFDETRNano

    configure_cpu()
    record = read_json(CHECKPOINT)
    weights = record["checkpoint"]
    if digest(weights) != record["checkpoint_sha256"] or record["status"] != "completed":
        raise ValueError("Incomplete or changed proposal checkpoint.")
    binding = dict(catalog=digest(catalog), checkpoint=digest(weights), code=digest(__file__))
    destination = out / "proposals.json"
    current = read_json(destination) if destination.exists() else dict(binding=binding, rows={}, excluded={})
    if current["binding"] != binding:
        raise ValueError("The annotation run inputs have changed.")
    previous, private, visual = references()
    detector = RFDETRNano.from_checkpoint(weights, device="cpu")
    if detector.class_names != ["syringe"] or detector.model_config.resolution != EDGE:
        raise ValueError("The checkpoint must have the expected class and size.")
    if catalog != CATALOG:
        initial = read_json(CATALOG)["rows"]
        previous["hashes"].update(row["sha256"] for row in initial)
        previous["difference"].update(int(row["dhash"], 16) for row in initial)
    for row in eligible(catalog):
        key = row["id"]
        if key in current["rows"] or key in current["excluded"]:
            continue
        asset = Path("data") / row["image"]
        if digest(asset) != row["sha256"]:
            raise ValueError(f"Changed photo {key}.")
        if row["sha256"] in private | previous["hashes"] or close(raster_hashes(asset), visual | previous["difference"]):
            current["excluded"][key] = "Protected raster or visual signature"
            continue
        expected = row.get("annotations", {}).get("classes", row.get("publisher_classes", {})).get("syringe", 0)
        image = read_image(asset)
        boxes = suppress(list(ranked_region(detector, image, EDGE)), .55)[:max(4, expected)]
        current["rows"][key] = dict(group=row["group"], image=row["image"], sha256=row["sha256"],
                                    source_url=row["source_url"], license=row["license"],
                                    expected_from_publisher=expected, proposals=boxes, review="unreviewed")
        if len(current["rows"]) % 20 == 0:
            write_json(destination, current)
            print("Proposed", len(current["rows"]), flush=True)
    write_json(destination, current)
    print("Proposals", len(current["rows"]), "Protected", len(current["excluded"]), flush=True)


def sheets(out):
    data = read_json(out / "proposals.json")
    source = list(data["rows"].items())
    counts = Counter()
    for start in range(0, len(source), 30):
        entries = source[start:start + 30]
        canvas = Image.new("RGB", (TILE[0] * 6, TILE[1] * 5), "#f0f2f4")
        draw = ImageDraw.Draw(canvas)
        placements = []
        for index, (identifier, row) in enumerate(entries):
            image = read_image(Path("data") / row["image"])
            w, h = TILE
            x, y = index % 6 * w, index // 6 * h
            fitted = ImageOps.contain(image, (w - 12, h - 33))
            x0, y0 = x + (w - fitted.width) // 2, y + 3
            canvas.paste(fitted, (x0, y0))
            quantity = min(len(row["proposals"]), max(1, row["expected_from_publisher"]))
            for number, proposal in enumerate(row["proposals"][:quantity], 1):
                l, t, r, b = proposal["bbox"]
                draw.rectangle((x0+l*fitted.width, y0+t*fitted.height, x0+r*fitted.width, y0+b*fitted.height),
                               outline="#e83b58", width=2)
                draw.text((x0+l*fitted.width+2, y0+t*fitted.height+2), str(number), fill="#ff174e")
            draw.text((x+6, y+h-27), f"{start+index+1}  {identifier}", fill="#10243a")
            draw.text((x+6, y+h-13), f"Publisher count {row['expected_from_publisher']}", fill="#39495a")
            placements.append(dict(index=start+index+1, id=identifier, image_xywh=[x0,y0,*fitted.size],
                                   proposed_indices=list(range(quantity))))
            counts[row["group"]] += 1
        canvas.save(out / f"review-{start//30+1:02}.jpg", quality=92)
        write_json(out / f"review-{start//30+1:02}.json", dict(rows=placements))
    print("Sheet rows", dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "sheets"))
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--queue", type=Path, default=OUT)
    options = parser.parse_args()
    if options.action == "collect":
        collect(options.catalog, options.queue)
    else:
        sheets(options.queue)
