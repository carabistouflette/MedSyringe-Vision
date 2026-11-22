"""Prepare local COCO proposals for review of public syringe photographs."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from medsyringe.data import coco, digest, read_image, read_json, write_json
from medsyringe.detection.detectors import suppress
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART
from scripts.experiments.public_data.public_augmentation_ablation import EDGE
from scripts.experiments.public_data.public_training_intake import close, raster_hashes, references
from scripts.experiments.transfer.rfdetr_compare import ranked_region
from scripts.experiments.expanded.sam3_study import configure_cpu

CHECKPOINT = ART / "public_augmentation_capture.json"


def make_queue(source, destination, limit):
    from rfdetr import RFDETRNano

    checkpoint = read_json(CHECKPOINT)
    weights = Path(checkpoint["checkpoint"])
    if checkpoint["status"] != "completed" or digest(weights) != checkpoint["checkpoint_sha256"]:
        raise ValueError("Public-only proposal checkpoint is missing or changed.")
    if destination.exists():
        raise FileExistsError("Choose a new path for this annotation batch.")
    configure_cpu()
    model = RFDETRNano.from_checkpoint(str(weights), device="cpu")
    if model.class_names != ["syringe"] or model.model_config.resolution != EDGE:
        raise ValueError("Unexpected proposal model classes or resolution.")
    earlier, protected, private_difference = references()
    digest_source = digest(source)
    order = sorted(read_json(source)["rows"], key=lambda row: (
        row["group"], sha256(("local-review-v1/" + row["id"]).encode()).hexdigest()))
    if limit:
        # Round-robin the projects so a pilot covers different conditions and backgrounds.
        by_group = {}
        for row in order:
            by_group.setdefault(row["group"], []).append(row)
        from itertools import zip_longest

        order = [row for step in zip_longest(*by_group.values()) for row in step if row is not None][:limit]
    if not order or any(not row.get("positive_review_sha256") for row in order):
        raise ValueError("The proposal batch requires public presence-reviewed input.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        archive = Path(temporary) / destination.name
        records = []
        with ZipFile(archive, "w") as package:
            for row in order:
                asset = Path("data") / row["image"]
                if digest(asset) != row["sha256"] or row["sha256"] in protected | earlier["hashes"]:
                    raise ValueError(f"Changed or protected image {row['id']}.")
                hashes = raster_hashes(asset)
                if close(hashes, private_difference) or close(hashes, earlier["difference"]):
                    raise ValueError(f"Possible protected image {row['id']}.")
                image = read_image(asset)
                proposals = suppress(list(ranked_region(model, image, EDGE)), .55)[:5]
                exported = f"{row['id']}.png"
                with BytesIO() as payload:
                    image.save(payload, format="PNG", compress_level=2)
                    package.writestr(f"images/{exported}", payload.getvalue())
                records.append(dict(id=row["id"], image=exported, group=row["group"],
                                    source_url=row["source_url"], source_sha256=row["sha256"],
                                    width=image.width, height=image.height, objects=proposals,
                                    review_required=True))
            coco_rows = coco(records)
            coco_rows["info"] = dict(description="Unreviewed RF-DETR proposals. Review every instance.")
            data_file = Path(temporary) / "instances_default.json"
            write_json(data_file, coco_rows)
            package.write(data_file, "annotations/instances_default.json")
            ledger = Path(temporary) / "proposal_ledger.json"
            write_json(ledger, dict(source_sha256=digest_source, checkpoint_sha256=digest(weights),
                                    rows=records, empty_proposals_require_review=True))
        archive.replace(destination)
        ledger.replace(destination.with_suffix(".json"))
    print("Prepared", len(records), "public review images and",
          sum(len(row["objects"]) for row in records), "unreviewed proposals", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("The review limit must be nonnegative.")
    make_queue(args.source, args.output, args.limit)
