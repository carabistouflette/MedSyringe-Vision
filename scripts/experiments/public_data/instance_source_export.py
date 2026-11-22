"""Export the reviewed public syringe pilot as repository-relative COCO detection data."""
from collections import Counter

from medsyringe.data import coco, digest, read_image, read_json, write_json
from scripts.experiments.public_data.detection_public_benchmark import transformed_hashes
from scripts.experiments.public_data.instance_source_audit import ART, ROOT, fingerprint_collision, protection
from scripts.experiments.public_data.instance_source_supplement import MANIFEST

ORIGINAL = ROOT / "origin-license-verified.json"
OUTPUT = ROOT / "reviewed-annotations.json"
COCO = ROOT / "reviewed-coco.json"
SUMMARY = ART / "instance_source_summary.json"


def run():
    manual = read_json(MANIFEST)
    publisher = read_json(ORIGINAL)["rows"]
    rows = [
        dict(row, annotation_origin="local visual review")
        for row in (*manual["rows"], *manual["negative_rows"])
    ]
    rows += [
        dict(row, objects=row["annotations"]["boxes"], annotation_origin="Open Images publisher box")
        for row in publisher
    ]
    rows.sort(key=lambda row: row["id"])
    known, private_sha, private_dhash = protection()
    ids, hashes, fingerprints = set(), set(), set()
    for row in rows:
        identifier, path = row["id"], row["image"]
        image = read_image(path)
        if (identifier in ids or digest(path) != row["sha256"] or row["sha256"] in hashes
                or row["sha256"] in private_sha | known["sha256"] or list(image.size) != row["dimensions"]):
            raise ValueError(f"Photo identity or byte collision in {identifier}")
        current = transformed_hashes(image)
        if fingerprint_collision(current, private_dhash | known["dhash"] | fingerprints):
            raise ValueError(f"Perceptual collision in {identifier}")
        row["width"], row["height"] = image.size
        ids.add(identifier)
        hashes.add(row["sha256"])
        fingerprints.update(current)

    write_json(OUTPUT, dict(rows=rows, independent_human_review=False,
                            coordinates="orientation-corrected normalized xyxy",
                            source_manifests={str(path): digest(path) for path in (MANIFEST, ORIGINAL)}))
    write_json(COCO, coco(rows))
    positives = [row for row in rows if row["objects"]]
    write_json(SUMMARY, dict(
        target_positive_photos=500, positive_photos=len(positives),
        verified_empty_photos=len(rows) - len(positives),
        syringe_instances=sum(len(row["objects"]) for row in rows),
        positive_acquisition_groups=len({row["group"] for row in positives}),
        positive_annotation_origins=dict(Counter(row["annotation_origin"] for row in positives)),
        multi_syringe_positive_photos=sum(len(row["objects"]) > 1 for row in positives),
        independent_human_approved=0,
        artifacts={str(path): digest(path) for path in (OUTPUT, COCO)}))
    print(read_json(SUMMARY), flush=True)


if __name__ == "__main__":
    run()
