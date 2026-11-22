"""Restore and verify the frozen public syringe counting reference."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

from medsyringe.data import dhash, digest, read_image, read_json, valid_box
from scripts.experiments.public_data.detection_public_benchmark import request, transformed_hashes


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
IMAGES = DATA / "crowd-next-reference" / "images"
MANIFEST = ROOT / "artifacts" / "detection_hillclimb" / "crowd_next_reference_manifest.json"
MANIFEST_SHA256 = "247fae9fc88ae4dc4598e86f9bd0092c9a4eed71ef904d1100d99a11969251ad"
MAX_BYTES = 24 * 1024 * 1024


def load_reference():
    if digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("The public count reference differs from its frozen manifest.")
    reference = read_json(MANIFEST)
    screen = ROOT / reference["history_screen"]
    if digest(screen) != reference["history_screen_sha256"]:
        raise ValueError("The historical source screen differs from its frozen record.")
    cases = reference["images"]
    audited = read_json(screen)["retained"]
    if set(audited) != {row["id"] for row in cases} or any(any(row.values()) for row in audited.values()):
        raise ValueError("The historical source screen does not clear the complete reference.")
    for field in ("id", "group", "file", "sha256"):
        if len({row[field] for row in cases}) != len(cases):
            raise ValueError(f"Duplicate reference {field}.")
    for row in cases:
        path = (DATA / row["file"]).resolve()
        if path.parent != IMAGES.resolve():
            raise ValueError(f"Unexpected image location for {row['id']}.")
        for field in ("source_url", "asset_url"):
            url = urlsplit(row[field])
            if url.scheme != "https" or not url.hostname:
                raise ValueError(f"Invalid {field} for {row['id']}.")
        if not row["author"] or not row["acquisition"] or not row["license"]:
            raise ValueError(f"Missing provenance for {row['id']}.")
        if any(obj["kind"] != "syringe" or not valid_box(obj["bbox"]) for obj in row["objects"]):
            raise ValueError(f"Malformed object reference for {row['id']}.")
    distribution = Counter(str(len(row["objects"])) for row in cases)
    if distribution != reference["counts"]["count_distribution"]:
        raise ValueError("Scene counts differ from the frozen summary.")
    return cases


def restore(row, path):
    raw = request(row["asset_url"], MAX_BYTES)
    if sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"The public source now serves different bytes for {row['id']}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".part", delete=False) as temp:
        temporary = Path(temp.name)
        try:
            temp.write(raw)
            temp.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def verify(restore_missing=False):
    cases = load_reference()
    variants_seen = set()
    for row in cases:
        path = DATA / row["file"]
        if not path.exists() and restore_missing:
            restore(row, path)
        if digest(path) != row["sha256"]:
            raise ValueError(f"The local bytes changed for {row['id']}.")
        image = read_image(path)
        if image.size != (row["width"], row["height"]) or dhash(image) != row["dhash"]:
            raise ValueError(f"The oriented raster changed for {row['id']}.")
        variants = transformed_hashes(image)
        if any((value ^ other).bit_count() <= 4 for value in variants for other in variants_seen):
            raise ValueError(f"A perceptual duplicate appears in the reference at {row['id']}.")
        variants_seen.add(int(row["dhash"], 16))
    instances = sum(len(row["objects"]) for row in cases)
    print(f"Verified {len(cases)} images across {len(cases)} acquisition groups and {instances} references.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "restore"))
    verify(restore_missing=parser.parse_args().action == "restore")
