"""Preserve embedded color profiles as sidecars to the frozen RGB derivatives."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageCms

from medsyringe.data import digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    files = args.root / "data/original-color"
    files.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for row in read_json(args.root / "artifacts/source_inventory.json"):
        if row["split"] != "train":
            continue
        path = args.source / row["source"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Digest mismatch for {row['source']}")
        with Image.open(path) as image:
            profile = image.info.get("icc_profile")
            if not profile:
                continue
        checksum = sha256(profile).hexdigest()
        (files / f"{checksum}.icc").write_bytes(profile)
        information = ImageCms.getProfileName(ImageCms.getOpenProfile(BytesIO(profile))).strip()
        manifest[row["id"]] = dict(icc_sha256=checksum, profile_name=information)
    write_json(args.root / "artifacts/color_profiles.json", manifest)
    print(len(manifest), "original images with preserved ICC profile sidecars")


if __name__ == "__main__":
    main()
