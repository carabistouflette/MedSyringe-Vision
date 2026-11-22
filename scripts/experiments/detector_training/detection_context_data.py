"""Prepare separate device sources for new development and confirmation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from medsyringe.data import dhash, digest, leakage, read_image, read_json, valid_box, write_json
from scripts.experiments.transfer.detection_device_scene_data import MANIFEST as MULTI
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.detector_audits.detection_medbin_data import MANIFEST as MEDBIN, history
from scripts.experiments.public_data.detection_public_benchmark import near_any, order, request, sheet

POOL = DATA / "device-context-catalog.json"
STAGED = DATA / "device-context-staged.json"
REVIEW = ART / "device_context_reference_review.json"
RELEASE = ART / "device_context_manifest.json"
ASSETS = DATA / "device-context"
PROJECTS = {
    "syringe2-w3svf": dict(owner="juliejoyy", split="device-context-development"),
    "syringe-detection-fnu4f": dict(owner="michaels-workspace-ldsqh", split="device-context-reserve"),
}


def protect(existing, row):
    existing["hashes"].add(row["sha256"])
    existing["difference"].add(int(row["dhash"], 16))


def fetch(slug, row):
    key, url = row["id"], row["url"]
    if not key.isalnum() or urlsplit(url).hostname != "source.roboflow.com":
        return None
    file = ASSETS / f"{key}.jpg"
    try:
        if not file.exists():
            content = request(url, 14 * 1024 * 1024)
            with Image.open(BytesIO(content)) as image:
                if image.format != "JPEG" or image.width * image.height > 30_000_000:
                    return None
                image.verify()
            file.write_bytes(content)
        img = read_image(file)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    config = PROJECTS[slug]
    row = dict(id=key, title=slug, image=str(file.relative_to("data")), sha256=digest(file),
               dhash=dhash(img), dimensions=list(img.size), image_url=url, author=config["owner"],
               source_url=f"https://universe.roboflow.com/{config['owner']}/{slug}",
               group=f"roboflow-context:{slug}", split=config["split"],
               license="https://creativecommons.org/licenses/by/4.0/")
    return row, img


def prepare():
    if STAGED.exists() or RELEASE.exists():
        raise ValueError("The contextual sources have already been staged.")
    ASSETS.mkdir(exist_ok=True)
    known = history()
    for manifest in (MEDBIN, MULTI):
        for row in read_json(manifest)["rows"]:
            protect(known, row)
    staged = []
    for slug, pool in read_json(POOL).items():
        positives = [r for r in pool if r.get("annotations", {}).get("classes", {}).get("syringe")]
        known_positives = {p["id"] for p in positives}
        other = [r for r in pool if r["id"] not in known_positives]
        sample = sorted(positives, key=lambda r: order(r["id"]))[:22]
        sample += sorted(other, key=lambda r: order(r["id"]))[:6]
        selected = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            for result in executor.map(lambda r: fetch(slug, r), sample):
                if result is None:
                    continue
                row, raster = result
                if row["sha256"] in known["hashes"] or near_any(raster, known["difference"]):
                    continue
                protect(known, row)
                selected.append(row)
        sheet(selected, f"device-context-{slug}", 4, 3)
        staged.extend(selected)
        print(slug, len(selected), "native inputs to review", flush=True)
    write_json(STAGED, dict(rows=staged, catalog_sha256=digest(POOL)))


def freeze():
    if RELEASE.exists():
        raise ValueError("This contextual corpus is already sealed.")
    reviewed = read_json(REVIEW)
    if reviewed["candidates_sha256"] != digest(STAGED):
        raise ValueError("The candidates differ from the reviewed set.")
    admitted = []
    for row in read_json(STAGED)["rows"]:
        refs = reviewed["references"].get(row["id"])
        if refs is None:
            continue
        if digest(Path("data") / row["image"]) != row["sha256"] or any(
            not valid_box(box["bbox"]) for box in refs["objects"]
        ):
            raise ValueError(f"Invalid source or box in reviewed image {row['id']}.")
        admitted.append({**row, **refs})
    if len(admitted) != len(reviewed["references"]) or leakage(admitted):
        raise ValueError("The two projects overlap or references point outside the staged set.")
    if any(sum(bool(row["objects"]) and row["split"] == config["split"] for row in admitted) < 6
           for config in PROJECTS.values()):
        raise ValueError("Each of the two source groups needs at least six reviewed positive photographs.")
    write_json(RELEASE, dict(rows=admitted, references_sha256=digest(REVIEW),
                             scope="Separate public sources with unresolved specimen identities. The office "
                                   "capture is development-only. The closeup device source is reserve-only. "
                                   "The listed sources do not overlap the existing collected image hashes."))
    print("Contextual references sealed", len(admitted), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze"))
    (prepare if p.parse_args().action == "prepare" else freeze)()
