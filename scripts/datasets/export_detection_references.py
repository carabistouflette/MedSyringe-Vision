"""Export validated public references in separate COCO partitions."""
import argparse
from pathlib import Path

from medsyringe.data import coco, leakage, read_json, write_json
from scripts.experiments.detector_training.detection_hillclimb_corpus import ART

MANIFESTS = ("public_manifest.json", "public_additional_fit.json", "openimages_additional_manifest.json",
             "phlebotomy.json", "dental_manifest.json")


def export(directory):
    for manifest in MANIFESTS:
        rows = read_json(ART/manifest)["rows"]
        if leakage(rows):
            raise ValueError(f"Cross-partition overlap in {manifest}.")
        for partition in sorted({row["split"] for row in rows}):
            subset = [row for row in rows if row["split"]==partition]
            result = coco(subset)
            if len(result["images"])!=len(subset) or len(result["annotations"])!=sum(len(r["objects"]) for r in subset):
                raise ValueError(f"COCO export changed the reference count in {partition}.")
            write_json(directory/(Path(manifest).stem+"-"+partition+".json"),result)
            print(partition,len(result["images"]),len(result["annotations"]),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    export(parser.parse_args().output)
