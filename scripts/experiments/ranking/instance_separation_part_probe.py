"""Inspect named plunger proposals on the already exposed six-syringe scene."""
from math import ceil, floor

from medsyringe.detection.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.measurement.part_geometry import containment
from scripts.experiments.ranking.instance_separation import ART, RESULT
from scripts.experiments.occlusion.occlusion_detector import raster
from scripts.experiments.parts.part_verifier_study import infer
from scripts.experiments.relocalization.relocalization_reserve import evidence
from scripts.experiments.transfer.rfdetr_verifier import expanded
from scripts.experiments.expanded.sam3_study import LOCK, Segmenter, configure_cpu

OUT = ART / "instance_separation_plunger_probe.json"


def run():
    rows, incumbents, _, _ = evidence()
    row = next(r for r in rows if r["id"] == "refine-8670507")
    key = row["id"]
    base = [p["bbox"] for p in incumbents[key]]
    targets = read_json(RESULT)["six_syringes"]
    image = raster(row)
    x, y, r, b = expanded([min(p[0] for p in base), min(p[1] for p in base),
                           max(p[2] for p in base), max(p[3] for p in base)], .25)
    view = (floor(x*image.width), floor(y*image.height), ceil(r*image.width), ceil(b*image.height))
    configure_cpu()
    sam = Segmenter(756, "xpu")
    observations = infer(sam, image.crop(view))
    detections = []
    for item in observations["rod"]:
        box = project(item["bbox"], view, image.size, edge=0.)
        detections.append(dict(bbox=box, score=item["score"],
                               incumbent=[containment(box, p) for p in base],
                               candidates=[containment(box, t["candidate"]["bbox"]) for t in targets]))
    write_json(OUT, dict(code=digest(__file__), checkpoint=digest(LOCK), evaluation=digest(RESULT),
                         photo=key, view=view, rods=detections,
                         scope="Post-evaluation probe. Containment is measured without a learned or qualified rule."))
    print([(round(p["score"], 3), [round(x, 2) for x in p["candidates"]],
            [round(x, 2) for x in p["incumbent"]]) for p in detections], flush=True)


if __name__ == "__main__":
    run()
