"""Probe named part evidence on the observed replacement failure and a lookalike."""
from math import ceil, floor

from medsyringe.detection.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.measurement.part_geometry import PARTS, containment
from scripts.experiments.parts.part_verifier_study import infer
from scripts.experiments.relocalization.relocalization_reserve import evidence, raster
from scripts.experiments.relocalization.relocalization_selector import ART, RESULT
from scripts.experiments.transfer.rfdetr_verifier import expanded
from scripts.experiments.expanded.sam3_study import LOCK, Segmenter, configure_cpu

OUTPUT = ART / "relocalization_part_probe.json"
FIELD = "refine-9905634"
NEGATIVE = "refine-37499105"


def run():
    rows, baseline, _, _ = evidence()
    report = read_json(RESULT)
    change, = report["cohorts"]["reserve"]["arms"]["learned-views"]["changes"][FIELD]
    configure_cpu()
    segmenter = Segmenter(756, "xpu")
    found = {}
    for row in rows:
        key = row["id"]
        if key not in (FIELD, NEGATIVE):
            continue
        incumbent, = baseline[key]
        photo = raster(row)
        a, b, c, d = expanded(incumbent["bbox"], .25)
        window = (floor(a * photo.width), floor(b * photo.height), ceil(c * photo.width), ceil(d * photo.height))
        local = infer(segmenter, photo.crop(window))
        support = {}
        for label in PARTS:
            support[label] = []
            for item in local[label][:5]:
                box = project(item["bbox"], window, photo.size, edge=0.)
                support[label].append(dict(bbox=box, score=item["score"],
                    incumbent_coverage=containment(box, incumbent["bbox"]),
                    replacement_coverage=containment(box, change["bbox"]) if key == FIELD else None))
        reference = row["objects"][0]["bbox"] if key == FIELD else None
        found[key] = dict(parts=support, window=window, incumbent=incumbent["bbox"],
                          replacement=change["bbox"] if key == FIELD else None,
                          incumbent_iou=box_iou(reference, incumbent["bbox"]) if reference else None,
                          replacement_iou=box_iou(reference, change["bbox"]) if reference else None)
        print(key, {label: [(round(p["score"], 3), round(p["incumbent_coverage"], 3),
                             p["replacement_coverage"]) for p in values[:2]]
                    for label, values in support.items()}, flush=True)
    write_json(OUTPUT, dict(model=digest(LOCK), code=digest(__file__), evaluation=digest(RESULT), evidence=found,
                            scope="Post-evaluation diagnostic on two known images. No rule is tuned or qualified."))


if __name__ == "__main__":
    run()
