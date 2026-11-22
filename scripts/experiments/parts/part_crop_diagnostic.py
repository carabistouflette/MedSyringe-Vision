"""Measure the effect of giving the part prompts a manually identified syringe crop."""
from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.measurement.part_geometry import PARTS, PROMPTS
from scripts.experiments.crowds.low_count_candidate_diagnostic import MINIMUM, infer
from scripts.experiments.parts.part_verifier_report import integer_counts
from scripts.experiments.parts.part_verifier_study import (
    FREEZE, WORK, Segmenter, configure_cpu, crowd, landmark_audit, load, records, source,
)

OUTPUT = crowd.ART / "part_crop_diagnostic.json"
CACHE = WORK / "part-crop-landmarks.json"


def original_coordinates(box, region):
    l, t, r, b = region
    return [l + box[0]*(r-l), t + box[1]*(b-t), l + box[2]*(r-l), t + box[3]*(b-t)]


def run():
    load()
    configure_cpu()
    signature = dict(model=digest(FREEZE), code=digest(__file__))
    cache = read_json(CACHE) if CACHE.exists() else dict(signature=signature, predictions={})
    if cache["signature"] != signature:
        raise ValueError("The crop diagnostic or its sealed model changed.")
    segmenter = None
    summaries = {}
    for role in ("fit", "transfer"):
        annotated = [r for r in source(role) if any(o.get("parts") for o in r["objects"])]
        archived = records(role)
        full = {r["id"]: dict(prompt_parts={p: [item for item in archived[r["id"]]["prompt_parts"][p]
                                                if item["score"] >= MINIMUM] for p in PARTS})
                for r in annotated}
        local = cache["predictions"].setdefault(role, {})
        for row in annotated:
            if row["id"] in local:
                continue
            if segmenter is None:
                segmenter = Segmenter(756, "xpu")
            obj = next(o for o in row["objects"] if o.get("parts"))
            image = crop(read_image(row["image"]), obj["bbox"])
            evidence = infer(segmenter, image, tuple(PROMPTS.values()))
            local[row["id"]] = dict(prompt_parts={
                key: [{**p, "bbox": original_coordinates(p["bbox"], obj["bbox"])} for p in evidence[PROMPTS[key]]]
                for key in PARTS})
            write_json(CACHE, cache)
            print(role, row["id"], {p: len(v) for p, v in local[row["id"]]["prompt_parts"].items()}, flush=True)
        summaries[role] = dict(full=integer_counts(landmark_audit(annotated, full)),
                               manual_crop=integer_counts(landmark_audit(annotated, local)),
                               predictions=local)
    write_json(OUTPUT, dict(freeze=digest(FREEZE), baseline_cache=digest(WORK / "part-features-fit.json"),
                            threshold=MINIMUM, outcomes=summaries,
                            scope="Manually bounded syringe crops. Fitting landmarks and one separate known-author photo."))
    for role, values in summaries.items():
        print(role, {k: v for k, v in values.items() if k != "predictions"}, flush=True)


if __name__ == "__main__":
    run()
