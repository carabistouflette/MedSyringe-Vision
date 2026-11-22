"""Localize proposal failures on the already observed prospective pilot."""
from medsyringe.data import digest, read_image, write_json
from medsyringe.detectors import box_iou
from scripts.low_count_prospective import OUTPUT as PILOT, inputs
from scripts.low_count_recovery import crowd, configure_cpu
from scripts.sam3_study import Segmenter

RESULT = crowd.ART / "low_count_candidate_diagnostic.json"
PROMPTS = ("syringe", "prefilled syringe", "laboratory pipette", "transfer pipette", "serological pipette")
MINIMUM = .50


def infer(model, image, prompts):
    import torch

    with torch.inference_mode():
        picture = model.processor(images=image, return_tensors="pt",
                                  size=dict(height=model.edge, width=model.edge)).pixel_values.to(model.device)
        vision = model.model.get_vision_features(pixel_values=picture)
        observed = {}
        for name in prompts:
            token = model.processor(text=name, return_tensors="pt").to(model.device)
            result = model.model(vision_embeds=vision, **token)
            post = model.processor.post_process_instance_segmentation(
                result, threshold=MINIMUM, target_sizes=[(image.height, image.width)])[0]
            observed[name] = [dict(
                bbox=[max(0., min(1., a / total)) for a, total in zip(box, image.size * 2, strict=True)],
                score=score) for box, score in zip(post["boxes"].cpu().tolist(), post["scores"].cpu().tolist(), strict=True)]
        return observed


def project(box, top, bottom, height):
    l, t, r, b = box
    return [l, (t*(bottom-top)+top)/height, r, (b*(bottom-top)+top)/height]


def oracle(boxes, proposals):
    return sum(any(box_iou(ref["bbox"], p["bbox"]) >= .50 for p in proposals) for ref in boxes)


def run():
    configure_cpu()
    model = Segmenter(756, "xpu")
    outcomes = []
    for row in inputs():
        picture = read_image(row["image"])
        full = infer(model, picture, PROMPTS)
        details = dict(id=row["id"], reference_count=len(row["objects"]), full=full)
        if row["objects"]:
            tiles = []
            for start, end in ((0., .48), (.26, .74), (.52, 1.)):
                top, bottom = round(start*picture.height), round(end*picture.height)
                bands = infer(model, picture.crop((0, top, picture.width, bottom)), PROMPTS[:2])
                tiles.extend({**p, "bbox": project(p["bbox"], top, bottom, picture.height), "band": [start, end]}
                             for name, candidates in bands.items() for p in candidates
                             if (start == 0. or p["bbox"][1] > .025) and (end == 1. or p["bbox"][3] < .975))
            counts = {name: oracle(row["objects"], values) for name, values in full.items()}
            counts["tiles syringe or prefilled syringe"] = oracle(row["objects"], tiles)
            counts["full and tiled syringe prompts"] = oracle(row["objects"], [*tiles, *full[PROMPTS[0]],
                                                                             *full[PROMPTS[1]]])
            details.update(oracle_box_coverage=counts, tiles=tiles)
            print(row["id"], counts, flush=True)
        else:
            print(row["id"], {name: len(objects) for name, objects in full.items()}, flush=True)
        outcomes.append(details)
    write_json(RESULT, dict(pilot=digest(PILOT), cases=outcomes, minimum_proposal_score=MINIMUM,
                            scope="The same pilot was already scored. Oracle coverage allows different proposals per "
                                  "reference and does not describe a fixed operating detector."))


if __name__ == "__main__":
    run()
