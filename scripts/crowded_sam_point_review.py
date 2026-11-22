"""Measure click-assisted SAM 3 masks on the public overlap development photograph."""
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import ImageOps

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.crowded_sam_resolution import FOCAL
from scripts.detection_guided_geometry import EDGE, GuidedMasks
from scripts.detection_hillclimb_experiment import ART
from scripts.expanded_detection_study import short
from scripts.sam3_study import LOCK, configure_cpu, mask_geometry

ANCHORS = ART / "public_overlap_count_case.json"
OUTPUT = ART / "crowded_click_review_development.json"


def click_geometry(model, pixels, points, labels, embedding):
    torch = model.torch
    arguments = dict(input_points=torch.tensor([points], device=model.model.device, dtype=torch.float32),
                     input_labels=torch.tensor([labels], device=model.model.device, dtype=torch.int64))
    output = model.model(image_embeddings=embedding, multimask_output=True, **arguments)
    scores = output.iou_scores[0].cpu()
    masks = model.processor.post_process_masks(output.pred_masks.cpu(), pixels["original_sizes"])[0]
    return [[dict(bbox=region, score=float(quality), mask_fraction=fraction)
             for proposal, quality in zip(alternatives, qualities, strict=True)
             for region, fraction in [mask_geometry(proposal)] if region is not None]
            for alternatives, qualities in zip(masks, scores, strict=True)]


def run():
    if OUTPUT.exists():
        raise FileExistsError("The click-assisted comparison is already measured.")
    row = read_json(FOCAL)
    source = Path("data") / row["image"]
    reference = read_json(ANCHORS)["manual_reference"]
    if digest(source) != row["sha256"] or row["sha256"] != reference["sha256"]:
        raise ValueError("The click references and photo differ.")
    anchors = np.asarray([entry["coordinate"] for entry in reference["anchors"]], dtype=np.float32)
    if len(anchors) != len(row["objects"]):
        raise ValueError("Use one click per manually identified body.")
    configure_cpu()
    detector = GuidedMasks()
    picture = ImageOps.contain(read_image(source), (EDGE, EDGE))
    pixels = detector.processor(images=picture, return_tensors="pt", size={"height": EDGE, "width": EDGE})
    normalized_distance = np.linalg.norm((anchors[:, None] - anchors[None, :]) * picture.size, axis=2)
    np.fill_diagonal(normalized_distance, np.inf)
    nearest = normalized_distance.argmin(axis=1)
    # Coordinates passed directly to the model are on its square preprocessor raster.
    positive = (anchors * EDGE).tolist()
    cases = {"one click": ([[point] for point in positive], [[1] for _ in positive]),
             "one click and adjacent exclusion": (
                 [[point, positive[peer]] for point, peer in zip(positive, nearest, strict=True)],
                 [[1, 0] for _ in positive])}
    started = perf_counter()
    report = {}
    with detector.torch.inference_mode():
        tensor = pixels.pixel_values.to(device=detector.model.device, dtype=detector.model.dtype)
        embedding = detector.model.get_image_embeddings(tensor)
        for name, (points, labels) in cases.items():
            began = perf_counter()
            options = click_geometry(detector, pixels, points, labels, embedding)
            chosen = [max(group, key=lambda p: p["score"]) for group in options if group]
            overlap = [[box_iou(candidate["bbox"], ref["bbox"]) for candidate in group]
                       for group, ref in zip(options, row["objects"], strict=True)]
            report[name] = dict(
                actual=short(evaluate([row], {row["id"]: chosen}, 0.)),
                clicked_instance_matches=sum(bool(group) and
                    group[max(range(len(group)), key=lambda j: proposals[j]["score"])] >= .50
                    for group, proposals in zip(overlap, options, strict=True)),
                oracle_clicked_instance_matches=sum(max(group, default=0) >= .50 for group in overlap),
                geometry_seconds=perf_counter() - began, chosen=chosen,
                per_click_iou=overlap)
    write_json(OUTPUT, dict(results=report, total_inference_seconds=perf_counter() - started,
                            checkpoint_sha256=digest(LOCK), image_sha256=row["sha256"],
                            reference_sha256=digest(FOCAL), anchors_sha256=digest(ANCHORS),
                            scope="Human-assisted review on one development image. The ten input points already "
                                  "encode the count. Click-conditioned masks cannot establish automatic recall or count."))
    print({name: [entry["clicked_instance_matches"], entry["oracle_clicked_instance_matches"]]
           for name, entry in report.items()}, flush=True)


if __name__ == "__main__":
    run()
