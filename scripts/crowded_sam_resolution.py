"""Compare SAM 3 input scale, text and instance suppression on public crowd photos."""
import argparse
from base64 import b64decode, b64encode
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.detection_crowd_field_data import MANIFEST as FIELD
from scripts.detection_crowd_source_data import MANIFEST as ARCHIVE
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.expanded_detection_study import short
from scripts.sam3_study import LOCK, Segmenter, configure_cpu

FOCAL = ART / "crowded_sam_focal_reference.json"
OUTPUT = ART / "crowded_sam_development.json"
CACHE = DATA / "proposals" / "crowded-sam"
PROMPTS = ("syringe", "transparent plastic syringe", "syringe barrel")
EDGES = (504, 756)
MASK_EDGE = 144
FLOORS = (.12, .22, .35, .50)


def rows(role):
    if role == "development":
        return [read_json(FOCAL), *read_json(FIELD)["rows"]]
    # NathanF in Commons and Nathan Forget on Flickr are linked by the nathanf profile.
    return [row for row in read_json(ARCHIVE)["rows"] if row["id"] != "6522808"]


def binding(role, edge):
    return dict(checkpoint=digest(LOCK), implementation=digest(__file__), edge=edge,
                prompts=list(PROMPTS), inputs={row["id"]: [row["sha256"], row["objects"]] for row in rows(role)})


class CrowdedMasks(Segmenter):
    def candidates(self, picture):
        import torch

        size = dict(height=self.edge, width=self.edge)
        with torch.inference_mode():
            pixels = self.processor(images=picture, return_tensors="pt", size=size).pixel_values.to(self.device)
            vision = self.model.get_vision_features(pixel_values=pixels)
            for prompt in PROMPTS:
                tokens = self.processor(text=prompt, return_tensors="pt").to(self.device)
                output = self.model(vision_embeds=vision, **tokens)
                postprocess = self.processor.post_process_instance_segmentation
                predicted = postprocess(output, threshold=.08, target_sizes=[(MASK_EDGE, MASK_EDGE)])[0]
                boxes = predicted["boxes"].cpu().tolist()
                scores = predicted["scores"].cpu().tolist()
                masks = predicted["masks"].cpu().numpy()
                candidates = []
                for box, score, mask in zip(boxes, scores, masks, strict=True):
                    box = [max(0., min(1., coordinate / MASK_EDGE)) for coordinate in box]
                    if not valid_box(box) or not mask.any():
                        continue
                    y, x = np.where(mask)
                    silhouette = [x.min() / MASK_EDGE, y.min() / MASK_EDGE,
                                  (x.max() + 1) / MASK_EDGE, (y.max() + 1) / MASK_EDGE]
                    candidates.append(dict(bbox=box, silhouette=[float(p) for p in silhouette], score=score,
                                           mask=b64encode(np.packbits(mask).tobytes()).decode("ascii")))
                yield prompt, sorted(candidates, key=lambda item: item["score"], reverse=True)[:24]


def collect(role, edge):
    configure_cpu()
    checkpoint = binding(role, edge)
    output = CACHE / f"{role}-{edge}.json"
    record = read_json(output) if output.exists() else dict(binding=checkpoint, frames={})
    if record["binding"] != checkpoint:
        raise ValueError("The cached SAM inputs or implementation changed.")
    todo = [row for row in rows(role) if row["id"] not in record["frames"]]
    if todo:
        detector = CrowdedMasks(edge, "xpu")
        for row in todo:
            path = Path("data") / row["image"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"The public image {row['id']} has changed.")
            picture = read_image(path)
            start = perf_counter()
            views = dict(detector.candidates(picture))
            record["frames"][row["id"]] = dict(views=views, seconds=perf_counter() - start)
            write_json(output, record)
            print(edge, role, row["id"], {key: len(value) for key, value in views.items()}, flush=True)


def masks(candidates):
    return [dict(score=r["score"], bbox=r["bbox"], silhouette=r["silhouette"],
                 pixels=np.frombuffer(b64decode(r["mask"]), dtype=np.uint8)) for r in candidates]


def mask_iou(first, second):
    return np.count_nonzero(np.unpackbits(first & second)) / max(1, np.count_nonzero(np.unpackbits(first | second)))


def select(stream, geometry, suppression, floor):
    picked = []
    for current in sorted(stream, key=lambda item: item["score"], reverse=True):
        if current["score"] < floor:
            break
        if suppression == "mask":
            is_duplicate = any(mask_iou(current["pixels"], known["pixels"]) >= .45 for known in picked)
        else:
            is_duplicate = any(box_iou(current[geometry], known[geometry]) >= .45 for known in picked)
        if not is_duplicate:
            picked.append(current)
    return [dict(bbox=r[geometry], score=r["score"]) for r in picked]


def load(role):
    frames = {}
    for edge in EDGES:
        record = read_json(CACHE / f"{role}-{edge}.json")
        expected = binding(role, edge)
        if record["binding"] != expected or set(record["frames"]) != set(expected["inputs"]):
            raise ValueError("All frozen frames must be collected for each SAM resolution.")
        frames[edge] = {key: {p: masks(values) for p, values in f["views"].items()}
                        for key, f in record["frames"].items()}
    return frames


def evaluate_configuration(cohort, frames, configuration):
    edge, stream, geometry, suppression, floor = configuration
    names = PROMPTS if stream == "all prompts" else (stream,)
    predicted = {key: select([r for p in names for r in views[p]], geometry, suppression, floor)
                 for key, views in frames[edge].items()}
    stats = evaluate(cohort, predicted, 0.)
    return dict(metrics=short(stats), counts={key: len(val) for key, val in predicted.items()},
                accepted=predicted)


def measure(role):
    cohort = rows(role)
    frames = load(role)
    control = (504, "syringe", "bbox", "box", .22)
    if role == "development":
        combinations = product(EDGES, (*PROMPTS, "all prompts"), ("bbox", "silhouette"), ("box", "mask"), FLOORS)
        results = [dict(configuration=values, **evaluate_configuration(cohort, frames, values))
                   for values in combinations]
        # Rank by true object detection first, then count error and false positives.
        results.sort(key=lambda trial: (
            -trial["metrics"]["f1"],
            sum(abs(trial["counts"][r["id"]] - len(r["objects"])) for r in cohort),
            trial["metrics"]["fp"], trial["configuration"]), reverse=False)
        selected = results[0]
        target = OUTPUT
    else:
        development = read_json(OUTPUT)
        selected = dict(configuration=development["selected"]["configuration"])
        selected.update(evaluate_configuration(cohort, frames, selected["configuration"]))
        results = [selected]
        target = ART / "crowded_sam_transfer.json"
    if target.exists():
        raise FileExistsError("The comparison is already scored.")
    write_json(target, dict(selected=selected, control=dict(configuration=control,
                    **evaluate_configuration(cohort, frames, control)), tested=len(results), leading=results[:10],
                    counts_reference={r["id"]: len(r["objects"]) for r in cohort},
                    acquisition_groups={r["id"]: r["group"] for r in cohort},
                    cached_inputs={str(edge): digest(CACHE / f"{role}-{edge}.json") for edge in EDGES},
                    protocol="Mask-based identity and three text formulations at two scales. The Flickr and Commons "
                             "NathanF views belong to a shared provenance group. Transfer is a fixed candidate "
                             "check on three other previously reviewed acquisitions. No Fil Rouge data included."))
    print(role, selected["configuration"], selected["metrics"], selected["counts"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    parser.add_argument("role", choices=("development", "transfer"))
    parser.add_argument("--edge", type=int, choices=EDGES)
    args = parser.parse_args()
    if args.action == "measure":
        measure(args.role)
    elif args.edge:
        if args.role == "transfer" and not OUTPUT.exists():
            parser.error("Freeze the development choice before collecting the transfer sources.")
        collect(args.role, args.edge)
    else:
        parser.error("Select a fixed SAM input scale.")
