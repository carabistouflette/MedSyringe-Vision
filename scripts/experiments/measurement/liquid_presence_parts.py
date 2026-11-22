"""Test whether separately visible syringe parts can support an empty-barrel verdict."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from scripts.experiments.measurement.liquid_fields_audit import ART, REFERENCES
from scripts.experiments.expanded.sam3_study import LOCK, Segmenter, configure_cpu

OUTPUT = ART / "liquid_presence_part_experiment.json"
PARTS = ("syringe nozzle", "syringe plunger seal", "rubber stopper inside syringe")
MIN_SCORE = .40
PART_EXTENT = .32
MAX_GAP = .18


class VisibleParts(Segmenter):
    def measure(self, image):
        import torch

        wide = image.width > image.height
        axis = 0 if wide else 1
        if max(image.size) < 1.6 * min(image.size):
            return dict(status="indeterminate", reason="The image axis is ambiguous.", parts={})
        candidates = {}
        with torch.inference_mode():
            pixels = self.processor(images=image, size={"height": self.edge, "width": self.edge},
                                    return_tensors="pt").pixel_values.to(self.device)
            embedding = self.model.get_vision_features(pixel_values=pixels)
            for kind in PARTS:
                inputs = self.processor(text=kind, return_tensors="pt").to(self.device)
                result = self.processor.post_process_instance_segmentation(
                    self.model(vision_embeds=embedding, **inputs), threshold=MIN_SCORE, target_sizes=[(144, 144)])[0]
                proposals = []
                for box, score in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), strict=True):
                    coordinates = [max(0., min(1., value / 144)) for value in box]
                    if valid_box(coordinates) and coordinates[axis + 2] - coordinates[axis] <= PART_EXTENT:
                        center = (coordinates[axis] + coordinates[axis + 2]) / 2
                        proposals.append(dict(bbox=coordinates, score=score, axis_center=center))
                candidates[kind] = sorted(proposals, key=lambda p: p["score"], reverse=True)[:3]
        if not all(candidates.values()):
            return dict(status="indeterminate", reason="The nozzle or rubber seal was not consistently visible.",
                        parts=candidates)
        nozzle, first, second = (candidates[name][0]["axis_center"] for name in PARTS)
        gap = abs(nozzle - float(np.mean([first, second])))
        agreed = abs(first - second) <= .08
        return dict(status="empty geometry candidate" if agreed and gap <= MAX_GAP else "indeterminate",
                    gap_in_crop_lengths=gap, seal_agreement=agreed, parts=candidates)


def run():
    configure_cpu()
    model = VisibleParts(504, "xpu")
    observations = []
    for row in read_json(REFERENCES)["rows"]:
        path = Path(row["image"])
        if digest(path) != row["sha256"]:
            raise ValueError("The reference image changed.")
        region = ImageOps.contain(crop(read_image(path), row["bbox"]), (504, 504))
        initial = model.measure(region)
        turned = model.measure(region.transpose(Image.Transpose.ROTATE_180))
        fields = row["fields"]
        verified = initial["status"] == turned["status"] == "empty geometry candidate"
        observations.append(dict(id=row["id"], visible_liquid=fields["liquid_presence"]["value"],
                                 empty_candidate_both_orientations=verified, original=initial, rotated=turned))
        print(row["id"], initial["status"], turned["status"], flush=True)
    write_json(OUTPUT, dict(rows=observations, min_score=MIN_SCORE, maximum_part_extent=PART_EXTENT,
                            maximum_gap=MAX_GAP, reference_sha256=digest(REFERENCES), checkpoint_sha256=digest(LOCK),
                            scope="Exploratory part prompts with manually reviewed crop geometry. "
                                  "No changes to production presence, volume or type."))


if __name__ == "__main__":
    run()
