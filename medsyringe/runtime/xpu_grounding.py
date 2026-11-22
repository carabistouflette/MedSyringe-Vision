"""Use the pinned Grounding DINO checkpoint on the optional Intel runtime."""
import torch

from medsyringe.detection.detectors import PROMPT, RAW_MINIMUM


class AcceleratedGrounder:
    def __init__(self, grounder):
        if grounder.name != "dino" or not torch.xpu.is_available():
            raise ValueError("Grounding DINO acceleration requires the Intel runtime.")
        self.model = grounder.model.to("xpu")
        self.processor = grounder.processor

    @torch.inference_mode()
    def propose(self, image):
        inputs = self.processor(images=image, text=f"{PROMPT}.", return_tensors="pt").to("xpu")
        output = self.model(**inputs)
        record = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=RAW_MINIMUM, target_sizes=[(image.height, image.width)])[0]
        axes = (*image.size, *image.size)
        candidates = []
        for box, score in zip(record["boxes"].cpu().tolist(), record["scores"].cpu().tolist(), strict=True):
            x, y, r, b = [max(0., min(1., coordinate/extent)) for coordinate, extent in zip(box, axes, strict=True)]
            if r-x >= .005 and b-y >= .005:
                candidates.append(dict(bbox=[x, y, r, b], score=score))
        return sorted(candidates, key=lambda p: -p["score"])[:100]
