"""Serve expanded SAM and liquid requests through the local image worker protocol."""
import json
import sys
import traceback

import numpy as np

from medsyringe.measurement.liquid_color import describe_region, display_rgb, lab_pixels
from medsyringe.measurement.liquid_context import contextual_support
from medsyringe.measurement.liquid_region import RegionFeatures
from medsyringe.runtime.local_models import decode_picture
from scripts.experiments.expanded.expanded_liquid import mask_syringe, support
from scripts.experiments.expanded.sam3_study import Segmenter, configure_cpu


def main():
    channel = sys.stdout
    sys.stdout = sys.stderr
    configure_cpu()
    segmenter = None
    encoder = None
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            options = payload["options"]
            if segmenter is None:
                segmenter = Segmenter(options["edge"], options["device"])
            image = decode_picture(payload["image"])
            if payload["operation"] == "detect":
                result = segmenter.predict(image)
            elif payload["operation"] == "liquid":
                if encoder is None:
                    encoder = RegionFeatures(options["device"])
                screen, profile = display_rgb(image)
                rgb = np.asarray(screen)
                frame = dict(rgb=rgb, lab=lab_pixels(rgb), mask=mask_syringe(segmenter, image), grid=encoder.extract(image))
                selected = options["liquid"]
                supported = support(frame, selected["model"], selected["cutoff"])
                if selected.get("context") is not None:
                    supported = contextual_support(frame, supported, selected["context"])
                result = describe_region(rgb, supported, profile)
            else:
                raise ValueError("Unsupported expanded worker operation.")
            response = {"result": result}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"error": f"{type(error).__name__}: expanded analysis failed."}
        channel.write(json.dumps(response) + "\n")
        channel.flush()


if __name__ == "__main__":
    main()
