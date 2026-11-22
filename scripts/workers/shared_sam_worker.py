"""Serve the existing SAM operations from one resident segmentation model."""
import json
import sys
import traceback
from itertools import chain

from medsyringe.detection.additive import overlapping_tiles
from medsyringe.detection.detectors import suppress
from medsyringe.runtime.local_models import decode_picture
from scripts.experiments.measurement.liquid_measure import measure_crop
from scripts.experiments.expanded.sam3_study import Segmenter, configure_cpu, crop_predictions


def dispatch(segmenter, image, operation):
    if operation == "detect":
        return segmenter.predict(image)
    if operation == "detect with tiles":
        windows = chain([(0, 0, *image.size)], overlapping_tiles(image.size, .66))
        return suppress(list(crop_predictions(segmenter.predict, image, windows)), .45)
    if operation == "liquid":
        return measure_crop(segmenter, image)
    raise ValueError("Unsupported SAM operation.")


def main():
    channel = sys.stdout
    sys.stdout = sys.stderr
    configure_cpu()
    segmenter = None
    configuration = None
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            options = payload["options"]
            settings = options["edge"], options["device"]
            if segmenter is None:
                segmenter = Segmenter(*settings)
                configuration = settings
            if settings != configuration:
                raise ValueError("A running SAM worker cannot change resolution or device.")
            image = decode_picture(payload["image"])
            operation = payload.get("operation") or "detect with tiles"
            response = dict(result=dispatch(segmenter, image, operation))
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = dict(error=f"{type(error).__name__}: the shared SAM worker failed.")
        channel.write(json.dumps(response) + "\n")
        channel.flush()


if __name__ == "__main__":
    main()
