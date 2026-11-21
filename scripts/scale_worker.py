"""Read targeted cropped digits through the isolated local PaddleOCR environment."""
import json
import sys
import traceback
from pathlib import Path

from medsyringe.graduation_ocr import collect, supplement_graduations
from medsyringe.local_models import decode_picture

MODELS = {"tiny": "PP-OCRv6_tiny_rec", "medium": "PP-OCRv6_medium_rec"}
MODEL_ROOT = Path("data/ocr-v6/models")


def main():
    channel = sys.stdout
    sys.stdout = sys.stderr
    from paddleocr import TextRecognition

    readers = {}
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            options = payload["options"]
            inscriptions, policy = options["inscriptions"], options["policy"]
            image = decode_picture(payload["image"])
            for name in policy["readers"]:
                if name not in readers:
                    model = MODELS[name]
                    readers[name] = TextRecognition(model_name=model, model_dir=str(MODEL_ROOT / model),
                                                    device="cpu", cpu_threads=2, enable_mkldnn=False)
            observations = collect(image, inscriptions, {name: readers[name] for name in policy["readers"]})
            result = {"result": supplement_graduations(inscriptions, observations, policy)}
        except Exception:
            traceback.print_exc(file=sys.stderr)
            result = {"error": "Stroke-guided text recognition failed."}
        channel.write(json.dumps(result)+"\n")
        channel.flush()


if __name__ == "__main__":
    main()
