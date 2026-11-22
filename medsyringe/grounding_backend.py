"""Enable the validated Intel grounder and retry failed images on the CPU."""
import logging
from importlib.metadata import version
from pathlib import Path

from .pipeline_extensions import verified_config

RELEASE = Path("artifacts/expanded/xpu_release.json")
LOGGER = logging.getLogger(__name__)


class AccelerationUnavailable(Exception):
    pass


class IntelGrounder:
    def __init__(self, original):
        from .xpu_grounding import AcceleratedGrounder

        self.original = original
        self.accelerated = AcceleratedGrounder(original)

    def propose(self, image):
        try:
            return self.accelerated.propose(image)
        except RuntimeError as exc:
            raise AccelerationUnavailable("The Intel grounder could not analyze the image.") from exc


def enable_intel(engine, root):
    original = engine.detector
    try:
        config = verified_config(root, RELEASE)
        base_model = {key: engine.identity.get(key) for key in config["model"]} if config else {}
        if not config or config["model"] != base_model:
            raise ValueError("The Intel runtime evidence does not match the loaded pipeline.")
        if any(version(package) != expected for package, expected in config["packages"].items()):
            raise ValueError("The Intel runtime packages differ from the validated environment.")
        engine.detector = IntelGrounder(original)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.warning("Intel Grounding DINO unavailable. Using CPU. %s", exc)
        original.model.to("cpu")
        engine.identity["primary_backend"] = "cpu fallback"
        return
    engine.identity["primary_backend"] = "xpu"


def analyze(engine, image, ocr, **options):
    try:
        return engine.analyze(image, ocr, **options)
    except AccelerationUnavailable:
        LOGGER.warning("Intel Grounding DINO failed. Repeating the image on CPU.")
        original = engine.detector.original
        original.model.to("cpu")
        engine.detector = original
        engine.identity["primary_backend"] = "cpu fallback"
        return engine.analyze(image, ocr, **options)
