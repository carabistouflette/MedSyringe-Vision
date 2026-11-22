from types import SimpleNamespace
from unittest.mock import Mock

from medsyringe.runtime.grounding_backend import AccelerationUnavailable, analyze


def test_device_error_retries_the_entire_image_once_on_cpu():
    calls = []
    original = SimpleNamespace(model=Mock())
    xpu = SimpleNamespace(original=original)
    engine = SimpleNamespace(detector=xpu, identity={"primary_backend": "xpu"})
    image = object()

    def infer(frame, ocr):
        calls.append((engine.detector, frame, ocr))
        if engine.detector is xpu:
            raise AccelerationUnavailable()
        return {"objects": []}

    engine.analyze = infer
    assert analyze(engine, image, True) == {"objects": []}
    assert calls == [(xpu, image, True), (original, image, True)]
    assert engine.identity["primary_backend"] == "cpu fallback"
    original.model.to.assert_called_once_with("cpu")
