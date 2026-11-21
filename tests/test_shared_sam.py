from pathlib import Path
from types import SimpleNamespace

import pytest

from medsyringe import shared_sam


def fixture_engine():
    options = {"edge": 504, "device": "xpu", "cutoff": .9}
    previous = SimpleNamespace(options=options, process=None, python=Path("local"))
    support = SimpleNamespace(options={k: options[k] for k in ("edge", "device")}, process=None)
    return SimpleNamespace(sam3=previous, crop_verifier=SimpleNamespace(support_worker=support),
                           expanded=SimpleNamespace(worker=None))


def test_shared_worker_keeps_all_calls_with_one_model(monkeypatch, tmp_path):
    calls = []

    class Worker:
        def __init__(self, root, python, mode, options):
            self.options = options
            self.python = python
            self.process = None

        def run(self, image, operation=None):
            calls.append((image, operation))
            return operation

    monkeypatch.setattr(shared_sam, "ImageWorker", Worker)
    engine = fixture_engine()
    support = engine.crop_verifier.support_worker
    worker = shared_sam.share(engine, tmp_path)
    assert engine.sam3 is worker
    assert worker.options["cutoff"] == .9
    assert engine.sam3.run("full scene") is None
    assert support.run("resized scene", operation="detect") == "detect"
    assert engine.sam3.run("crop", operation="liquid") == "liquid"
    assert calls == [("full scene", None), ("resized scene", "detect"), ("crop", "liquid")]
    assert support.process is None
    with pytest.raises(ValueError, match="must be set once"):
        shared_sam.share(engine, tmp_path)


def test_shared_worker_rejects_mismatched_device(tmp_path):
    engine = fixture_engine()
    engine.crop_verifier.support_worker.options["device"] = "cpu"
    with pytest.raises(ValueError, match="different segmentation configurations"):
        shared_sam.share(engine, tmp_path)
