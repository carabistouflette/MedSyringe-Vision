"""Choose a consistent local device for detection and the SAM workers."""
import os

from .grounding_backend import enable_intel


def device_request():
    requested = os.environ.get("MEDSYRINGE_DEVICE")
    if requested is None and os.environ.get("MEDSYRINGE_PRIMARY_DEVICE") == "xpu":
        return "xpu"
    if requested not in (None, "cpu", "xpu"):
        raise ValueError("MEDSYRINGE_DEVICE must be cpu or xpu.")
    return requested


def configure_device(engine, root, requested):
    if requested is None:
        return
    if requested == "xpu":
        enable_intel(engine, root)
        if engine.identity.get("primary_backend") == "xpu":
            return
    verifier = engine.crop_verifier
    for worker in (engine.sam3, engine.expanded.worker, verifier.support_worker if verifier else None):
        if worker is not None:
            worker.options = {**worker.options, "device": "cpu"}
