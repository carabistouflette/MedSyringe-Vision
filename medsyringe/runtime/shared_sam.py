"""Route the fallback, corroboration, and apparent-liquid operations to one model."""
from pathlib import Path

from medsyringe.runtime.local_models import ImageWorker

ENTRY = Path(__file__).resolve().parents[2] / "scripts/workers/shared_sam_worker.sh"


def share(engine, root):
    original = engine.sam3
    support = engine.crop_verifier.support_worker if engine.crop_verifier else None
    if support is None or original is None or engine.expanded.worker is not None:
        raise ValueError("The shared SAM route requires the validated fallback and corroboration workers.")
    if original.process is not None or support.process is not None or original.python == ENTRY:
        raise ValueError("The shared SAM route must be set once before inference.")
    options = original.options
    if any(options[k] != support.options[k] for k in ("edge", "device")):
        raise ValueError("Fallback and corroboration use different segmentation configurations.")
    worker = ImageWorker(root, ENTRY, "sam3", dict(options))

    def corroborate(image, operation):
        return worker.run(image, operation=operation)

    support.run = corroborate
    engine.sam3 = worker
    return worker
