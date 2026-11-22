"""Apply a qualified detector policy through the pinned local appearance worker."""
from pathlib import Path
from threading import Lock

from .data import digest, read_json
from .local_models import ImageWorker, WorkerError
from .pipeline_extensions import verified_config

RELEASE = Path("artifacts/detection_hillclimb/appearance_release.json")
ENTRY = Path(__file__).resolve().parents[1] / "scripts/appearance_worker.sh"


class AppearanceRuntime:
    def __init__(self, root, detector, mode=None, pack_crops=False):
        self.worker = ImageWorker(root, ENTRY, "appearance", {})
        self.detector = detector
        self.mode = mode
        self.pack_crops = pack_crops
        self.release = digest(root / RELEASE)
        self.lock = Lock()
        self.status = None

    def detect(self, image):
        ranked, cutoff, supplemental = self.detector(image)
        with self.lock:
            if self.mode == "veto-only" and not any(p["score"] >= cutoff for p in ranked):
                self.status = "no accepted detections needed crop verification"
                return ranked, cutoff, supplemental
            self.worker.options = dict(ranked=ranked, floor=cutoff, release_sha256=self.release)
            try:
                packed = None
                if self.pack_crops:
                    from scripts.appearance_text_encoder import pack_patches

                    packed = pack_patches(image, [p for p in ranked if p["score"] >= cutoff])
                scored = (self.worker.run(packed, operation="appearance crops") if packed is not None
                          else self.worker.run(image))
            except WorkerError:
                self.status = "baseline retained after an unavailable crop verifier"
                return ranked, cutoff, supplemental
            self.status = "local crop verification applied"
        return scored, cutoff, supplemental

    def close(self):
        with self.lock:
            child = self.worker.process
            if child is not None:
                self.worker.stop(child)
                self.worker.process = None


def install_appearance(engine, root):
    config = verified_config(root, RELEASE)
    if config is None:
        return None
    if engine.identity != config["baseline"]:
        raise ValueError("The crop appearance release does not match the local baseline.")
    if config.get("encoder") == "text-only":
        proof = root/config["runtime_parity"]["path"]
        if digest(proof) != config["runtime_parity"]["sha256"] or not read_json(proof)["passed"]:
            raise ValueError("The appearance encoder does not have a matching parity result.")
    if not (root / ".venv/bin/python").is_file():
        raise FileNotFoundError("Restore the pinned Python environment for crop verification.")
    pack_crops = config.get("encoder") == "text-only" and config.get("mode") == "veto-only"
    runtime = AppearanceRuntime(root, engine.expanded.detect, config.get("mode"), pack_crops)
    engine.expanded.detect = runtime.detect
    engine.identity["appearance_detection_sha256"] = runtime.release
    return runtime
