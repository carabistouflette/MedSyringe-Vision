"""Add verified nominal-capacity evidence without revising apparent fill."""
from pathlib import Path
from threading import Lock

from PIL import ImageOps

from .attributes import crop
from .data import digest
from .local_models import ImageWorker, WorkerError
from .pipeline_extensions import verified_config

RELEASE = Path("artifacts/fill/graduation_capacity_release.json")
ENTRY = Path(__file__).resolve().parents[1] / "scripts/scale_worker.sh"


class ScaleReader:
    def __init__(self, root, policy):
        self.worker = ImageWorker(root, ENTRY, "ocr", {})
        self.policy = policy
        self.lock = Lock()

    def apply(self, image, report):
        for obj in report["objects"][:10]:
            evidence = obj["inscriptions"]
            if evidence.get("capacity") or not evidence.get("candidates"):
                continue
            region = ImageOps.contain(crop(image, obj["bbox"]), (1900,1900))
            try:
                with self.lock:
                    self.worker.options = {"inscriptions": evidence, "policy": self.policy}
                    revised = self.worker.run(region)
            except WorkerError:
                continue
            if revised["capacity"] is None or revised["scale"] is None:
                continue
            obj["inscriptions"] = revised
            obj["scale"] = revised["scale"]["value"]
            obj["evidence"]["scale"] = {**revised["scale"], "reason": "Supported by stroke-guided printed markings"}
            fill = obj.get("fill_level")
            if fill and fill["status"] == "insufficient evidence":
                if fill["reason"] == "A visible 20 mL or 60 mL capacity is required.":
                    fill["reason"] = "Nominal capacity was read. The visible marks and piston did not establish the fill."
        return report

    def close(self):
        worker = self.worker
        if worker.process is not None:
            worker.stop(worker.process)
            worker.process = None


def scale_reader(engine, root):
    config = verified_config(root, RELEASE)
    if config is None:
        return None
    if engine.identity != config["baseline"]:
        raise ValueError("The graduation release does not match the established reader and detector.")
    engine.identity["graduation_capacity_sha256"] = digest(root / RELEASE)
    return ScaleReader(root, config["policy"])
