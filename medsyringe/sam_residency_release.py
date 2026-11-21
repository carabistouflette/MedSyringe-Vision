"""Activate the verified shared SAM worker on the Intel runtime."""
from pathlib import Path

from .appearance_runtime import RELEASE as DETECTOR
from .data import digest, read_json
from .pipeline_extensions import verified_config
from .shared_sam import share

RELEASE = Path("artifacts/detection_hillclimb/sam_residency_release.json")
PAIRING = Path("artifacts/detection_hillclimb/sam_residency_confirmation.json")


def activate(engine, root):
    validated = verified_config(root, RELEASE)
    if validated is None:
        return
    if engine.identity.get("primary_backend") != "xpu" or digest(root / DETECTOR) != validated["detector_sha256"]:
        raise ValueError("This SAM runtime release requires its qualified Intel detector.")
    if digest(root / PAIRING) != validated["pairing_sha256"] or not read_json(root / PAIRING)["passed"]:
        raise ValueError("The paired SAM operations did not reproduce their verified outputs.")
    share(engine, root)
    support = engine.crop_verifier.support_worker
    support.run = engine.detection_checks.observe(support.run, "SAM corroboration")
    engine.identity["sam_residency_sha256"] = digest(root / RELEASE)
