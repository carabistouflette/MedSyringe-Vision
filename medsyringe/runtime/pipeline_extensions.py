"""Load only independently checked optional local inference capabilities."""
from pathlib import Path

from medsyringe.data import digest, read_json
from medsyringe.runtime.local_models import ImageWorker

SAM_RELEASE=Path("artifacts/sam3_release.json")
MARKING_RELEASE=Path("artifacts/capacity_ocr_release.json")
LIQUID_RELEASE=Path("artifacts/liquid_appearance_release.json")


def verified_config(root,path):
    if not (root/path).exists():
        return None
    config=read_json(root/path)
    evidence=root/config["evidence"]
    if digest(evidence)!=config["evidence_sha256"] or not read_json(evidence)["passed"]:
        raise ValueError(f"Unverified optional model evidence for {path.name}.")
    package=Path(__file__).resolve().parents[2]
    if any(digest(package/name)!=expected for name,expected in config["code"].items()):
        raise ValueError(f"Optional model code changed for {path.name}.")
    if any(digest(root/name)!=expected for name,expected in config.get("models",{}).items()):
        raise ValueError(f"Optional model weights changed for {path.name}.")
    return config


def worker_for(root,path,mode):
    config=verified_config(root,path)
    if config is None:
        return None
    if not (root/config["executable"]).is_file():
        raise FileNotFoundError(config["executable"])
    return ImageWorker(root,Path(config["executable"]),mode,config["options"])
