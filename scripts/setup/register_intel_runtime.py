"""Bind the optional Intel app backend to a passing paired regression."""
from importlib.metadata import version
from pathlib import Path

from medsyringe.data import digest, read_json, write_json

ART = Path("artifacts/expanded")


def main():
    summary = ART / "runtime_pair.json"
    if not read_json(summary)["passed"] or not version("torch").endswith("+xpu"):
        raise ValueError("A successful paired Intel runtime regression is required.")
    raw = read_json(ART / "runtime_pair_xpu.json")
    benchmark = "scripts/experiments/expanded/expanded_runtime_pair.py"
    if raw["script_sha256"] != digest(benchmark):
        raise ValueError("The paired benchmark script changed.")
    paths = (benchmark, "medsyringe/analysis/inference.py", "medsyringe/runtime/xpu_grounding.py",
             "medsyringe/runtime/grounding_backend.py")
    write_json(ART / "xpu_release.json", dict(id="intel-grounding-dino",
               code={name: digest(name) for name in paths}, evidence=str(summary), evidence_sha256=digest(summary),
               model=raw["outputs"][0]["result"]["model"],
               packages={name: version(name) for name in ("torch", "torchvision", "transformers")}))


if __name__ == "__main__":
    main()
