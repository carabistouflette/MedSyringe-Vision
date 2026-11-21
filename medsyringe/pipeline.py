"""Application inference with the frozen detector and validated scale-reader extension."""
import argparse
from pathlib import Path
from time import perf_counter

from .appearance_runtime import install_appearance
from .data import digest, read_image, write_json
from .detection_checks import DetectionChecks
from .grounding_backend import analyze
from .hardware import configure_device, device_request
from .production import ProductionEngine
from .scale_runtime import scale_reader


class MedSyringeEngine(ProductionEngine):
    def __init__(self, root):
        super().__init__(root)
        self.scale_reader = scale_reader(self, root)
        self.appearance = install_appearance(self, root)
        self.detection_checks = DetectionChecks(self)

    def analyze(self, image, ocr=False):
        started = perf_counter()
        self.detection_checks.begin()
        report = super().analyze(image, ocr)
        self.detection_checks.attach(report)
        if ocr and self.scale_reader is not None:
            self.scale_reader.apply(image, report)
            report["elapsed_ms"] = 1000*(perf_counter()-started)
        return report

    def close(self):
        try:
            try:
                if self.appearance is not None:
                    self.appearance.close()
            finally:
                if self.scale_reader is not None:
                    self.scale_reader.close()
        finally:
            super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--ocr", action="store_true")
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument("--intel", action="store_true", help="Use the validated Intel XPU runtime.")
    hardware.add_argument("--device", choices=("cpu", "xpu"), help="Choose the local detection and SAM device.")
    args = parser.parse_args()
    try:
        requested = "xpu" if args.intel else args.device or device_request()
    except ValueError as exc:
        parser.error(str(exc))
    engine = MedSyringeEngine(args.root)
    try:
        configure_device(engine, args.root, requested)
        report = analyze(engine, read_image(args.image), args.ocr)
    finally:
        engine.close()
    report["source_sha256"] = digest(args.image)
    write_json(args.output, report)
    print(f"Saved {len(report['objects'])} proposed objects to {args.output}")


if __name__ == "__main__":
    main()
