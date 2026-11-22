"""The application and command-line entry point for validated local inference."""
import argparse
from pathlib import Path

from medsyringe.runtime.crop_verifier_runtime import install_crop_verifier
from medsyringe.data import digest, read_image, write_json
from medsyringe.runtime.grounding_backend import analyze, enable_intel
from medsyringe.analysis.inference import Engine


class ProductionEngine(Engine):
    def __init__(self, root):
        super().__init__(root)
        self.crop_verifier = install_crop_verifier(self, root)

    def close(self):
        try:
            if self.crop_verifier is not None:
                self.crop_verifier.close()
        finally:
            super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--intel", action="store_true")
    args = parser.parse_args()
    engine = ProductionEngine(args.root)
    try:
        if args.intel:
            enable_intel(engine, args.root)
        record = analyze(engine, read_image(args.image), args.ocr)
    finally:
        engine.close()
    record["source_sha256"] = digest(args.image)
    write_json(args.output, record)
    print(f"Saved {len(record['objects'])} proposed objects to {args.output}")


if __name__ == "__main__":
    main()
