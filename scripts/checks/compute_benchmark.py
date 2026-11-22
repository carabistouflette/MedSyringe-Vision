"""Run same-image warm latency checks serially on both frozen CPU backbones."""
from pathlib import Path
from time import perf_counter

from medsyringe.data import read_image, read_json, write_json
from medsyringe.detection.detectors import Grounder, THREADS, suppress
import numpy as np

IMAGE_IDS = ("136214041", "26451793", "113356854")


def main():
    root = Path(".")
    rows = {r["id"]: r for r in read_json("artifacts/validation_manifest.json")["rows"]}
    inputs = [(identifier, read_image(root / "data" / rows[identifier]["image"])) for identifier in IMAGE_IDS]
    output = {}
    for name in ("dino", "owl"):
        start = perf_counter()
        detector = Grounder(root, name)
        load = 1000*(perf_counter()-start)
        suppress(detector.propose(inputs[0][1]))
        results = []
        for identifier, im in inputs * 2:
            started = perf_counter()
            suppress(detector.propose(im))
            results.append(dict(id=identifier, milliseconds=1000*(perf_counter()-started)))
        times = [item["milliseconds"] for item in results]
        output[name] = dict(load_ms=load, median_ms=float(np.median(times)),
                            p95_ms=float(np.percentile(times, 95)), samples=results)
        print(name, output[name]["median_ms"], flush=True)
        del detector
    write_json("artifacts/serial_cpu_benchmark.json", dict(threads=THREADS,
        scope="Batch 1. Excludes image decode and includes CPU preprocessing, forward and NMS.",
        results=output))


if __name__ == "__main__":
    main()
