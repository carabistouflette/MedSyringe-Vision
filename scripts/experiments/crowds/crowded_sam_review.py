"""Generate a separate set of crowded-scene review boxes with the measured SAM 3 policy."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.data import digest, read_image, read_json, write_json
from scripts.experiments.crowds.crowded_sam_resolution import LOCK, OUTPUT, CrowdedMasks, configure_cpu, masks, select


def review(image_path, output_path, device):
    if output_path.resolve() == image_path.resolve() or output_path.exists():
        raise FileExistsError("Choose a new result path.")
    edge, prompt, geometry, identity, floor = read_json(OUTPUT)["selected"]["configuration"]
    if prompt == "all prompts":
        raise ValueError("The review runner expects a single frozen prompt.")
    configure_cpu()
    detector = CrowdedMasks(edge, device)
    image = read_image(image_path)
    start = perf_counter()
    stream = next(results for label, results in detector.candidates(image) if label == prompt)
    proposed = select(masks(stream), geometry, identity, floor)
    write_json(output_path, dict(
        status="review proposals",
        proposed_count=len(proposed), proposals=proposed, dimensions=image.size,
        image_sha256=digest(image_path), checkpoint_sha256=digest(LOCK), policy_sha256=digest(OUTPUT),
        device=device, inference_seconds=perf_counter() - start,
    ))
    print(f"{len(proposed)} syringe proposals for review", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    args = parser.parse_args()
    review(args.image, args.output, args.device)
