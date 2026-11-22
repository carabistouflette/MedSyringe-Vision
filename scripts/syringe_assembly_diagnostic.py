"""Diagnose remaining thresholds and measure expert-located pump crops."""
from PIL import Image

from medsyringe.additive import project
from medsyringe.data import digest, read_json, write_json
from scripts.occlusion_detector import ART, EDGE, INITIAL, raster
from scripts.occlusion_proposals import oracle
from scripts.rfdetr_compare import ranked_region
from scripts.syringe_assembly_study import OUTPUT, selected_policy
from scripts.thin_syringe_tiles import inputs, observations


def pump_probe(row):
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    settings = read_json(INITIAL)
    if digest(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise ValueError("The archived RF-DETR checkpoint changed.")
    model = RFDETRNano.from_checkpoint(settings["checkpoint"], device="cpu")
    image = raster(row)
    w, h = image.size
    w1 = tuple(round(v * unit) for v, unit in zip((.18, .34, .56, .54), (w, h, w, h), strict=True))
    w2 = tuple(round(v * unit) for v, unit in zip((.22, .385, .515, .48), (w, h, w, h), strict=True))
    results = {}
    for name, window in (("wide pump area", w1), ("tight pump area", w2)):
        for angle in (0, 90):
            view = image.crop(window)
            if angle:
                view = view.transpose(Image.Transpose.ROTATE_90)
            found = []
            for p in ranked_region(model, view, EDGE):
                x, y, X, Y = p["bbox"]
                box = [1-Y, x, 1-y, X] if angle else p["bbox"]
                found.append({**p, "bbox": project(box, window, image.size, edge=0)})
            results[f"{name} {angle}"] = oracle(row["objects"], found)
    return results


def run():
    prospective, _ = inputs("prospective")
    data = observations("prospective", prospective)
    locked = selected_policy()
    rule = locked["contenders"][locked["selected"]]["rule"] if locked["selected"] else None
    final_stages = {}
    if rule:
        for row in prospective:
            if not row["objects"]:
                continue
            ranked = data[row["id"]]["pools"][rule["view"]]
            final_stages[row["id"]] = dict(ranked=oracle(row["objects"], ranked),
                confidence_filter=oracle(row["objects"], [p for p in ranked if p["score"] >= rule["floor"]]))
    earlier, _ = inputs("history")
    pump = next(row for row in earlier if row["id"] == "crop-reserve-chemyx")
    crops = pump_probe(pump)
    write_json(ART / "syringe_assembly_error_audit.json", dict(results_sha256=digest(OUTPUT),
        prospective_stages=final_stages, pump_privileged_crop=crops, code_sha256=digest(__file__),
        scope="Post-evaluation diagnostic on known scenes. Expert-oriented pump crops are never used for automatic scores."))
    print({key: results["matched"] for key, results in crops.items()}, flush=True)


if __name__ == "__main__":
    run()
