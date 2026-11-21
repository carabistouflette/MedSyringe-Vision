"""Benchmark and diagnose SAM 3 as a local source of supplemental detections."""
import argparse
from hashlib import sha256
from itertools import chain
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from statistics import median
from time import perf_counter

from medsyringe.additive import additions, overlapping_tiles, project
from medsyringe.auxiliary_detector import load_verified_recovery
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.optimize_detector import FINAL, counters
from scripts.precision_study import source
from scripts.rfdetr_verifier import baseline_for, inputs
from scripts.zoom_study import regions

ART = Path("artifacts")
MODELS = Path("data/sam3")
WEIGHTS = MODELS / "facebook-sam3"
MODEL = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
CHECKSUM = "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"
RELEASE = ART / "rfdetr_verified_release.json"
LOCK = ART / "sam3_checkpoint.json"
PREPARED = MODELS / "diagnostic.json"
REPORT = ART / "sam3_diagnostic.json"
PROMPT = "syringe"
RAW_MINIMUM = .02


def download(token_path):
    from huggingface_hub import HfApi, get_token, snapshot_download

    token = token_path.read_text().strip() if token_path else get_token()
    if not token:
        raise ValueError("An authorized Hugging Face read credential is needed.")
    HfApi().auth_check(MODEL, token=token)
    snapshot_download(MODEL, token=token, revision=REVISION, local_dir=WEIGHTS,
                      allow_patterns=["LICENSE", "config.json", "model.safetensors", "*tokenizer*",
                                      "*processor_config.json", "special_tokens_map.json", "vocab.json",
                                      "merges.txt"], max_workers=4)
    lock_checkpoint()


def lock_checkpoint():
    names = [path for path in sorted(WEIGHTS.iterdir()) if path.is_file()]
    for name in ("config.json", "model.safetensors"):
        if not (WEIGHTS / name).is_file():
            raise ValueError(f"Checkpoint file is missing: {name}")
    hashes = {p.name: digest(p) for p in names}
    if hashes["model.safetensors"] != CHECKSUM:
        raise ValueError("SAM 3 checkpoint differs from the published LFS SHA-256.")
    write_json(LOCK, dict(source=MODEL, revision=REVISION, files=hashes))


def configure_cpu():
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def cpu_probe(edge):
    import torch
    import transformers
    from transformers import Sam3Config, Sam3Model

    configure_cpu()
    torch.manual_seed(607)
    config = (Sam3Config.from_pretrained(WEIGHTS, local_files_only=True)
              if (WEIGHTS / "config.json").exists() else Sam3Config())
    config.image_size = edge
    config._attn_implementation = "sdpa"
    started = perf_counter()
    model = Sam3Model(config).eval()
    load_seconds = perf_counter()-started
    stats = []
    pixels = torch.zeros(1, 3, edge, edge)
    tokens = torch.tensor([[49406, 1000, 49407]])
    attention = torch.ones_like(tokens)
    for _ in range(2):
        began = perf_counter()
        with torch.inference_mode():
            outputs = model(pixel_values=pixels, input_ids=tokens, attention_mask=attention)
        stats.append(dict(edge=edge, seconds=perf_counter()-began,
                          box_shape=list(outputs.pred_boxes.shape), mask_shape=list(outputs.pred_masks.shape),
                          peak_rss_kib=getrusage(RUSAGE_SELF).ru_maxrss))
        print(stats[-1], flush=True)
        del outputs
    target = ART / ("sam3_cpu_probe.json" if edge == 1008 else f"sam3_cpu_probe_{edge}.json")
    write_json(target, dict(
        scope="Architecture and resource probe with random weights and synthetic input; no detector accuracy measured.",
        parameters=sum(p.numel() for p in model.parameters()), load_seconds=load_seconds,
        device="cpu", threads=4, dtype="float32", transformers=transformers.__version__,
        torch=torch.__version__, measurements=stats))
    print(stats, flush=True)


def prepare():
    configure_cpu()
    cutoff = read_json(FINAL)["cutoff"]
    original = inputs("fil_rouge")
    misses = {row["id"] for row in read_json(ART / "rfdetr_refined_fil_rouge.json")["verified"]["false_negatives"]}
    positives = sorted((r for r in original if r["id"] not in misses),
                       key=lambda row: sha256(row["id"].encode()).digest())[:4]
    pens = inputs("pen_reserve")
    negatives = sorted(pens, key=lambda row: sha256(row["id"].encode()).digest())[:2]
    failed = [row for row in original if row["id"] in misses]
    baseline = {**baseline_for("fil_rouge", original), **baseline_for("pen_reserve", pens)}
    runner = load_verified_recovery(Path("."), FINAL)
    if runner is None:
        raise ValueError("The approved local hybrid must be active.")
    cases = {r["id"]: role for group, role in ((failed, "known_miss"), (positives, "positive_control"),
                                               (negatives, "negative_development_control")) for r in group}
    rows = failed + positives + negatives
    for row in rows:
        identifier = row["id"]
        baseline[identifier] = runner.apply(source(row), baseline[identifier], cutoff)
    write_json(PREPARED, dict(rows=rows, roles=cases, baseline={r["id"]: baseline[r["id"]] for r in rows},
                              release_sha256=digest(RELEASE), base_sha256=digest(FINAL)))


def mask_geometry(mask):
    coordinates = (mask > 0).nonzero()
    if not len(coordinates):
        return None, 0.
    top, left = coordinates.amin(dim=0).tolist()
    bottom, right = coordinates.amax(dim=0).tolist()
    height, width = mask.shape
    return [left/width, top/height, (right+1)/width, (bottom+1)/height], len(coordinates)/(height*width)


class Segmenter:
    def __init__(self, edge, device):
        import torch
        from transformers import Sam3Config, Sam3Model, Sam3Processor

        if device == "xpu" and not torch.xpu.is_available():
            raise ValueError("PyTorch does not see the local Intel GPU.")

        lock = read_json(LOCK)
        if lock["revision"] != REVISION or any(digest(WEIGHTS / x) != h for x, h in lock["files"].items()):
            raise ValueError("SAM 3 files differ from the pinned official checkpoint.")
        config = Sam3Config.from_pretrained(WEIGHTS, local_files_only=True)
        config.image_size = edge
        self.model = Sam3Model.from_pretrained(WEIGHTS, local_files_only=True, config=config,
                                               attn_implementation="sdpa").eval().to(device)
        self.processor = Sam3Processor.from_pretrained(WEIGHTS, local_files_only=True)
        self.edge = edge
        self.device = device

    def predict(self, image):
        import torch

        with torch.inference_mode():
            inputs = self.processor(images=image, text=PROMPT, return_tensors="pt",
                                    size={"height":self.edge,"width":self.edge}).to(self.device)
            output = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(output, threshold=RAW_MINIMUM)[0]
            predictions = []
            for box, score, mask in zip(result["boxes"].tolist(), result["scores"].tolist(), result["masks"], strict=True):
                box = [max(0., min(1., v)) for v in box]
                if valid_box(box):
                    # SAM's instance postprocessor returns binary masks.
                    mask_box, foreground = mask_geometry(mask)
                    predictions.append(dict(bbox=box, score=score, mask_bbox=mask_box,
                                            mask_area_fraction=foreground))
        return suppress(predictions, .45)


def crop_predictions(predict, image, windows):
    for window in windows:
        for prediction in predict(image.crop(window)):
            box = project(prediction["bbox"], window, image.size)
            if box is None:
                continue
            mask = prediction["mask_bbox"]
            mask = project(mask, window, image.size, edge=0.) if mask else None
            yield {**prediction, "bbox": box, "mask_bbox": mask,
                   "mask_area_fraction": prediction["mask_area_fraction"] *
                   ((window[2]-window[0])*(window[3]-window[1])) / (image.width*image.height)}


def coverage(row, found):
    return [{view: dict(best_iou=max((box_iou(p["bbox"], o["bbox"]) for p in proposals), default=0.),
                        best_matching_score=max((p["score"] for p in proposals
                                                 if box_iou(p["bbox"], o["bbox"]) >= .5), default=None),
                        best_mask_box_iou=max((box_iou(p["mask_bbox"],o["bbox"]) for p in proposals
                                               if p["mask_bbox"]), default=0.))
             for view, proposals in found.items()} for o in row["objects"]]


def measure(edge, device):
    import torch
    import transformers

    configure_cpu()
    data = read_json(PREPARED)
    row = next(r for r in data["rows"] if data["roles"][r["id"]] == "positive_control")
    image = source(row)
    started = perf_counter()
    segmenter = Segmenter(edge, device)
    load_seconds = perf_counter()-started
    times = []
    for _ in range(2):
        started = perf_counter()
        predictions = segmenter.predict(image)
        times.append(perf_counter()-started)
    details = dict(device=device, torch=torch.__version__, transformers=transformers.__version__,
                   edge=edge, image=row["id"], checkpoint_sha256=digest(LOCK),
                   load_seconds=load_seconds, inference_seconds=times,
                   predictions=predictions, peak_rss_kib=getrusage(RUSAGE_SELF).ru_maxrss,
                   coverage=coverage(row, {"full":predictions}))
    write_json(ART/f"sam3_measured_{device}_{edge}.json",details)
    print(device,edge,times,details["coverage"],flush=True)


def diagnose(edge, device, selected_views):
    import torch
    import transformers

    configure_cpu()
    prepared = read_json(PREPARED)
    if prepared["release_sha256"] != digest(RELEASE) or prepared["base_sha256"] != digest(FINAL):
        raise ValueError("The active detector changed after the diagnostic was prepared.")
    rows = prepared["rows"]
    base = prepared["baseline"]
    floor = read_json(FINAL)["cutoff"]
    rf_cache = MODELS.parent / "rfdetr/proposals_fil_rouge_public.json"
    weak = read_json(rf_cache)["payload"]["predictions"]["full_and_tiles"]
    key = dict(input_sha256=digest(PREPARED), checkpoint_sha256=digest(LOCK),
               code_sha256=digest(__file__), edge=edge, prompt=PROMPT, device=device,
               torch=torch.__version__, transformers=transformers.__version__)
    cache_path = MODELS / f"sam3-proposals-{device}-{edge}.json"
    previous = read_json(cache_path) if cache_path.exists() else {}
    frames = previous["frames"] if previous.get("key") == key else {}
    segmenter = None
    for row in rows:
        identifier = row["id"]
        frame = frames.setdefault(identifier,dict(proposals={},times={}))
        missing = [view for view in selected_views if view not in frame["proposals"]]
        if not missing:
            continue
        if segmenter is None:
            segmenter = Segmenter(edge, device)
        image = source(row)
        base_windows = regions(base[identifier], image.size, floor)
        # Use one high-ranked RF-DETR region even when the source score is
        # below the deployed branch's operating point.
        rf_ranked = [{**p, "score": .31} for p in weak.get(identifier, []) if p["score"] >= .04]
        accepted = [p for p in base[identifier] if p["score"] >= floor]
        rf_windows = regions([*accepted, *rf_ranked], image.size, floor)
        windows = dict(full=[(0, 0, *image.size)],
                       guided=list(dict.fromkeys(chain(base_windows,rf_windows)))[:3],
                       tiles=overlapping_tiles(image.size,.66))
        proposals, times, views = frame["proposals"], frame["times"], {}
        for label in missing:
            group = windows[label]
            elapsed = 0.
            for window in group:
                if window not in views:
                    started = perf_counter()
                    views[window] = list(crop_predictions(segmenter.predict, image, [window]))
                    elapsed += perf_counter()-started
            proposals[label] = suppress(list(chain.from_iterable(views[w] for w in group)), .45)
            times[label] = elapsed
        frames[identifier] = dict(proposals=proposals, times=times, coverage=coverage(row,proposals),
                                  peak_rss_kib=getrusage(RUSAGE_SELF).ru_maxrss)
        write_json(cache_path,dict(key=key,frames=frames))
        print(identifier, "seconds", round(sum(times.values()),2), "coverage", frames[identifier]["coverage"],
              flush=True)
    main = evaluate(rows,base,floor)
    trials = []
    for view in selected_views:
        for threshold in (.20,.35,.50,.65,.80):
            added = {k:additions(base[k],frame["proposals"][view],floor,support=None,
                                 aux_cutoff=threshold,iou=.3) for k,frame in frames.items()}
            audit = evaluate(rows,added,floor)
            trials.append(dict(view=view,score=threshold, **counters(audit), guard=compare(main,audit)))
    write_json(REPORT.with_name(f"sam3_diagnostic_{device}_{edge}.json"),dict(scope="Known misses and previously consulted development controls.",
                           parameters=key,baseline=counters(main),trials=trials,roles=prepared["roles"],
                           latency_seconds={key:median(f["times"][key] for f in frames.values())
                                            for key in selected_views},
                           frames={key:{k:v for k,v in val.items() if k!="proposals"} for key,val in frames.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download","lock","probe","prepare","measure","diagnose"))
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--edge", type=int, default=1008, choices=(504,672,1008))
    parser.add_argument("--device", choices=("cpu","xpu"), default="cpu")
    parser.add_argument("--views", nargs="+", choices=("full","guided","tiles"),
                        default=("full","guided","tiles"))
    args = parser.parse_args()
    if args.stage == "download":
        download(args.token_file)
    elif args.stage == "lock":
        lock_checkpoint()
    elif args.stage == "probe":
        cpu_probe(args.edge)
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "measure":
        measure(args.edge,args.device)
    else:
        diagnose(args.edge,args.device,args.views)


if __name__ == "__main__":
    main()
