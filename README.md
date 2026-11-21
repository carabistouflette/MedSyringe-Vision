# MedSyringe Vision

MedSyringe Vision is a local research application for syringe localization, scene counts, visual review, and reference checks. It keeps observations and human revisions in a local SQLite database. Its detector runs from pinned weights on CPU or Intel Arc, and its reports contain the image evidence and original model output.

The current release returned the exact count on all **177 Fil Rouge photos** and matched **172 of 177 reference boxes** at IoU 0.50. These photos come from one capture group. On **23 external native-resolution photographs**, it matched **39 of 48** instances, produced four false boxes and returned 20 exact counts. On the five crowded photographs in that source, it matched 24 of 30 instances and counted two scenes exactly. [Full live regression](artifacts/detection_hillclimb/sam_residency_full_live.json) · [Independent device audit](artifacts/detection_hillclimb/device_scene_audit_results.json)

- [Scientific report](docs/report.md)
- [Expanded Fil rouge study and release](docs/expanded_report.md)
- [Precision, fill review, and Intel runtime experiments](docs/improvement_regression.md)
- [Graduation OCR, piston landmarks, and additive detection](docs/graduation_piston_recovery.md)
- [Detection across acquisition domains](docs/detection_generalization.md)
- [Apparent fill measurement and validation](docs/fill_measurement.md)
- [Specification status and acceptance evidence](docs/requirements.md)
- [Experimental protocol](docs/protocol.md)
- [Model and tool research](docs/research.md)
- [Presentation](docs/presentation.pdf)
- [Evaluation artifacts](artifacts/)
- [Engineering and reproduction guide](docs/engineering.md)

## Run locally

Use Linux x86-64, Python 3.12, uv, and curl. Install ExifTool for source preparation and Tesseract with English language data for optional marking extraction. Exact Python dependencies are in `uv.lock`. The delivered workspace already contains the released checkpoints and worker environments. For a new checkout, restore the trained TorchScript recovery model and the sealed SAM and OCR workers before launching. The setup guides list their files, revisions and checksums. [RF-DETR recovery setup](docs/rfdetr_verification.md) · [SAM and OCR worker setup](docs/sam3_markings_liquid.md)

```bash
uv sync --locked --python 3.12
bash scripts/fetch_models.sh
HF_HUB_OFFLINE=1 uv run --locked uvicorn medsyringe.app:app --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Upload a JPEG, PNG, WebP, or HEIC image and select **Analyze**. Inspect accepted boxes and choose **Add candidate to review** when a highlighted proposal is correct. Supported nominal capacity, measured lightness and estimated fill are shown with their evidence. To review fill, select **Mark fill on image**, click the two numbered graduations that bracket the piston and the liquid-facing edge, enter the values, and save the review. The ZIP contains the observation JPEG, the optional reference image, and JSON with the original predictions and revision history.

The first analysis checks model weights and release evidence by SHA-256. The app and CLI work offline after setup. Uploaded files are saved in the local observation database, allowing inspection of native-resolution crops. ZIP reports include the original bytes when available and record their SHA-256. Apparent fill is an image-based estimate. [RF-DETR verification and recovery](docs/rfdetr_verification.md)

The review panel can open a candidate's native-resolution image crop before adoption. Crops also follow a box drawn or edited in the current draft. The inspection leaves the saved observation at its current revision.

To restore the released weights and three packaged Python environments in a clean checkout at the bundle’s recorded Git revision, run the command below. The bundle also includes the base Python interpreter. Linux x86-64 and compatible host libraries are required. The Intel device is used by the verified SAM worker. [Runtime bundle documentation](docs/local_runtime_bundle.md)

```bash
python3 scripts/runtime_bundle.py restore /path/to/medsyringe-runtime
HF_HUB_OFFLINE=1 .venv/bin/python -m uvicorn medsyringe.app:app --host 127.0.0.1 --port 8765
```

```bash
HF_HUB_OFFLINE=1 uv run --locked python -m medsyringe.pipeline image.HEIC --ocr --output result.json
uv run --locked python -m pytest -q
```

Use `MEDSYRINGE_DEVICE=cpu` with the API launcher, or `--device cpu` with the image CLI, to run detection and all SAM operations on the CPU. The same settings accept `xpu` with the pinned Intel environment. [Local runtime and device selection](docs/local_runtime_bundle.md)

Use `bash scripts/serve_intel.sh` on the delivered Intel Arc machine for the benchmarked XPU runtime. To restore the launcher in a separate checkout, first restore the XPU environment described in the local worker setup, then install the app dependencies there.

```bash
uv pip install --python data/sam3/xpu-env/bin/python fastapi==0.135.1 starlette==1.10.0 uvicorn==0.41.0 python-multipart==0.0.22
bash scripts/serve_intel.sh
```

## Detection optimization

The app applies a fixed detection configuration. A local SigLIP 2 crop check sends weak accepted detections to human review. On independently sourced MedBin photos, the gate removed 15 false positives and preserved all 11 existing matches. On 26 additional negative scenes, correct empty results rose from 20 to 25. Recall on the fresh mixed Commons group remains 15/22. On Intel Arc the application shares one SAM worker across proposals, corroboration and liquid measurements. [Cross-domain detection report](docs/detection_generalization.md) · [Offline runtime qualification](docs/local_runtime_bundle.md)

Earlier alternatives and comparisons are described in the [detector optimization report](docs/detector_optimization.md), the [lookalike precision study](docs/precision_study.md), the [supervised training study](docs/supervised_training.md), the [hybrid and public-data experiments](docs/hybrid_experiments.md), and the [RF-DETR verification study](docs/rfdetr_verification.md).

## Prepare the supplied source

```bash
uv run --locked python -m medsyringe.data audit --source "/path/to/Images"
uv run --locked python -m scripts.source_profiles --source "/path/to/Images"
uv run --locked python -m medsyringe.data coco
```

Preparation reads the originals and writes oriented RGB PNG files under `data/original`. It records input hashes and excludes the exact byte duplicate. The prepared images unlock the **Review draft** queue and the approved COCO export. Use a separate experiment directory to re-train and re-test while preserving the delivered freeze. See the engineering guide for the complete sequence.
