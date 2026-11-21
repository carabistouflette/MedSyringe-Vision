# MedSyringe Vision

MedSyringe Vision is a local research application for syringe localization, scene counts, visual review, and reference checks. It keeps observations and human revisions in a local SQLite database. Its detector runs from pinned weights on CPU or Intel Arc, and its reports contain the image evidence and original model output.

The original single-view baseline found **10 of 15 reference objects** in a small external test of ten Open Images scenes. It made **zero false-positive detections** at the selected cutoff. Both missed scenes contained multiple syringes. Visual scale and barrel color are experimental suggestions. Their diagnostics use the single supplied photo session, so they have no independent performance estimate.

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

To restore the released weights and three packaged Python environments in a clean checkout at the bundle’s recorded Git revision, run the command below. The bundle also includes the base Python interpreter. Linux x86-64 and compatible host libraries are required. The Intel device is used by the verified SAM worker. [Runtime bundle documentation](docs/local_runtime_bundle.md)

```bash
python3 scripts/runtime_bundle.py restore /path/to/medsyringe-runtime
HF_HUB_OFFLINE=1 .venv/bin/python -m uvicorn medsyringe.app:app --host 127.0.0.1 --port 8765
```

```bash
HF_HUB_OFFLINE=1 uv run --locked python -m medsyringe.pipeline image.HEIC --ocr --output result.json
uv run --locked python -m pytest -q
```

Use `bash scripts/serve_intel.sh` on the delivered Intel Arc machine for the benchmarked XPU runtime. To restore the launcher in a separate checkout, first restore the XPU environment described in the local worker setup, then install the app dependencies there.

```bash
uv pip install --python data/sam3/xpu-env/bin/python fastapi==0.135.1 starlette==1.10.0 uvicorn==0.41.0 python-multipart==0.0.22
bash scripts/serve_intel.sh
```

## Detection optimization

The app uses one validated detection configuration for all uploads. A local SigLIP 2 crop check sends weak accepted detections to human review. The interface requires no profile or threshold setting. The latest release retained every previously matched reference and all 177 exact Fil Rouge counts in the full-capture replay. On 26 new negative scenes, correct empty results rose from 20 to 25. Recall on the fresh mixed external cohort remains 15/22. [Cross-domain detection report](docs/detection_generalization.md) · [Live qualification](artifacts/detection_hillclimb/precision_only_live_check.json)

Earlier alternatives and comparisons are described in the [detector optimization report](docs/detector_optimization.md), the [lookalike precision study](docs/precision_study.md), the [supervised training study](docs/supervised_training.md), the [hybrid and public-data experiments](docs/hybrid_experiments.md), and the [RF-DETR verification study](docs/rfdetr_verification.md).

## Prepare the supplied source

```bash
uv run --locked python -m medsyringe.data audit --source "/path/to/Images"
uv run --locked python -m scripts.source_profiles --source "/path/to/Images"
uv run --locked python -m medsyringe.data coco
```

Preparation reads the originals and writes oriented RGB PNG files under `data/original`. It records input hashes and excludes the exact byte duplicate. The prepared images unlock the **Review draft** queue and the approved COCO export. Use a separate experiment directory to re-train and re-test while preserving the delivered freeze. See the engineering guide for the complete sequence.
