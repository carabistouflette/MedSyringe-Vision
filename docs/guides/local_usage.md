# Run and review locally

Use Linux x86-64, Python 3.12, uv, and curl. Install ExifTool for source preparation and Tesseract with English language data for optional marking extraction. Exact Python dependencies are in `uv.lock`. The delivered workspace already contains the released checkpoints and worker environments. For a new checkout, restore the trained TorchScript recovery model and the sealed SAM and OCR workers before launching. The setup guides list their files, revisions and checksums. [RF-DETR recovery setup](../studies/detection/rfdetr_verification.md) · [SAM and OCR worker setup](../studies/measurement/sam3_markings_liquid.md)

```bash
uv sync --locked --python 3.12
bash scripts/setup/fetch_models.sh
HF_HUB_OFFLINE=1 uv run --locked uvicorn medsyringe.app:app --host 127.0.0.1 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Upload a JPEG, PNG, WebP, or HEIC image and select **Analyze**. Inspect accepted boxes and choose **Add candidate to review** when a highlighted proposal is correct. Supported nominal capacity, measured lightness and estimated fill are shown with their evidence. To review fill, select **Mark fill on image**, click the two numbered graduations that bracket the piston and the liquid-facing edge, enter the values, and save the review. The ZIP contains the observation JPEG, the optional reference image, and JSON with the original predictions and revision history.

The first analysis checks model weights and release evidence by SHA-256. The app and CLI work offline after setup. Uploaded files are saved in the local observation database, allowing inspection of native-resolution crops. ZIP reports include the original bytes when available and record their SHA-256. Apparent fill is an image-based estimate. [RF-DETR verification and recovery](../studies/detection/rfdetr_verification.md)

Choose **Count only** when you need detections, the scene count, and review candidates. This retains detector corroboration and skips the crop measurements. In the CLI, select it with `--count-only`. [Local inference qualification](inference_qualification.md)

The review panel can open a candidate's native-resolution image crop before adoption. If the proposal overlaps one draft object, it can replace that object's box while keeping the object count. The review links its geometry to the source proposal and clears measurements from the former crop. Crops also follow boxes drawn or edited in the current draft.

To restore the released weights and three packaged Python environments in a clean checkout at the bundle’s recorded Git revision, run the command below. The bundle also includes the base Python interpreter. Linux x86-64 and compatible host libraries are required. The Intel device is used by the verified SAM worker. [Runtime bundle documentation](local_runtime_bundle.md)

```bash
python3 scripts/setup/runtime_bundle.py restore /path/to/medsyringe-runtime
HF_HUB_OFFLINE=1 .venv/bin/python -m uvicorn medsyringe.app:app --host 127.0.0.1 --port 8765
```

```bash
HF_HUB_OFFLINE=1 uv run --locked python -m medsyringe.pipeline image.HEIC --ocr --output result.json
uv run --locked python -m pytest -q
```

Use `MEDSYRINGE_DEVICE=cpu` with the API launcher, or `--device cpu` with the image CLI, to run detection and all SAM operations on the CPU. The same settings accept `xpu` with the pinned Intel environment. [Local runtime and device selection](local_runtime_bundle.md)

Use `bash scripts/setup/serve_intel.sh` on the delivered Intel Arc machine for the benchmarked XPU runtime. To restore the launcher in a separate checkout, first restore the XPU environment described in the local worker setup, then install the app dependencies there.

```bash
uv pip install --python data/sam3/xpu-env/bin/python fastapi==0.135.1 starlette==1.10.0 uvicorn==0.41.0 python-multipart==0.0.22
bash scripts/setup/serve_intel.sh
```

## Prepare the supplied source

```bash
uv run --locked python -m medsyringe.data audit --source "/path/to/Images"
uv run --locked python -m scripts.datasets.source_profiles --source "/path/to/Images"
uv run --locked python -m medsyringe.data coco
```

Preparation reads the originals and writes oriented RGB PNG files under `data/original`. It records input hashes and excludes the exact byte duplicate. The prepared images unlock the **Review draft** queue and the approved COCO export. Use a separate experiment directory to re-train and re-test while preserving the delivered freeze. See the [engineering guide](engineering.md) for the complete sequence.
