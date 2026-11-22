# MedSyringe Vision

MedSyringe Vision is a local research application for syringe localization, scene counts, visual review, and reference checks. It keeps observations and human revisions in a local SQLite database. Its detector runs from pinned weights on CPU or Intel Arc, and its reports contain the image evidence and original model output.

## Quick start

Run these commands from the repository root on Linux x86-64 with Python 3.12, uv and curl. For a new checkout, follow the [model and worker setup](docs/guides/local_usage.md) to restore the RF-DETR, SAM and OCR assets.

```bash
uv sync --locked --python 3.12
bash scripts/fetch_models.sh
HF_HUB_OFFLINE=1 uv run --locked uvicorn medsyringe.app:app --host 127.0.0.1 --port 8765
```

Open [the local application](http://127.0.0.1:8765) to inspect detections and review a scene. For a single image from the command line, use the same pipeline directly.

```bash
HF_HUB_OFFLINE=1 uv run --locked python -m medsyringe.pipeline image.HEIC --count-only --output result.json
```

See the [usage guide](docs/guides/local_usage.md) for full analysis, image review, exports and Intel Arc setup. See the [offline runtime bundle](docs/guides/local_runtime_bundle.md) to restore a released environment.

## Work on the project

| I want to… | Start here |
| --- | --- |
| Change the web app, inference pipeline or local storage | [Code map](docs/guides/engineering.md#code-map) |
| Run checks, reproduce an experiment or find a worker | [Script index](scripts/README.md) |
| Prepare source images or work on annotations | [Source preparation](docs/guides/local_usage.md#prepare-the-supplied-source) · [Dataset documentation](docs/README.md#datasets-and-annotation) |
| Understand measured performance and limitations | [Release results](docs/reference/release_results.md) · [Product validation](docs/guides/product_validation.md) |
| Find a report, study or protocol | [Documentation index](docs/README.md) |

```bash
uv run --locked python -m pytest -q
```

Tests are grouped under [`tests/`](tests/) by application, runtime, detection, measurement and datasets. Run a single group with `uv run --locked python -m pytest -q tests/measurement`, for example.

[`artifacts/`](artifacts/) holds the versioned evaluation evidence and release manifests. Local images, checkpoints, worker environments and analysis outputs live under `data/`.
