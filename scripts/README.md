# Script index

Run the commands from the repository root with the locked Python environment. [Local setup](../docs/guides/local_usage.md)

## Run and verify the application

| Task | Entry point |
| --- | --- |
| Fetch pinned model assets | [`bash scripts/fetch_models.sh`](fetch_models.sh) |
| Launch on the configured Intel Arc machine | [`bash scripts/serve_intel.sh`](serve_intel.sh) |
| Package or restore the runtime | [`python -m scripts.runtime_bundle`](runtime_bundle.py) · [Bundle guide](../docs/guides/local_runtime_bundle.md) |
| Exercise the API with an image | [`python -m scripts.smoke_app IMAGE`](smoke_app.py) |
| Exercise the review UI with an image | [`python -m scripts.browser_smoke IMAGE`](browser_smoke.py) · [Browser prerequisites](../docs/guides/engineering.md#re-run-in-a-fresh-experiment-directory) |
| Compare inference schedules and latency | [Benchmark script](benchmark_inference.py) · [Qualification procedure](../docs/guides/inference_qualification.md) |

## Prepare data and reproduce studies

Use the linked procedure to find the inputs, run order and output paths for each study.

| Task | Start here |
| --- | --- |
| Prepare Fil Rouge source images and export references | [Source preparation](../docs/guides/local_usage.md#prepare-the-supplied-source) · [Source profiles](source_profiles.py) |
| Prepare public images and review annotations | [Dataset card](../docs/datasets/public_syringe_dataset.md) · [Acquisition and augmentation protocol](../docs/datasets/public_training_protocol.md) |
| Reproduce detector selection and holdout evaluation | [Engineering and reproduction](../docs/guides/engineering.md#re-run-in-a-fresh-experiment-directory) |
| Inspect cross-domain detector audits | [Detection generalization](../docs/studies/detection/detection_generalization.md) · [Audit runner](detection_hillclimb_audit.sh) |
| Re-run the independent crowd audit | [Frozen protocol](../docs/crowd_next_audit_protocol.md) · [Audit script](crowd_next_audit.py) |
| Evaluate fill, OCR or liquid fields | [Measurement studies](../docs/README.md#measurement-studies) |
| Regenerate figures and provenance ledgers | [Report figures](report_figures.py) · [Provenance](provenance.py) |

## Worker entry points

The application launches these workers using their configured environments. The relevant setup guide documents their assets and dependencies.

| Worker | Entry point |
| --- | --- |
| Shared SAM | [Launcher](shared_sam_worker.sh) · [Worker](shared_sam_worker.py) |
| Expanded pipeline on XPU | [Launcher](expanded_worker_xpu.sh) · [Worker](expanded_worker.py) |
| Appearance verification | [Launcher](appearance_worker.sh) · [Worker](appearance_worker.py) |
| Scale reading | [Launcher](scale_worker.sh) · [Worker](scale_worker.py) |
| Local models | [Worker](local_model_worker.py) |

Release manifests and experiment protocols can bind script paths and contents by SHA-256. Use the matching source revision and inputs when replaying historical results. See the [engineering guide](../docs/guides/engineering.md) for an isolated workspace procedure.
