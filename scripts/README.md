# Script index

Run the commands from the repository root with the locked Python environment. [Local setup](../docs/guides/local_usage.md)

| Directory | Contents |
| --- | --- |
| [`setup/`](setup/) | Model installation, runtime bundles and launchers |
| [`checks/`](checks/) | Smoke checks, benchmarks and runtime parity |
| [`datasets/`](datasets/) | Source preparation and evidence export |
| [`workers/`](workers/) | Subprocesses launched by the application |
| [`experiments/`](experiments/) | Studies grouped by topic. Some model helpers are also imported by workers. |

## Run and verify the application

| Task | Entry point |
| --- | --- |
| Fetch pinned model assets | [`bash scripts/setup/fetch_models.sh`](setup/fetch_models.sh) |
| Launch on the configured Intel Arc machine | [`bash scripts/setup/serve_intel.sh`](setup/serve_intel.sh) |
| Package or restore the runtime | [`python -m scripts.setup.runtime_bundle`](setup/runtime_bundle.py) · [Bundle guide](../docs/guides/local_runtime_bundle.md) |
| Exercise the API with an image | [`python -m scripts.checks.smoke_app IMAGE`](checks/smoke_app.py) |
| Exercise the review UI with an image | [`python -m scripts.checks.browser_smoke IMAGE`](checks/browser_smoke.py) · [Browser prerequisites](../docs/guides/engineering.md#re-run-in-a-fresh-experiment-directory) |
| Compare inference schedules and latency | [Benchmark script](checks/benchmark_inference.py) · [Qualification procedure](../docs/guides/inference_qualification.md) |

## Prepare data and reproduce studies

Use the linked procedure to find the inputs, run order and output paths for each study.

| Task | Start here |
| --- | --- |
| Prepare Fil Rouge source images and export references | [Source preparation](../docs/guides/local_usage.md#prepare-the-supplied-source) · [Source profiles](datasets/source_profiles.py) |
| Prepare public images and review annotations | [Dataset card](../docs/datasets/public_syringe_dataset.md) · [Acquisition and augmentation protocol](../docs/datasets/public_training_protocol.md) |
| Reproduce detector selection and holdout evaluation | [Engineering and reproduction](../docs/guides/engineering.md#re-run-in-a-fresh-experiment-directory) |
| Inspect cross-domain detector audits | [Detection generalization](../docs/studies/detection/detection_generalization.md) · [Audit runner](experiments/detector_training/detection_hillclimb_audit.sh) |
| Find the independent crowd audit | [Frozen protocol](../docs/crowd_next_audit_protocol.md) · [Audit script](experiments/crowds/crowd_next_audit.py) |
| Evaluate fill, OCR or liquid fields | [Measurement studies](../docs/README.md#measurement-studies) |
| Regenerate figures and provenance ledgers | [Report figures](experiments/baselines/report_figures.py) · [Provenance](checks/provenance.py) |

## Worker entry points

The application launches these workers using their configured environments. The relevant setup guide documents their assets and dependencies.

| Worker | Entry point |
| --- | --- |
| Shared SAM | [Launcher](workers/shared_sam_worker.sh) · [Worker](workers/shared_sam_worker.py) |
| Expanded pipeline on XPU | [Launcher](workers/expanded_worker_xpu.sh) · [Worker](workers/expanded_worker.py) |
| Appearance verification | [Launcher](workers/appearance_worker.sh) · [Worker](workers/appearance_worker.py) |
| Scale reading | [Launcher](workers/scale_worker.sh) · [Worker](workers/scale_worker.py) |
| Local models | [Worker](workers/local_model_worker.py) |

## Replaying historical results

Study results can bind source paths and contents by SHA-256. For a frozen replay, check out the revision recorded by that study in a separate worktree and use the commands and inputs for that revision. Commit `130e379` has the layout immediately before the source reorganization and the historical crowd audit protocol keeps its original commands. For a new run, use the current paths in a separate experiment workspace. See the [engineering guide](../docs/guides/engineering.md#re-run-in-a-fresh-experiment-directory).
