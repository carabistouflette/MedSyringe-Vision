# Hybrid detector experiments without additional original photos

The deployed detector's detections were preserved in every additive candidate. Tiled or independently corroborated proposals could be appended at the fixed operating score. Original detections retain their original scores and order. Candidate projection rejects boxes clipped at internal tile edges, and suppression applies only to the appended proposals. We check individual matched reference boxes at IoU ≥0.50 as a separate safeguard.

## Experiments

| Experiment | Role | Evidence |
| --- | --- | --- |
| SSDLite whole image, 66% overlapping tiles, and both combined | Compare a locally supervised auxiliary branch at multiple scales | [Additive calibration and missed box analysis](../../../artifacts/additive_calibration.json) |
| YOLOE-26 Nano visual prompts from an original and a public syringe exemplar | Compare a recent open-vocabulary detector by itself and as an added proposal source | [Visual prompt calibration](../../../artifacts/visual_prompt_selection.json) · [Public-photo evaluation](../../../artifacts/visual_prompt_audit.json) |
| SSDLite and YOLOE joint corroboration | Compare two-model agreement without requiring an additional full-frame Grounding DINO match | [Cross-model study](../../../artifacts/cross_prompt_study.json) |
| Grounding DINO zooms around unresolved low-score proposals | Bound high-resolution compute and check candidate recovery | [Zoom study](../../../artifacts/zoom_study.json) |
| Train-only small-object mosaics | Four quadrants, two positive and two negative sources, precisely translated annotations | [Mosaic experiment](../../../artifacts/mosaic_study.json) |
| Public phlebotomy transfer | Matched training updates on existing data or a 50% mix of existing and public data | [Transfer study](../../../artifacts/public_transfer_study.json) |

The new evaluation includes 13 Wikimedia Commons photos from 13 recorded authors, with 12 reference boxes. Five source photos were excluded for known author overlap. We froze the manifest and the development-selected visual exemplars and settings before scoring the new set. The manifest carries source links, creators, terms and exact hashes. Reference boxes are single-pass visual annotations. The sample includes glass, historic and prefilled syringes. Its metric estimates do not replace a target-domain clinical evaluation. Previously inspected sets are development or named historical checks. Possible overlap with generic pretraining data remains unknown. [Photo evidence](../../../artifacts/additive_external_manifest.json)

## Public training source

Zenodo record 16924786 provides a labeled simulated phlebotomy dataset with CC BY 4.0 terms. The source article reports 11,884 frames. The experiment checks the download against Zenodo's file digest and scans only the supplied training directory, containing 8,319 candidate images. It samples 120 syringe-positive and 40 syringe-negative images reproducibly, removes repeated export parents and near hashes, converts syringe polygons to boxes, and records the sampled hashes and original ZIP members. All selected frames are a single public training group. With the training video grouping unverified, the native validation and test directories are excluded from the project evaluation. The original photos remain local, and no new user photos were needed. [Source paper](https://arxiv.org/abs/2602.04624) · [Zenodo record and license](https://doi.org/10.5281/zenodo.16924786) · [Sample manifest](../../../artifacts/zenodo_training_manifest.json)

## Measured outcomes

| Comparison | Observations |
| --- | --- |
| Tiled SSDLite and YOLOE additions | Across 72 SSDLite settings, 270 YOLOE settings and 32 two-model corroboration settings, no configuration recovered the development miss. The tiled SSDLite raw candidate reached IoU 0.507 versus 0.473 for the full view and survived duplicate suppression. The remaining acceptance checks excluded it. |
| YOLOE visual prompts on the 13 new public photos | The deployed model had 10 TP and 3 FP. YOLOE alone had 2 TP and 2 FP at its development-selected point. Recall was 0.833 and 0.167 respectively. |
| Proposal-guided zoom on 27 previously inspected scenes | The best F1 setting recovered one extra syringe and introduced one additional false positive, from 14 TP / 9 FP to 15 TP / 10 FP. All original instance matches were preserved. Precision declined from 0.609 to 0.600, and the predefined calibration criterion requiring both a recovery and unchanged precision was unmet. On the 13 already inspected public photos, this fixed zoom point left the 10 TP and 3 FP unchanged. [Zoom regression](../../../artifacts/zoom_external_audit.json) |
| Mosaics | The strongest mosaic epoch retrieved 4/15 of the previously inspected Open Images boxes, versus 3/15 for the prior non-mosaic SSDLite recipe. The released model retrieved 9/15. |
| Matched public transfer | With 100 updates per arm, the 50% public training mix changed calibration from 4 TP / 3 FP to 3 TP / 0 FP, and the previously inspected 13 public scenes from 1 TP / 0 FP to 0 TP / 0 FP at the respective calibration thresholds. |

The deployed model remains active. The new candidates do not meet the recall and precision release criteria. The clearest useful signal is the additional recovery from targeted zoom. The public dataset makes a larger supervised experiment possible without additional original photos, but the tested 50% blend cannot establish a benefit. The [RF-DETR transfer study](rfdetr_transfer.md) evaluates this separate pretraining phase followed by adaptation to project boxes, with a reserve built from unseen Zenodo clips. The [research review](hybrid_research.md) discusses RF-DETR, promptable segmentation, and synthetic render controls as candidates for that comparison.

## Run reproducibly

The project's locked environment runs the supervised, additive, ROI and mosaic branches without adding any production dependency.

```bash
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.additive_study select
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.zoom_study
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.zoom_study --audit
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.mosaic_study
```

For the transfer experiment, download the official Zenodo ZIP to `data/zenodo/phlebotomy.zip`, then run `HF_HUB_OFFLINE=1 uv run --locked python -m scripts.public_transfer_study`. The ZIP itself remains a local input. The YAML class name is resolved from the archive and the checksum is checked before preparation.

YOLOE was run locally in a separate experimental environment using Ultralytics 8.4.40. The official segmentation checkpoint is in `data/models/yoloe/yoloe-26n-seg.pt` on this machine. The artifact records its full SHA-256. The package and weights are excluded from the application install and are not redistributed here. Disable Ultralytics synchronization and use locally prepared assets for this test. [Official YOLOE-26 paper](https://arxiv.org/abs/2602.00168) · [Ultralytics visual prompt usage](https://docs.ultralytics.com/models/yoloe/#predict-usage) · [Ultralytics license terms](https://www.ultralytics.com/license)

```bash
mkdir -p data/models/yoloe data/yoloe-home
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt \
  -o data/models/yoloe/yoloe-26n-seg.pt
uv venv --python "$(pwd)/.venv/bin/python" data/yoloe-env
uv pip install --python data/yoloe-env/bin/python --no-deps \
  ultralytics==8.4.40 psutil==7.2.2 polars==2.1.0 polars-runtime-32==2.1.0 ultralytics-thop==2.2.5
export PYTHONPATH="$(pwd)/data/yoloe-env/lib/python3.12/site-packages:$(pwd)"
export YOLO_CONFIG_DIR="$(pwd)/data/yoloe-home"
export HF_HUB_OFFLINE=1
.venv/bin/python -m scripts.visual_prompt_study select
.venv/bin/python -m scripts.visual_prompt_study audit
.venv/bin/python -m scripts.cross_prompt_study
```

[Recent small-object and low-data methods, with source and license notes](hybrid_research.md)
