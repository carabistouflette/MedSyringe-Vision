# Stroke-guided capacity, piston landmarks, and additive detection

## Release status

The application retries unread nominal capacities on high-resolution scale crops. The reader uses local PP-OCRv6 weights, applies a fixed acceptance rule, and records the printed evidence in the observation. Existing detector and apparent-fill decisions retain their release criteria.

| Output | Status | Evidence |
| --- | --- | --- |
| Stroke-guided nominal capacity and visual scale | Implemented and validated on available controls | Six correct additions in 59 crop replays, one addition through full-image inference |
| Piston landmark localization | Implemented but not fully validated | Fit on 15 annotated examples, checked on seven development and six regression images |
| Additional automatic fill estimates | Implemented but not fully validated | Three in-bracket additions and one out-of-bracket addition, so the candidate remains experimental |
| Additive recovery of weak detections | Implemented but not fully validated | Crop and SAM policies failed their development criteria |
| Piston-bounded apparent lightness | Implemented but not fully validated | One false clear report cleared, five supported calibration measurements lost |
| Absolute opacity | Not applicable with available data | No registered clear reference at a comparable exposure |

The experiments used the existing JPEGs, prepared crops, public controls, and local weights. The capture is from one session. Previously examined images serve as regression evidence. The detector study reapplied acquisition-group, source, author and image-hash checks before fit and selection. Landmark fit, development and replay use disjoint acquisition groups. The full orange 60 mL group was kept outside landmark fitting.

## Capacity and printed scale

Observed numeric anchors estimate the arrangement of the scale. Dark transverse strokes provide alternative locations for short, high-resolution reading windows. Tiny and medium PP-OCRv6 inspect the actual crops. The layout supplies window locations. Released capacity requires directly recognized number and unit evidence from the printed scale.

Twenty-seven configurations compared recognition backbones and their combination, three confidence values, and three stroke-strength values on 31 amber development crops. The selected policy added one correct capacity and six fill values in their visual reference brackets. [Frozen selection and development cases](../../../artifacts/fill/graduation_ocr_selection.json)

| Regression group | Images | Previous nominal capacity reads | Correct new reads | Wrong new reads |
| --- | ---: | ---: | ---: | ---: |
| Orange 20 mL | 35 | 24 | 4 | 0 |
| Orange 60 mL | 10 | 3 | 1 | 0 |
| Clear 20 mL | 14 | 8 | 1 | 0 |
| Total | 59 | 35 | 6 | 0 |

Four additional fill estimates were emitted on the colored regression groups. IMG_3296 was read as 18 mL against a 19–22 mL visual bracket. The automatic fill extension failed. The deployed extension adds only capacity and visual scale after the existing fill decision.

Five full-JPEG controls ran through the production detector and Intel primary grounder. IMG_3294 gained a read of 20 mL. IMG_3316 kept its 16 mL fill in its original and reversed orientations. The first triggered retry took 5.94 seconds including cold worker startup. All four controls with existing capacities bypassed the retry. The Chromium check verified the new capacity, its printed evidence in the ZIP and preserved fill abstention on IMG_3294. The released CPU command-line path also emitted 20 mL for IMG_3294. [Crop audit](../../../artifacts/fill/graduation_ocr_regression.json) · [Live release evidence](../../../artifacts/fill/graduation_capacity_runtime.json) · [Browser check](../../../artifacts/fill/graduation_capacity_browser_smoke.json)

## Piston and lightness

The new annotations mark the front shoulder, the liquid-facing piston rim and the rear flange on 28 existing crops. The normalized coordinates come from manual clicks on enlarged contact sheets and are approximate. Three clear photos were included in fit to expose the landmark heads to a visible rim without colored liquid.

Separate logistic heads use frozen DINOv2 patch features. Joint prediction constrains their alignment and order along the barrel. A high-resolution grayscale check verifies a dark transverse edge around the proposed rim. The 24-setting ablation compares dense features, dense features with Lab, confidence gating and rim refinement. The selected model answered six of seven development references within 6% of barrel length. On the orange 60 mL replay it answered five of six, with median piston error 2.24% of barrel length and maximum 6.02%. These errors are relative to the approximate manual landmarks.

The landmark-only projection added no fill values under the current numbered-graduation rule. Agreement between the color-based edge and the new landmark prediction added three replies on the colored replay. Two agreed with the visual brackets. The incorrect IMG_3296 reading survived the agreement rule. [Landmark annotations](../../../artifacts/fill/piston_landmarks.tsv) · [Piston replay](../../../artifacts/fill/piston_regression.json) · [Agreement replay](../../../artifacts/fill/piston_ocr_agreement_regression.json)

Restricting pixels to the verified interval before the piston removed the single false report on the 14 clear photos. The six public controls stayed free of false reports. The supported calibration measurements fell from 58 to 53 of 68. Among supported measurements the median apparent-lightness discrepancy moved from 2.38 to 2.17 L*. A conservative veto for very short verified columns removed no results. Both appearance variants remain experimental. [Bounded region](../../../artifacts/expanded/lumen_appearance_regression.json) · [Short-column veto](../../../artifacts/expanded/short_lumen_veto_regression.json)

## Additive detection and SAM masks

Published detections and vetoes were retained. Recovery ran only when there was no accepted detection. At IoU 0.50, every previously matched instance had to remain matched and the candidate had to add no false positives. The available development groups had no positive scenes that the released detector left naturally empty. For the development simulation, accepted predictions were hidden and the search operated on real below-cutoff proposals. Naturally empty frames remained eligible for false-positive counting.

Twenty-five crop rescue configurations compared frozen DINOv2 verification, SAM corroboration and grouped logistic feature models. The best diagnostic setting recovered 43 matches with eight false positives in the simulation. It made no additions in either replay. The mask experiment ran text-only and box-prompted SAM 3 on 51 padded development crops and compared 24 policies. Its best diagnostic setting matched 42 targets with one false positive. No candidate passed selection. [Crop-rescue trials](../../../artifacts/expanded/additive_recovery_selection.json) · [Mask trials](../../../artifacts/expanded/additive_mask_selection.json) · [Frozen replay](../../../artifacts/expanded/additive_mask_regression.json)

After freezing selection, a separate diagnostic ran both SAM prompts on each of the three known public misses. Box-prompted mask IoUs were 0.929, 0.689 and 0.723. The corresponding text-only scores were 0.882, 0.143 and 0.141. A strong prompted mask can localize the hint without independently corroborating the object class. This inspection only explains the missed cases. [Three-case diagnostic](../../../artifacts/expanded/public_miss_mask_probe.json)

## Reproduction

Run from a separate repository copy against the configured read-only image source. Preserve the recorded freeze and model files. For a fresh selection, collect fit or development first, run the corresponding selection stage, and then collect or evaluate the regression groups.

```bash
source_dir="/path/to/Images_fil_rougev1"
export HF_HUB_OFFLINE=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2
for group in development holdout holdout-60 clear; do
  data/ocr-v6/env/bin/python -m scripts.experiments.measurement.graduation_ocr_study collect --source "$source_dir" --split "$group"
done
.venv/bin/python -m scripts.experiments.measurement.graduation_ocr_study audit --source "$source_dir"
.venv/bin/python -m scripts.experiments.measurement.piston_study audit --source "$source_dir"
.venv/bin/python -m scripts.experiments.measurement.fill_agreement audit --source "$source_dir"
data/sam3/xpu-env/bin/python -m scripts.experiments.measurement.piston_study appearance
.venv/bin/python -m scripts.experiments.baselines.recovery_study audit
data/sam3/xpu-env/bin/python -m scripts.experiments.baselines.recovery_mask_study audit
```

The application and the production CLI in `medsyringe.pipeline` load the same capacity release. To recheck its live release in an experiment copy, run `scripts.experiments.measurement.graduation_capacity_release` with the configured JPEG directory after collecting the crop audit.
