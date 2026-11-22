# MedSyringe Vision — Expanded Fil rouge evaluation

## Abstract

This study revisits detection, visible markings and liquid appearance on 177 JPEGs from the expanded Fil Rouge capture. Two learned hybrids improved aggregate precision by combining Grounding DINO, local RF-DETR, and SAM 3 segmentation evidence. Each lost at least one previously detected public instance and was rejected under the per-instance recall requirement. Two-view OCR corroboration increased correct nominal-capacity readings from 38 to 44 on 68 calibration crops. A subsequent evidence rule removed a false capacity on the clear 20-scale regression. A color-aware liquid localization model added one supported measurement in calibration and failed the color controls required for release. At this stage only the validated OCR changes were activated. A later study released a separate DINOv2 verifier after preserving every previous match. [Original release assessment](../../../artifacts/expanded/release_assessment.json) · [Follow-up experiments](improvement_regression.md)

## Data and protocol

The delivery adds 103 distinct observations and 74 visually matched JPEG re-exports of previously studied HEICs. All 177 frames belong to one acquisition session. Each shows one syringe. The expanded material improves coverage of uncolored interiors, warm amber and orange appearances, viewpoint changes, glare, shadows and clipping. The capture contains no syringe-free scene. External author-disjoint distractors supply a separate precision control. [Read-only inventory](../../../artifacts/expanded/inventory.json)

The split assigns whole acquisition blocks, including 95 fit frames, 68 calibration frames and 14 held-out views of an uncolored 20-scale syringe. All 14 held-out frame IDs were absent from the previous HEIC delivery. The old and expanded files can show the same physical syringes, and the study lacks a second session. The held-out block measures transfer to a new setup in the supplied capture. The separate public reserve contains seven syringe images and fourteen distractor images from prior deployment checks. Its files and their author groups and close perceptual hashes were excluded from the current fitting and calibration. Public scores describe a regression cohort and do not estimate population accuracy. [Split audit](../../../artifacts/expanded/split_review.json)

Local visual references cover the 177 object boxes and 146 colored-interior centerlines. The first label set remains frozen for the primary comparisons. Four imprecise boxes found during failure analysis were independently redrawn on enlarged source photos and stored in an adjudicated derivative. This produced a post-score sensitivity check, not a revised training or test result. Independent human ratification of the draft references remains pending. [Frozen annotations](../../../artifacts/expanded/annotations.json) · [Centerlines](../../../artifacts/expanded/liquid_traces.json) · [Adjudicated boxes](../../../artifacts/expanded/annotations_adjudicated.json)

## Detection experiments

The control uses the installed two-view Grounding DINO path, verified RF-DETR additions, and SAM 3's empty-scene fallback. The new hybrid collects full-image SAM 3 boxes and masks, full-image RF-DETR queries, and ranked proposals from both Grounding DINO orientations even when an object has already been accepted. Candidate representations combine detector confidences, cross-source overlaps, shape statistics, and mask geometry. They carry no camera metadata, filename, acquisition group, or class prevalence at inference.

The first search compared 492 settings with per-source baselines, simple unions, rule-based agreement, logistic regression and histogram gradient boosting. It varied weighted coordinates, mask boxes, anchor boxes, additive retention, group weights, overlap labels and score cutoffs. The fit set supplied 1,365 proposals. No additive setting passed the acceptance constraints while adding a match on calibration. The first winner was unweighted logistic reranking with an IoU 0.50 training target. Its two public matching losses on fit triggered an additional 83-policy search on fit and calibration alone, yielding twelve eligible settings. The second winner used burst weighting, an IoU 0.70 training target and mask-supported boxes. Calibration used the numerically checked XPU Grounding DINO implementation, and the evaluation replays used the CPU control. [Initial trials](../../../artifacts/expanded/detector_trials.json) · [Recall-constrained trial archive](../../../artifacts/expanded/detector_recall_freeze.json)

| Cohort | Detector | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed calibration, 87 photos | Installed | 68 | 23 | 4 | 74.7% | 94.4% | 83.4% | 86.6% | 50.0% |
| Mixed calibration, 87 photos | Initial hybrid | 68 | 7 | 4 | 90.7% | 94.4% | 92.5% | 90.2% | 57.3% |
| Mixed calibration, 87 photos | Recall-constrained hybrid | 68 | 8 | 4 | 89.5% | 94.4% | 91.9% | 90.6% | 56.3% |
| Clear 20-scale block, 14 photos | Installed | 13 | 1 | 1 | 92.9% | 92.9% | 92.9% | 92.5% | 47.2% |
| Clear 20-scale block, 14 photos | Initial hybrid | 12 | 0 | 2 | 100.0% | 85.7% | 92.3% | 92.1% | 47.5% |
| Clear 20-scale block, 14 photos | Recall-constrained hybrid | 13 | 1 | 1 | 92.9% | 92.9% | 92.9% | 93.2% | 47.3% |
| Public regression, 21 photos | Installed | 4 | 16 | 3 | 20.0% | 57.1% | 29.6% | 32.7% | 27.2% |
| Public regression, 21 photos | Initial hybrid | 4 | 2 | 3 | 66.7% | 57.1% | 61.5% | 64.1% | 56.1% |
| Public regression, 21 photos | Recall-constrained hybrid | 4 | 4 | 3 | 50.0% | 57.1% | 53.3% | 64.4% | 54.6% |

The recall-constrained hybrid recovered one public syringe and lost another. Matching counts alone would have hidden this exchange. The initial hybrid also lost an object in the clear block. Both hybrids were kept as research candidates. On adjudicated development boxes the installed detector matched all 163 local development instances. This shows the sensitivity of the first development result to four reference rectangles. The 14-photo held-out score still uses the original references. [Clear-block metrics](../../../artifacts/expanded/detector_guarded_test.json) · [Public regressions](../../../artifacts/expanded/detector_guarded_reserve.json) · [Annotation sensitivity](../../../artifacts/expanded/localization_sensitivity.json)

The verified CPU model and the XPU port were compared on nine development controls. Warm median latency for the two-view Grounding DINO portion fell from approximately 7 to 1.5 seconds with identical accepted counts and strongest-box IoU above 0.998 on all nine. The comparison excludes inter-process image serialization. The acceleration remains a benchmarked research path. [Device probe](../../../artifacts/expanded/grounding_xpu_probe.json)

## Markings and type

The reader keeps the existing PP-OCRv6 tiny detector and recognizer, quarter turns, barrel-aligned views and original-resolution patches. When an original reading is missing, two local contrast transformations run on the same crop. They must agree to supply additional capacity or visible scale evidence. A capacity without supported scale evidence requires matching unit-and-value output from both contrast views. Nominal volume, graduation type and actual fill volume remain separate fields.

| Reference-crop experiment | Correct nominal capacity | Wrong capacity | Abstentions | Median OCR time |
| --- | ---: | ---: | ---: | ---: |
| Calibration, original | 38 / 68 | 0 | 30 | 3.17 s |
| Calibration, one CLAHE view | 51 / 68 | 0 | 17 | 4.45 s |
| Calibration, one sharpened view | 44 / 68 | 0 | 24 | 4.51 s |
| Calibration, two corroborating views | 44 / 68 | 0 | 24 | 4.80 s |
| Held-out clear block, original | 8 / 14 | 1 | 5 | 2.61 s |
| Held-out clear block, frozen two-view configuration | 8 / 14 | 1 | 5 | 4.54 s |
| Clear-block replay with untyped-capacity guard | 8 / 14 | 0 | 6 | Same OCR passes |

All six extra corroborated capacity readings had the visible inscription `20 mL` on source patches. A separate diagnostic using detector-proposed boxes for those six frames increased the reads from five to six, recovering IMG_3297. The clear-block initial error was a `26 mL` reading on IMG_3190, which the follow-up corroboration requirement suppressed. That check is a regression after discovery. The frozen policy plus the guard both read all six separately selected 60-scale controls correctly at the source orientation and after a physical 180-degree raster rotation. These are deliberately legible controls and their results do not estimate overall OCR accuracy. The single-view CLAHE candidate answered more calibration examples and was retained in the report as a coverage alternative. [Per-frame calibration](../../../artifacts/expanded/markings_calibration.json) · [Detector-crop check](../../../artifacts/expanded/markings_detected_recovery_check.json) · [Capacity safety regression](../../../artifacts/expanded/markings_guard.json) · [Rotation evidence](../../../artifacts/expanded/markings_rotation_controls.json)

## Apparent liquid color, lightness and opacity

The liquid study compared color alone, the installed DINOv2 prior, and 24 new combinations of dense or color-augmented patch features, regularization and cutoff. The selected model intersects DINOv2 region support with a SAM 3 silhouette and measures median CIELAB L*, a*, b* and chroma after applying the embedded Display P3 profile. On 68 calibration crops the new model returned 60 measurements, of which 59 passed the centerline check. The installed model scored 58 and color alone scored 51. Supported median absolute L* error was 2.37 for the new model and 2.38 for the installed model. The new model's median CIE76 color difference was 7.35 against sampled reference traces. Exposure, white balance and reflection remain part of the measured appearance. [Calibration evidence](../../../artifacts/expanded/liquid_calibration.json)

The clear block exposed one false measurement in both learned models. Four thresholds for a nearby-background chroma check were evaluated on development first. The frozen check removed the clear-block error on regression and still reported two false colors on six separately reviewed public controls. The identified errors involved a colored environment and a colored syringe label. The candidate stays experimental. Eight candidate clear/colored pairings from the new capture were reviewed for relative opacity. Exposure differed in all eight and the plain background lacked controlled textured correspondence. Real opacity and the amount of liquid remain unresolved. [Clear-block evidence](../../../artifacts/expanded/liquid_test.json) · [Context-control audit](../../../artifacts/expanded/color_context_audit.json) · [Opacity audit](../../../artifacts/expanded/opacity_audit.json)

## Reproduce the study

Use a fresh experiment copy with the previously verified model assets, the existing locked root runtime, the isolated Intel XPU and PP-OCR environments, and a read-only source directory. Keep the original evidence when rerunning selection. The datasets, OCR and appearance caches are local. Public regression and calibration rows keep their attribution, licenses and hashes in the preexisting manifests.

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
source_dir=/path/to/Images_fil_rougev1
root_py=.venv/bin/python
xpu_py=data/sam3/xpu-env/bin/python
ocr_py=data/ocr-v6/env/bin/python

$root_py -m scripts.expanded_dataset "$source_dir"
$root_py -m scripts.expanded_annotations "$source_dir" --review data/expanded/review
$root_py -m scripts.expanded_proposals "$source_dir" fit cpu
$xpu_py -m scripts.expanded_proposals "$source_dir" fit sam
$xpu_py -m scripts.expanded_proposals "$source_dir" fit fallback
for branch in xpu sam fallback-xpu; do
  $xpu_py -m scripts.expanded_proposals "$source_dir" calibration "$branch"
done
$root_py -m scripts.expanded_detection_study select
$root_py -m scripts.expanded_recall_selection select
for cohort in test reserve; do
  $root_py -m scripts.expanded_proposals "$source_dir" "$cohort" cpu
  $xpu_py -m scripts.expanded_proposals "$source_dir" "$cohort" sam
  $xpu_py -m scripts.expanded_proposals "$source_dir" "$cohort" fallback
  $root_py -m scripts.expanded_detection_study "$cohort"
  $root_py -m scripts.expanded_recall_selection "$cohort"
done

for cohort in calibration test; do
  $ocr_py -m scripts.expanded_markings "$source_dir" "$cohort" batch
  $ocr_py -m scripts.expanded_markings "$source_dir" "$cohort" evaluate
done
$root_py -m scripts.expanded_markings_guard
$ocr_py -m scripts.expanded_ocr_controls "$source_dir"
$ocr_py -m scripts.expanded_detected_markings "$source_dir"
for stage in fit calibration select test audit; do
  $xpu_py -m scripts.expanded_liquid "$source_dir" "$stage"
done
$xpu_py -m scripts.expanded_color_context select
$xpu_py -m scripts.expanded_color_context audit
$root_py -m scripts.expanded_localization_sensitivity
$root_py -m scripts.expanded_release "$source_dir"
$root_py -m pytest -q
```

The combined runtime processed two original JPEGs. Clear IMG_3185 and colored IMG_3229 each returned one syringe and a 20 mL reading. The clear image abstained on appearance. Cold inference took 27.3 seconds and the second observation took 17.5 seconds, including OCR and liquid processing. These are two observed latencies under local machine load. Chromium verified file upload, the real detection overlay, markings, revision persistence and the ZIP export. All 177 expanded originals and 75 original HEIC entries retained their recorded byte hashes. The temporary browser server and its model workers were stopped after verification. [Combined runtime evidence](../../../artifacts/expanded/release_assessment.json) · [Browser check](../../../artifacts/expanded/browser_smoke.json) · [Source verification](../../../artifacts/expanded/source_reverification.json)

The project preserves the unsuccessful variants, release checks, and the experiment log. The local app starts from the README with no user-adjusted inference cutoff. [Experiment journal](expanded_experiment_log.md) · [Local launch](../../../README.md)
