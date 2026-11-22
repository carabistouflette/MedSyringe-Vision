# Fil Rouge precision, fill review, and Intel runtime

## Protocol

The active single-session collection contains 177 JPEGs in 95 fit, 68 calibration and 14 held-out frames. Earlier HEIC exports from the same capture retain their prior grouping. The public reserve has 21 scenes and seven reference syringes, with empty scenes and syringe-like objects. Its detections were previously inspected. The reserve and the captured JPEG replays are regression evidence. There is no new independent session in this study.

The crop experiment fits only on the existing fit groups. Calibration selects the method and fixed operating point. A cross-partition check compares group, source, author, exact hash, and perceptual hash within four bits where available. There were no additional collisions after the established exclusions. The frozen choice was then replayed on all 177 photos and the public reserve. A candidate must preserve each previously matched instance at IoU ≥ 0.50 and avoid extra false-positive detections.

## Detection

Grounding DINO generates proposals. The established RF-DETR and SAM 3 paths still recover candidates. The verifier learns appearance from 460 proposal crops across 116 fit frames from 27 groups. There were 290 positive crops and 170 hard negatives. Fit includes both accepted crops and additional proposals from the frozen branch caches. Near-overlap ambiguous training examples are excluded and weights balance the class and acquisition group. Thirty settings compare frozen DINOv2 Small embeddings, HOG, and both together at five internal operating points, with and without corroboration.

DINOv2 Small with two-detector corroboration won calibration. A weak crop goes to review unless supported by two detector branches at the same location. In the application, RF-DETR and SAM 3 are consulted for this decision only when necessary. The report preserves the original detector score and the reason for review. The 87-image calibration had 68 matches before and after, with false positives dropping from 23 to 7. [Selection trials](../../../artifacts/expanded/crop_verification_selection.json)

| Crop features | Fewest false positives with all matches preserved, crop-only | With corroboration |
| --- | ---: | ---: |
| HOG | 23 | 19 |
| DINOv2 Small | No eligible setting | 7 |
| DINOv2 Small + HOG | No eligible setting | 7 |

Corroboration preserved at least one match at every tested DINOv2 setting where crop evidence alone lost a match. The combined descriptor tied DINOv2 on the best calibration F1, so the production verifier uses the single embedding.

| Frozen regression cohort | Images | Matches | False positives before → after | Misses | F1 before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fil Rouge fit | 95 | 94 | 1 → 1 | 1 | 98.9% → 98.9% |
| Fil Rouge calibration | 68 | 65 | 3 → 3 | 3 | 95.6% → 95.6% |
| Fil Rouge clear-block test | 14 | 13 | 1 → 1 | 1 | 92.9% → 92.9% |
| Public reserve | 21 | 4 | 16 → 2 | 3 | 29.6% → 61.5% |

The cached Fil Rouge replay retained 172 of 177 annotated instances and had an exact count in 177 of 177 scenes. Five accepted boxes fell below IoU 0.50 against their reference. All four previously detected public syringes kept the same match, and three public syringes remained missed. Public precision moved from 20.0% to 66.7%, AP50 from 32.7% to 68.4%, and AP50–95 from 27.2% to 56.7%. AP includes the untouched subthreshold proposals and excludes the verifier's review-only boxes. Single-capture precision, recall and localization are unchanged. [Full capture and public audit](../../../artifacts/expanded/crop_verification_full_capture.json)

The initial crop-verifier diagnostic computed candidate AP from accepted boxes alone. Its fixed-cutoff metrics and match comparisons are valid. The full-capture report uses unchanged subthreshold candidates on both sides for the AP comparison.

A sensitivity analysis using the four previously corrected development annotations raised the descriptive match count to 176 of 177 for the same predictions. IMG_3190 still falls below the threshold, at IoU 0.484. The original labels remain the evaluation reference for the release decision. [Annotation corrections](../../../artifacts/expanded/annotations_adjudicated.json) · [Sensitivity analysis](../../../artifacts/expanded/localization_reference_sensitivity.json)

## Intel runtime

On the Intel Core Ultra 7 258V, the paired full inference benchmark changed the device for Grounding DINO and held the Python stack, OCR and SAM workers, and input rasters fixed. The eight inputs included two physically rotated controls and a public negative. The warm median across seven inputs fell from 10.55 s to 5.60 s, a 1.88× speedup. The comparison checked count, box IoU, capacity, scale, color, fill, and lightness within one L* unit. This benchmark predates the crop-verifier integration. [Paired controls and timings](../../../artifacts/expanded/runtime_pair.json)

The optional Intel launcher uses the installed local XPU environment and a pinned release. Device or inference errors retry the entire image through CPU Grounding DINO. The application and CLI use one detection configuration. Users do not enter a detection threshold. The optional launcher determines the device.

The staged CPU verifier was also checked in the main locked environment, which uses Transformers 4.57.3. For the rotated IMG_3203, the same-stack CPU box had IoU 1.000 with and without the verifier. Its overlap with the archive from Transformers 5 was 0.976, with the same count, capacity and fill. This cross-stack difference was logged and checked on the same image with the unchanged CPU detector.

The activated crop verifier passed nine full-input controls on each backend. They cover positive and negative scenes, two rotated fill controls, and a public lookalike sent to review. The default application and the production CLI load the same released verifier. The CPU CLI returned one syringe and the expected 16 mL on IMG_3316. [Activation evidence](../../../artifacts/expanded/crop_verifier_runtime.json)

## Fill and appearance

The fill experiment reused the frozen piston locator and rectified pixel strips parallel to the printed scale. It tried three anchor spans and optional support from a visibly ordered terminal graduation. The six variants all returned seven existing estimates and zero additional estimates on 31 development crops. A frozen representative was replayed on the 45 colored and 14 clear controls. The established 20 colored estimates were retained and the clear controls remained unanswered. Both the original estimate and its abstention rule remain deployed. [Selection](../../../artifacts/fill/geometry_selection.json) · [Replay](../../../artifacts/fill/geometry_regression.json)

The application supports human marking of two visible graduations and the liquid-facing piston edge. It projects in source-image pixels, validates the marks against the selected box and the nominal capacity, and interpolates between the two printed values. A review stores millilitres, percentage, reviewer, revision and normalized image landmarks. The export preserves the original model estimate. The visual interpolation is an image estimate with no physical confidence interval.

The Chromium end-to-end test analyzed IMG_3316 with the released verifier and Intel grounder, placed three image landmarks, saved the revision, and checked both estimates in the ZIP. It also checked the editor at a 390-pixel viewport. The coordinates in this test reproduce the automatic markings to exercise the review workflow. [Browser verification](../../../artifacts/fill/improvement_browser_smoke.json)

The appearance experiment compared twelve screening settings using barrel-aligned continuity, cross-section width, and optional exclusion of bright low-chroma pixels. The selected screen supported 58 of the 68 calibration photos, with median discrepancy 2.35 L* from visual centerline samples. The deployed baseline supported 58, at 2.38 L*. The candidate incorrectly reported color on one of 14 clear photos and two of six public controls. The deployed baseline produced one and zero reports respectively in the same replay. The candidate remains experimental. Absolute opacity is still unavailable from the provided photos. [Appearance selection](../../../artifacts/expanded/liquid_structure_selection.json) · [Appearance replay](../../../artifacts/expanded/liquid_structure_regression.json)

## Reproduction

Use the existing prepared detector and liquid caches and the read-only JPEG collection. The selection scripts refuse to overwrite an existing freeze. Run new selections in an experiment copy if reproducing from scratch. All commands run locally.

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
source_dir="/path/to/Images_fil_rougev1"
.venv/bin/python -m scripts.experiments.expanded.expanded_crop_verification audit --source "$source_dir"
.venv/bin/python -m scripts.experiments.expanded.crop_verification_release regress --source "$source_dir"
.venv/bin/python -m scripts.experiments.measurement.fill_geometry_study audit --source "$source_dir"
data/sam3/xpu-env/bin/python -m scripts.experiments.measurement.liquid_structure_study audit
data/sam3/xpu-env/bin/python -m scripts.experiments.expanded.expanded_runtime_pair cpu --source "$source_dir"
data/sam3/xpu-env/bin/python -m scripts.experiments.expanded.expanded_runtime_pair xpu --source "$source_dir"
data/sam3/xpu-env/bin/python -m scripts.experiments.expanded.expanded_runtime_pair compare
for group in a b; do
  .venv/bin/python -m scripts.experiments.expanded.crop_verification_release cpu --group "$group" --source "$source_dir"
  data/sam3/xpu-env/bin/python -m scripts.experiments.expanded.crop_verification_release xpu --group "$group" --source "$source_dir"
done
```

Archived runtime and audit records may be read directly without modifying the release. Reproducing timings naturally changes their digests. Use an experiment copy to preserve the delivered runtime release evidence.
