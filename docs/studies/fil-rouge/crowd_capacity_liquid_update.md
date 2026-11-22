# Public instance audit, crowded counting, and small-barrel measurements

## Public instance annotations

The combined public annotation dataset has 742 distinct positive photos or camera frames, 769 reviewed syringe boxes and 13 visually empty photos. Ten positives show multiple syringes. Reviewers examined 547 primary catalog photos and 504 additional proposals, correcting failed geometry against source pixels. The total includes 29 individual-rights-verified positive photos from the separate source pilot. Three cross-source duplicates or visual neighbors were removed when joining the reviewed exports. Use the normalized manifest to inspect attribution and eligibility. The companion COCO export uses pixel-space boxes and integer license identifiers. [Dataset summary](../../../data/instance-review-dataset/summary.json) · [Manifest](../../../data/instance-review-dataset/manifest.json) · [COCO export](../../../data/instance-review-dataset/instances.coco.json) · [Individual source checks](../../datasets/instance_source_review.md)

A second reader traced a separate raw-photo sample of 14 from the primary catalog without model overlays. Counts agree on all 14. All 34 paired boxes exceed IoU 0.75, with a median of 0.907 and a minimum of 0.793. The sample spans nine primary catalog groups and includes five multi-instance scenes. It measures agreement on this sample. The full dataset has one visual approval per photograph. [Independent geometry sample](../../../artifacts/detection_hillclimb/public_instance_blind_geometry_audit.json)

The split keeps publisher or acquisition families intact. It links shared contributors and assets, checks SHA-256 values and transformed visual fingerprints, and prioritizes eligible positive and hard-negative acquisition coverage. Groups related to previously scored photos carry the historical-reference-only role. This includes a catalog linked to a held reference family by a visual collision. The 325 historical-family positives are outside all new fitting and evaluation roles. The remaining 417 positive photos span 382 fitting-role, two development-role and 33 reserve-role examples. Individual asset provenance qualifies 25 of the fitting-role photos today. Publisher-declared licenses remain attached to 713 records with individual source verification pending. The independent crowd benchmark remains a separate protected acquisition set. This annotation release reaches the 500-photo geometry target. The current fitting pool remains small, and this iteration did not fit a new detector. [Primary review](../../../artifacts/detection_hillclimb/public_instance_review_batch_extended.json) · [Additional review](../../../artifacts/detection_hillclimb/public_instance_review_bulk.json) · [Group and visual collision evidence](../../../data/instance-review-dataset/manifest.json)

The final audit checked 755 raster SHA-256 values and EXIF-oriented COCO dimensions, all 769 annotation bindings, and the grouped split. It found zero exact collisions with the new crowd reference or Fil Rouge and zero collisions under the split leakage checks. The project suite completed with 119 passing tests and one existing dependency deprecation warning. [Integrity report](../../../artifacts/detection_hillclimb/public_instance_final_integrity.json)

## Crowded counting on new acquisitions

The public reference contains 25 photographs from 25 conservative acquisition groups. Two readers worked from the original images before inference and found 84 affirmative instances. One reader identified an unresolved additional instance in a discarded equipment scene. The primary comparison excludes that scene and uses 24 groups, 81 syringes and five hard negatives. Earlier public releases, annotated evaluations and Fil Rouge were screened for source and raster overlap. Repeated specimens can still occur across unrecognized acquisition aliases. [Reference and source audit](../../datasets/crowd_next_reference.md) · [Independent raw-photo counts](../../../artifacts/detection_hillclimb/crowd_next_reference_second_review.json) · [Frozen policy](../../../artifacts/detection_hillclimb/crowd_next_audit_freeze.json)

| Fixed method | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 | Exact count | Correct negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Released Intel engine | 33 | 1 | 48 | 97.1% | 40.7% | 57.4% | 40.0% | 30.5% | 10/24 | 4/5 |
| Released engine plus unmatched SAM instances | 36 | 1 | 45 | 97.3% | 44.4% | 61.0% | 43.9% | 32.9% | 12/24 | 4/5 |
| SAM mask branch alone | 65 | 9 | 16 | 87.8% | 80.2% | 83.9% | 75.1% | 48.0% | 18/24 | 3/5 |

Matching uses visible reference boxes at IoU 0.50. AP is calculated from the outputs surviving each fixed operating policy. The additive policy preserves all prior matches and all previously correct counts. It increases exact positive crowd counts from 6/19 to 8/19. Mean absolute count error falls from 2.04 to 1.92 syringes per image. The standalone branch has an error of 0.79 but adds eight false positives. The conservative policy activates only when the released branch already has multiple matched detections. Under-counted scenes with zero or one release prediction therefore remain the main failure mode. [Per-scene counts, errors, and XPU timings](../../../artifacts/detection_hillclimb/crowd_next_audit_results.json)

The new evaluation passes the preregistered acquisition and recall guards. The original frozen native regression rises from four to five false positives at the same policy. The later blind relabeling removes that apparent regression, and both versions remain available as a sensitivity analysis. The application therefore retains its released count. The SAM CLI continues to provide a separate local source of review suggestions. [Prior qualification and alternative labels](../detection/crowded_sam_evaluation.md)

## Small printed capacity and visual volume

The experimental reader aligns the barrel, finds a regular track of visible graduation strokes, and recognizes the individual numbers and nearby unit with local PP-OCRv6. It requires a consistent terminal number and an explicitly read mL unit. It checks fine roll and uses a bicubic 2× recovery only when the native crop has no accepted capacity. The separate fill logic requires visible stopper and chamber evidence and leaves its fields unknown on ambiguous crops.

| Controlled change | 5 mL regression photo, four rotations | Rendered, readable small scales |
|---|---:|---:|
| Original full-crop OCR | 0/4 | 0/7 |
| Separated digits and contrast-adjusted unit patch | 2/4 | 7/7 |
| Fine alignment | 3/4 | 7/7 |
| Additional unit crops and OpenCV cubic resize | 3/4 | 7/7 |
| Bicubic PIL fallback | 4/4 | 7/7 |

All four successful rotated views report a seated colored stopper, consistent with visually empty. The predicted volume is 0 mL with an apparent bracket of 0–0.2 mL. These rotations share one photograph. A five-photo public challenge gave no new field emissions, including the marked 1 mL view, the clear barrel and the yellow liquid. The later fixed-reader transfer across three photos of one oblique 5 mL specimen gave no capacity reads across twelve rotated crops. The original OCR read one of these views at 5 mL. The nominal capacity remains a separate output from visual contents and plunger position. [Primary study](../../../artifacts/detection_hillclimb/small_field_study_results_r5.json) · [Oblique public transfer](../../../artifacts/detection_hillclimb/small_field_public_transfer.json)

A follow-up diagnostic straightened and narrowed the crop on the same public photos. This recovered one 5 mL photo under all three trial widths. It is the same photo the original OCR read once. The two views with a stopper obscuring part of the scale remained unresolved. The 1 mL photo has only source metadata for its unit. The small-scale reader withheld a capacity on the hard-negative medicine container, where full-crop OCR read 10 mL when invoked directly. The diagnostic uses already observed photos. [Barrel-axis diagnostic](../../../artifacts/detection_hillclimb/small_capacity_axis_diagnostic.json)

On controlled images, presence is emitted correctly on five of eight cases. Five volume intervals contain the simulated amount. The experimental method abstains for a visible gap with unknown contents, a clear liquid that resembles air, and obscured terminal graduations. The color name agrees on two of the three emitted cases. The amber render is labeled yellow in the study and reported as orange by the reader. The reported apparent lightness of the emitted controls ranges from 62.1 to 78.6 L*. These are properties of the rendered pixels. Real transparent liquid may require an identifiable meniscus, optical reference, or additional acquisition evidence. [Rendered and camera cases](../../../artifacts/detection_hillclimb/small_field_study_results_r5.json)

A one-factor robustness check evaluated Gaussian blur, two exposure changes, perspective and JPEG compression on the public development crop and one rendered blue barrel. On the camera crop, five of six variants preserved the capacity, presence and visual fill interval. The JPEG quality-36 variant caused abstention for all three. On the rendered control, all six variants preserved the three supported outputs. This check measures sensitivity on two known parents. [Perturbation outputs](../../../artifacts/detection_hillclimb/small_barrel_robustness.json)

The three oblique views come from the same LabSnap acquisition by Nadina Wiórkiewicz. Their same-source relationship is kept in the evaluation. [Disposable syringe 5ml 1.jpg](https://commons.wikimedia.org/wiki/File:Disposable_syringe_5ml_1.jpg) · [Disposable syringe 5ml 2.jpg](https://commons.wikimedia.org/wiki/File:Disposable_syringe_5ml_2.jpg) · [Disposable syringe 5ml 3.jpg](https://commons.wikimedia.org/wiki/File:Disposable_syringe_5ml_3.jpg)

## Relative opacity under exposure variation

The optical trial measures contrast attenuation between registered views of the same background. An unchanged textured exterior patch provides a common exposure estimate in linear luminance. The fit refuses a mismatched control patch. With eight simulated exposure changes, the current direct estimator emitted seven times with mean absolute error 0.118. The exterior-normalized estimator emitted eight times with mean absolute error 0.0005 on the same controls. This experiment has no paired real photographs and does not establish physical opacity from a single photo. The application keeps its current matched-crop path. [Exposure challenge](../../../artifacts/detection_hillclimb/liquid_exposure_study.json)

## Reproduce the experiments locally

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
data/sam3/xpu-env/bin/python -m scripts.experiments.public_data.crowd_next_reference verify
data/sam3/xpu-env/bin/python -m scripts.experiments.crowds.crowd_next_audit freeze
data/sam3/xpu-env/bin/python -m scripts.experiments.crowds.crowd_next_audit released
data/sam3/xpu-env/bin/python -m scripts.experiments.crowds.crowd_next_audit sam
data/sam3/xpu-env/bin/python -m scripts.experiments.crowds.crowd_next_audit measure
data/ocr-v6/env/bin/python -m scripts.experiments.measurement.small_field_study run --output /tmp/medsyringe-small-field-replay.json
data/ocr-v6/env/bin/python -m scripts.experiments.measurement.small_field_public_transfer run
data/ocr-v6/env/bin/python -m scripts.experiments.measurement.small_capacity_axis_diagnostic
data/ocr-v6/env/bin/python -m scripts.experiments.measurement.small_barrel_robustness
.venv/bin/python -m scripts.experiments.measurement.liquid_exposure_study
.venv/bin/python -m scripts.experiments.public_data.instance_geometry_audit score
.venv/bin/python -m scripts.experiments.public_data.instance_review_dataset \
    artifacts/detection_hillclimb/public_instance_review_batch_extended.json \
    artifacts/detection_hillclimb/public_instance_review_bulk.json
```

The historical first through fourth OCR comparisons were recorded as development runs and remain archived separately. Reproduction uses the fixed final reader. None of the new trials uses Fil Rouge as fitting data or as an augmentation parent.

To inspect a separate source image, use the experimental command with a tight syringe crop or its normalized bounding box. Output retains the recognized markings, coordinates, apparent fill interval and field-level reasons for abstention.

```bash
data/ocr-v6/env/bin/python -m scripts.experiments.measurement.inspect_small_barrel syringe.jpg --output /tmp/syringe-evidence.json
```
