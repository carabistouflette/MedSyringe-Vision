# Supervised detector and localization study

This study tests ways to improve the precision of the currently released detector while recovering useful localization. It keeps the application configuration frozen during experimentation.

## What was trained

| Approach | Training signal | Controlled comparison |
| --- | --- | --- |
| One-class SSDLite | Background versus syringe and supervised anchor-box regression | Freeze MobileNetV3-Large features, train the category and box heads for eight CPU epochs |
| Explicit empty-scene loss | The eight most costly false foreground anchors in each negative scene | Original SSD objective versus extra background supervision, with other training settings held fixed |
| Photometric and orientation augmentation | Horizontal flips with transformed boxes and brightness perturbation | Add augmentation to the empty-scene objective |
| Box correction | Weighted ridge regression of normalized coordinate offsets to reference boxes | Half and full corrections, with and without the quality model |
| Localization quality | Gradient-boosted regression of proposal IoU based on shape, score and appearance | Score blending with and without box corrections |
| Cross-model agreement | Score support from a geometrically overlapping SSDLite detection | Compare score weights, matching overlaps and optional box averaging |

SSDLite trains on 145 images, 33 without a syringe. The refiner samples one Fil rouge image per burst and uses the annotated public development data. No photo from the 21-image audit participates in training, cutoff choice or epoch selection. The 17-image calibration sample was previously inspected during the lookalike study and is treated as development. The ten original Open Images examples are a historical regression only. The audit contains nine syringe boxes and twelve images without syringes, with 21 recorded author groups. Dataset checks compare group, author, exact hash, source URL and perceptual hash across partitions. The audit boxes are single-review visual references. Overlap with upstream pretrained datasets remains unknown.

## Development results

The common development sample contains six annotated syringes.

| Candidate | Precision | Recall | F1 | AP@50–95 |
| --- | ---: | ---: | ---: | ---: |
| Released detector | 0.357 | 0.833 | 0.500 | 0.663 |
| SSDLite standard | 0.800 | 0.667 | 0.727 | 0.525 |
| SSDLite plus empty-scene loss | 0.800 | 0.667 | 0.727 | 0.497 |
| SSDLite plus empty-scene loss and augmentation | 1.000 | 0.667 | 0.800 | 0.506 |
| Proposal IoU ranking | 0.625 | 0.833 | 0.714 | 0.576 |
| Frozen cross-model candidate | 0.625 | 0.833 | 0.714 | 0.650 |

The best SSDLite run has four true and zero false positives on this development set. It retrieved only 3/15 boxes on the historical Open Images regression. Thus replacing the detector with SSDLite alone would give up substantial recall. Explicit empty-scene loss alone tied the standard training at the selected F1, and adding flips and exposure changes removed the remaining false positive in this small sample. Coordinate regression was compared explicitly and was absent from the strongest calibration variant. The frozen cross-model candidate used 15% SSDLite score support at IoU ≥0.35 and left box coordinates unchanged. A base-model score-only control was also fixed on calibration at the strictest score keeping the same TP count.

[Training curves and selected checkpoint](../../../artifacts/supervised_study.json) · [Localization ablations](../../../artifacts/localization_study.json) · [Fusion trials](../../../artifacts/supervised_selection.json)

## Frozen comparison on the 21-image audit

| Candidate | True positives | False positives | Precision | Recall | F1 | AP@50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released detector | 7 | 17 | 0.292 | 0.778 | 0.424 | 0.261 |
| Released detector, stricter calibration cutoff | 2 | 4 | 0.333 | 0.222 | 0.267 | 0.261 |
| Supervised SSDLite | 1 | 0 | 1.000 | 0.111 | 0.200 | 0.368 |
| IoU quality reranking | 3 | 5 | 0.375 | 0.333 | 0.353 | 0.242 |
| Grounding DINO and SSDLite fusion | 4 | 6 | 0.400 | 0.444 | 0.421 | 0.352 |

Cross-model agreement raised point precision by 10.8 percentage points and AP@50–95 by 9.1 points. It lost three true detections at the chosen operating point. Its F1 was essentially unchanged. The score-only control lost five true detections, so the cross-model signal offered a better balance than that stricter base score under the fixed calibration protocol. The fusion retrieved 4/15 boxes instead of 9/15 on the historical Open Images regression and 67/74 instead of 70/74 on Fil rouge. The existing detector remains active. The audit is small and enriched for lookalikes, so its point precision cannot predict routine deployment precision.

[Audit scores and errors](../../../artifacts/supervised_comparison_supervised-audit.json) · [Score-only control](../../../artifacts/supervised_control_audit.json) · [Historical regression](../../../artifacts/supervised_comparison_test.json) · [Fil rouge regression](../../../artifacts/supervised_comparison_train.json)

## Scale and exposure-balance ablation

This follow-up used aspect-preserving padding and compared 320 px with 512 px, then limited each Fil rouge burst to one image in the 512 px run. It used four epochs per run. The 21-image audit was already consulted at this stage and is only an exploratory comparison here.

| Variant | Historical audit TP | FP | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Aspect-preserving 320 px | 0 | 0 | 0.000 | 0.000 |
| Aspect-preserving 512 px | 2 | 2 | 0.222 | 0.308 |
| Aspect-preserving 512 px, one image per Fil rouge burst | 0 | 4 | 0.000 | 0.000 |

The 512 px model retrieved two more targets than the padded 320 px variant, and both fell below the released detector. Subsampling reduced the number of training updates as well as the exposures per burst, so this experiment does not isolate burst weighting by itself. Weighted sampling at equal training steps is a cleaner future comparison. [Scale ablation and checkpoints](../../../artifacts/scale_study.json)

## Priorities from the measured failures

1. Increase the number of independently captured positive scenes with distant syringes, partial occlusion and clinical distractors, and include empty frames in the same acquisition environments. Positive and negative scenes from different domains can encourage the model to classify background style.
2. Review box targets at high resolution with a shared convention for visible tips, needles, and clipped objects. The tested coordinate regressions reduced calibration AP@50–95 from 0.663 to 0.467 at half strength and 0.200 at full strength, with the baseline score threshold held fixed.
3. Compare a larger supervised detector or partial backbone fine-tuning only after improving data coverage. Validate operating scores on several sufficiently populated acquisition groups and keep a new final test. The ranking gain from cross-model agreement came from scores because the selected configuration preserved the original box coordinates.

## Reproduction

Use the locked local environment and the prepared image manifests. Fetch the official SSDLite COCO checkpoint to `data/models/ssd/ssdlite320_mobilenet_v3_large_coco-a79551df.pth`. The measured input SHA-256 is recorded in the training artifact. Preserve the original selected files and use a separate copy of the workspace when rerunning selection.

```bash
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.supervised_study
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.localization_study
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.compare_supervised select
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.compare_supervised control-select
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.compare_supervised audit
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.compare_supervised control-audit
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.compare_supervised historical
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.compare_supervised train
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.scale_study
```

[Official TorchVision SSDLite model and published weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) · [Audit references and attribution](../../../artifacts/supervised_external_manifest.json)

## Recall release requirement

The released operating point is the reference for every subsequent precision improvement. At IoU ≥0.50, every individual reference box matched by the current detector must remain matched in each comparison. Recovering one missed instance cannot compensate for losing a previously matched instance. A precision-verifier release requires this evidence on a new holdout, Fil rouge, the historical Open Images set, and the previous external set. It also requires fewer false positives on the new holdout. The application verifies the assessment before loading the verifier. This is a release criterion on measured data, with no guarantee for unseen scenes. [Instance-level checks on previous candidates](../../../artifacts/recall_regression_guard.json)
