# Precision study against syringe lookalikes

The unified detector's previous external sample had 18 true positives and nine false positives. The false positives included glass ampoules, a microdispenser, and additional boxes near real syringes. On the original calibration images, two of the three errors were poorly localized boxes on real syringes, a distinct problem from recognizing lookalikes.

The tested verifier runs after the detector. It can send an accepted proposal to review, so it cannot recover a missed syringe. The cutoffs remain internal.

## Studies

- **Initial training.** Three positive and ten negative new public photos plus one image per Fil rouge burst. Negative examples include the base detector's proposals in negative training scenes. Calibration initially used the original validation set and then the previously scored 24-image set. The frozen result removed zero of 15 false positives in the 15-image lookalike-heavy holdout and was withheld from deployment. [Initial holdout evaluation](../../../artifacts/precision_holdout_initial_study.json)
- **Hard-negative training.** Earlier evaluation photos are explicitly development data for this study. Training adds ground-truth crops and accepted detector proposals from the original validation and the older 24-image set. A proposal counts as positive at IoU ≥0.50. A proposal with IoU <0.20 is eligible as a negative. The original Fil rouge burst sample, three independent public positives, and ten original negative training photos are included. The previously inspected 15-image sample is used only for calibration.
- **Frozen crop encoder and head.** TorchVision MobileNetV3 Small provides normalized embeddings from padded 160 × 160 crops. A weighted logistic head trains on original and mirrored crops. Candidate regularization strengths are 1, 10 and 100. Calibration minimizes false positives subject to retaining every baseline true positive.
- **New holdout and deployment.** A second new external sample of 17 manually reviewed photographs has six positive reference boxes, eleven negative scenes and 16 known authors. The author, SHA-256, source URL, group and difference-hash checks run against the training and previously seen partitions. The new sample is scored only after the candidate is frozen. Deployment requires fewer false positives and zero lost baseline true positives on the new set, and no loss of baseline true positives on the Fil rouge development set. This small, deliberately lookalike-heavy sample cannot estimate routine precision across real operator populations.

A higher base-detector cutoff was checked on the already seen 43-image set. At 0.82, precision rose from 0.721 to 0.852 while recall fell from 0.886 to 0.657. This tradeoff was considered as a diagnostic only.

A zero-shot prompt probe also compared scores for syringe, ampoule, vial, pipette, dropper and pen. Some true syringes scored higher as pipettes, while some ampoules still scored higher as syringes. This probe did not become a decision rule. [Prompt probe](../../../artifacts/semantic_prompt_pilot.txt)

Precision, recall and F1 are evaluated at the frozen operating point. The verifier runs only on already accepted boxes, so the candidate's exported AP curve omits below-operating-point proposals. Use the operating metrics for comparison.

## Hard-negative result

| Evaluation set | Detector | True positive | False positive | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Calibration, 15 scenes | Unified detector | 3 | 15 | 0.167 | 0.750 | 0.273 |
| Calibration, 15 scenes | Crop verifier | 3 | 3 | 0.500 | 0.750 | 0.600 |
| Fresh holdout, 17 scenes | Unified detector | 5 | 9 | 0.357 | 0.833 | 0.500 |
| Fresh holdout, 17 scenes | Crop verifier | 2 | 4 | 0.333 | 0.333 | 0.333 |

The fresh evaluation did not confirm the calibration improvement. The verifier removed five false positives and three true detections. The Fil rouge check also dropped from 70 to 67 correctly localized boxes, with zero false positives for either version. The deployment gate failed, so the API and app continue to use the existing unified detector. Further selection must use new images for independent assessment. The two negative-heavy samples diagnose distractors at intentionally high prevalence. The earlier measurement of 66.7% in a different sample describes that sample only. [Hard-negative evaluation](../../../artifacts/precision_holdout.json)

## Reproduction

Fetch the public encoder with `bash scripts/fetch_models.sh`. Retain the frozen base detector, the supplied derivative images and the manifest assets in a fresh workspace. Run `HF_HUB_OFFLINE=1 uv run --locked python -m scripts.precision_study fit` and then `HF_HUB_OFFLINE=1 uv run --locked python -m scripts.precision_study test` to reproduce the hard-negative study. The verifier loads in the app only if the checked release artifact exists. Rejected candidates remain inspectable.

[TorchVision MobileNetV3 Small weights and transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) · [TorchVision pretrained weight terms](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights) · [Training and calibration references](../../../artifacts/precision_study_manifest.json) · [Fresh holdout references](../../../artifacts/semantic_holdout_manifest.json)
