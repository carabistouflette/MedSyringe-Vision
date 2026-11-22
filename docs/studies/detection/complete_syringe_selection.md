# Ranking complete visible syringes before suppression

The [axis verifier](axis_instance_separation.md) identifies two overlapping candidates as distinct from the released boxes in a six-syringe photograph. Its automatic policy admits an incomplete higher-scoring proposal that blocks one of them. This experiment tests a learned estimate of completeness and a joint selection of candidate boxes.

## Fitting and frozen policy

The policy scores completeness of the visible device. A reference can omit regions behind gloves or other objects. Fitting uses reviewed boxes, fragments and lookalikes from public acquisition groups along with geometry-labeled detector proposals and constructed fragments. Positive detector examples overlap a reference at IoU 0.60 or higher and cover at least 81% of that reference. The proxy measures agreement with visible axis-aligned bounds. It does not measure how much of the physical syringe is exposed. There are 2,123 examples from 116 groups, including 228 positives, 26 reviewed fragments, 432 detector fragments, 332 constructed fragments and 28 reviewed lookalikes. User images and all reserved rows stay outside this fit. [Supervision ledger](../../../data/complete-syringe/supervision.json)

A weighted logistic model uses dimensions, a paired-edge profile along the projected axis, edge continuity and other crop gradients. The body-aware score combines this probability with the original RF-DETR score. The three arms compare detector ordering, body-aware greedy processing, and body-aware joint selection with the same candidate budget. A conflict graph uses conventional overlap or the frozen learned identity exception. A bounded exact weighted independent-set solver chooses a compatible subset from at most 14 proposals. It preserves all incumbent detections. A joint objective can recover two moderate-scoring compatible objects over one high-scoring conflicting proposal. It depends on accurate per-box weights and conflict edges. When two arms tie under the frozen development criteria, the deterministic order prefers the simpler greedy arm.

The 486 development settings are evaluated on 60 public photos. The best body-based settings each add two true detections and zero false positives against the same 15-match released baseline. The detector-ordering control adds one true detection and zero false positives. Both body variants selected conventional overlap on development. A fixed diagnostic policy enables the identity exception under the same settings for each ordering method. [Weights, selected rules and development grid](../../../artifacts/detection_hillclimb/complete_syringe_policy.json)

Five public acquisitions were hand-labeled before their first detector runs in this study. They cover four crossed needles, three adjacent syringes held together, two single held devices and a pipette negative. Source ID, URL, contributor, local hashes and rotated or mirrored perceptual fingerprints were screened against earlier data. A known contributor and several visually ambiguous images were excluded. These photos are local evaluation assets under the source terms. [Frozen reserve annotations](../../../artifacts/detection_hillclimb/complete_syringe_reserve.json) · [Four syringes by Piyapong Sayduang](https://www.pexels.com/photo/set-of-syringes-on-blue-surface-4541406/) · [Three syringes by Kaboompics](https://www.pexels.com/photo/person-holding-three-syringes-with-medicine-4046952/) · [Pexels license](https://www.pexels.com/license/)

## Frozen evaluation and ordering diagnosis

| Cohort and measure | Released detector | Selected body-aware greedy rule | Fixed active body-aware joint probe |
|---|---:|---:|---:|
| New reserve matched syringes / 9 | 7 | 7 | 8 |
| New reserve false positives | 1 | 1 | 1 |
| New reserve exact counts / 5 | 4 | 4 | 5 |
| Previously exposed matched syringes / 16 | 14 | 14 | 15 |
| Previously exposed false positives | 1 | 1 | 5 |

The new reserve gain from the active probe comes from the third syringe held in the gloved hand. All three active arms produce the same result there. The selected rule gains one match without adding false positives in the five-image comparison archive. It gains none on the new reserve. The active probe adds four false positives on the seven previously exposed photos and two false positives on the older 18-image archive. No rule qualifies for release. [Per-image results](../../../artifacts/detection_hillclimb/complete_syringe_results.json) · [Older archive guard](../../../artifacts/detection_hillclimb/complete_syringe_diagnostic.json)

On the exposed six-syringe photo, the completeness head rates the remaining valid proposal at 0.351 and its competing fragment at 0.342. The original detector favors the fragment at 0.402 versus 0.352. The 30/70 blended ranking reverses the improvement and retains the fragment under both greedy and joint selection. In a post-evaluation replay, the detector still provides proposals and the same minimum candidate score. Completeness supplies the ranking and acceptance score. This replay produces all six valid matches and zero false positives on that photo using the same conflict rule. It uses no reference boxes to choose the detections. The change was motivated by the exposed error and has no independent qualification. [Score and proposal audit](../../../artifacts/detection_hillclimb/complete_syringe_diagnostic.json)

The full test suite passes 156 tests. The additional checks cover fragment supervision, shared score floors, incumbent preservation and the exact joint solver's ability to bypass an individually stronger proposal. The split audit finds zero group, image-ID or content-hash collisions between fitting, development and the reserve. A second run reproduces the evaluation and diagnostic JSON byte for byte. [Follow-up with separately calibrated completeness scores](completeness_calibration.md)

## Reproduction

Restore source bytes using the exact image URLs in the reference manifest. Keep the existing public proposal caches and previous release experiment outputs. Work from the project root.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.complete_syringe_selection train
.venv/bin/python -m scripts.complete_syringe_reserve freeze
data/rfdetr/env/bin/python -m scripts.complete_syringe_reserve collect
data/sam3/xpu-env/bin/python -m scripts.complete_syringe_reserve baseline
.venv/bin/python -m scripts.complete_syringe_selection score
.venv/bin/python -m scripts.complete_syringe_diagnostic
.venv/bin/python -m pytest -q
~~~
