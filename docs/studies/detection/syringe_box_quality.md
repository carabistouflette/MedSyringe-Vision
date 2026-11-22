# Separate syringe identity and box quality

The earlier replay found two recoverable syringes excluded at different stages. One usable box lost to a poorer box for the same injector. Another accurate box failed the acceptance cutoff. This study keeps the candidate bank fixed and tests separate scores for object identity and localization quality. [Earlier ranking trace](syringe_object_verifier.md#ranking-trace)

## Protocol

Training uses the frozen RF-DETR candidate bank and the existing 143-image public fitting cohort. The identity head learns from clear background and proposals with an unambiguous syringe assignment. Its positive class includes recognizable fragments. Ambiguous assignments are skipped. The localization head learns reference IoU on associated proposals. Pair supervision uses two nearby boxes assigned to the same instance, with one usable localization and at least 0.12 difference in reference IoU. Each fit uses acquisition or instance weights to limit the effect of repeated boxes. Inputs include bounding-box size and aspect, RF-DETR confidence, corroboration across views, crop geometry and cached DINOv2-Small features at two scales.

The study compares the prior coupled score, geometry-only IoU regression, visual IoU regression, visual IoU regression with a same-instance difference loss, and the new identity score. The last two hybrid arms combine fitted identity with pair-supervised localization. One of those arms can transfer direct identity support from an overlapping alternative when the frozen duplicate model strongly favors a same-instance link. A transfer requires IoU of at least 0.35 and a predicted quality improvement greater than 0.04. Transfers use immutable input scores so support does not propagate through chains.

All arms preserve existing detections. The earlier overlap model controls conflicts. For each arm, a fixed grid of identity floors is searched on the 60 development images. Eligible rules must preserve every released match at IoU 0.50 and 0.75, avoid additional false positives in every image and preserve or improve each count error. Development chooses the best eligible gain, with strict IoU and count as tie breakers. The prospective data are scored after the rule and weights are locked. Qualification also requires at least one new valid match on the reserve and preservation on every replay cohort.

The public reserve contains eleven distinct reviewed acquisitions. Seven show one syringe each, including occlusion by a hand, a low-light example and surgical instruments around a syringe. Four show laboratory handling or specimen tubes. Published DVIDS attribution, source URLs, original-byte hashes and annotation coordinates are recorded. A visual reviewer labeled extents before inference. The source audit screens prior contributors, URLs, exact bytes and perceptual hashes, including previous holdouts and Fil Rouge fingerprints. The independent reserve has no crowded positives. Earlier dense images are included in separate replay cohorts. [Reserve references and exclusion reasons](../../../artifacts/detection_hillclimb/syringe_box_quality_reference.json)

## Fitting and development

The identity fit uses 2,236 labeled boxes from 116 acquisition groups. They include 260 usable boxes, 313 fragments and 1,663 backgrounds. Localization uses the 573 assigned boxes, and 877 same-instance pairs from 70 groups supply the optional pair loss. Of these pairs, 724 compare a usable localization with a fragment below IoU 0.50. No image ID, acquisition group, exact source URL or byte hash is shared across fit, development and reserve. Fil Rouge is outside fit, calibration and augmentation.

On the 254 same-instance development pairs, the earlier coupled score orders 87.8% correctly. Geometry IoU scores 87.0%, visual IoU scores 89.4%, and pair-supervised IoU scores 88.6%. Identity alone ranks 65.4% of these pairs correctly. Mean absolute IoU errors are 0.188, 0.211 and 0.243 for geometry, visual and pair supervision, respectively.

The development release matched 15 syringes and returned seven false positives. The prior calibrated recovery and the selected identity-plus-pair policy each matched 17 and returned seven false positives. The new policy gives the exact count on 39 of 60 images, one more than the prior calibrated rule. No additional match reaches IoU 0.75. Selection used detection and counting outcomes. The pair-ordering score is a separate diagnostic. [Training counts, model weights and frozen selection grid](../../../artifacts/detection_hillclimb/syringe_box_quality_policy.json)

## Independent results

On the 43 same-instance reserve pairs, visual IoU and pair-supervised IoU each order 39 correctly, compared with 37 for the earlier score. Geometry IoU orders 32 correctly. The visual regressor has the lowest absolute IoU error at 0.151. The pair-supervised regressor reaches 0.185. The pair-ordering sample contains repeated comparisons from the same acquisitions. The two additional correct orderings give preliminary evidence for visual localization, with equal counts for the two fitting objectives.

| Reserve policy | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 | Exact count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Released detector | 3 | 5 | 4 | 0.375 | 0.429 | 0.400 | 0.275 | 0.118 | 10/11 |
| Previous calibrated recovery | 4 | 6 | 3 | 0.400 | 0.571 | 0.471 | 0.341 | 0.148 | 8/11 |
| Previous active probe | 4 | 11 | 3 | 0.267 | 0.571 | 0.364 | 0.341 | 0.148 | 5/11 |
| Earlier identity and pair-supervised IoU | 4 | 6 | 3 | 0.400 | 0.571 | 0.471 | 0.341 | 0.148 | 8/11 |
| Fitted identity and pair-supervised IoU | 4 | 6 | 3 | 0.400 | 0.571 | 0.471 | 0.341 | 0.148 | 8/11 |

The reserve had one incumbent detection in every positive image. Four covered their reference poorly enough to miss IoU 0.50, so exact count masked those localization failures. The selected arm adds a candidate at IoU 0.778 on the Carzis photo and another at IoU 0.191 on the Rodriguez photo. The first recovers a localization match, and both increase the count on a photo that already contained one detection. Candidate-bank inspection finds a usable replacement for each of the four poorly localized incumbents, three above IoU 0.75. The ceiling uses the annotations and measures opportunity rather than detector performance. [Reference-assisted candidate ceilings](../../../data/syringe-box-quality/proposal-ceiling.json)

On the eight earlier object-verifier photos, each IoU regressor combined with the earlier identity score replaces the poor recovery choice with a correct box in the Stennis case. This changes the earlier recovery from one match and two false positives to two matches and one false positive. The second missed vaccination is still missed. On the seven other replay images containing crowded scenes, the selected arm restores two matches and adds two false positives. Its extra false positives and count regressions prevent deployment.

**Status.** The identity and localization comparison was executed locally. Model fitting and frozen evaluation reproduced byte for byte. All 181 tests passed, including four new checks for same-instance supervision, localization ordering, identity gating and duplicate protection. The reserve covers isolated syringes and includes one visual reviewer. Crowded transfer remains unverified by this reserve. [Frozen per-image outcomes and gates](../../../artifacts/detection_hillclimb/syringe_box_quality_results.json) · [Comparison figure](../../../data/syringe-box-quality/comparison.html)

## Replacement diagnostic

The previous relocalization geometry screen admits at least one box at reference IoU 0.50 or higher for each of the four poorly localized incumbents. The fixed duplicate model's direct same-instance signal at its existing distinctness cutoff of 0.10 retains usable alternatives in three of the four scenes. Among those three, ordering the linked boxes by visual IoU chooses a usable box in one scene and a fragment in two. Applying that ordering to an already correct detection also exposes a candidate with reference IoU 0.051 for an incumbent at IoU 0.886.

This is a retrospective audit of available links and box scores. It supports fitting a replacement decision on pairs of incumbent and proposed boxes, with explicit ownership and improvement supervision. A replacement should keep the same detection slot, exclude ambiguous claims from nearby syringes and preserve strong existing matches. The four audit cases contribute regression evidence for a future policy. Model fitting and threshold selection need separate acquisitions. [Candidate links and geometric eligibility](../../../data/syringe-box-quality/identity-link-audit.json) · [Earlier count-preserving study](count_preserving_relocalization.md)

## Reproduction

Run from the repository root after restoring the pinned local model weights and proposal archives from the overlap and object verification studies. Restore the public reserve rasters at their recorded paths and verify their hashes against the reference. The first two commands require fresh output paths. The current frozen policies can be rescored directly with their matching inputs.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.experiments.ranking.syringe_box_quality_reserve reference
.venv/bin/python -m scripts.experiments.ranking.syringe_box_quality freeze
data/rfdetr/env/bin/python -m scripts.experiments.ranking.syringe_box_quality_reserve proposals
data/sam3/xpu-env/bin/python -m scripts.experiments.ranking.syringe_box_quality_reserve release
.venv/bin/python -m scripts.experiments.ranking.syringe_box_quality_reserve features
.venv/bin/python -m scripts.experiments.ranking.syringe_box_quality score
.venv/bin/python -m scripts.experiments.ranking.syringe_box_quality_figure
.venv/bin/python -m pytest -q
~~~
