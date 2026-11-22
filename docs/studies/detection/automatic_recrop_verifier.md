# Verifying automatic second-pass syringe detections

The earlier recrop pilot recovered a usable box for one missed syringe in a pump image but assigned it a score of 0.032. This study trains crop verifiers and measures whether they can accept new complete syringes while controlling repeated counts and pipette detections. [Recrop pilot](syringe_assembly_and_thin_views.md#automatic-recrop-trigger-pilot)

## Data and frozen sequence

The study uses 143 public fitting images from 116 groups and 60 public development images from 23 groups. The 18-image archive contains previously examined scenes and serves as a regression check. Eight fresh public acquisitions contain five syringes and three negative scenes with pipettes and glassware. The fresh annotations and image checksums were frozen before policy selection. Fil Rouge contributed no fitting, selection or augmentation images. [Reserve provenance, coordinates, and exclusions](../../../artifacts/detection_hillclimb/recrop_verifier_reserve.json)

Exact SHA-256, source URL, acquisition, contributor and rotated or mirrored perceptual hashes were screened against the project's protected sources and existing study cohorts. A historical two-syringe photo and three historical negatives were removed. Additional crowded and blurred photos had uncertain assembly boundaries and were excluded before inference. The fresh cohort consequently assesses narrow, partly occluded single devices and pipette negatives. Only the archive covers crowded overlapping devices in this experiment. The same reviewer visually annotated the fresh boxes and inspected the sources. This small challenge cohort does not estimate performance across public imagery or Fil Rouge. [Example positive by Wesley Farnsworth](https://www.dvidshub.net/image/6630055/col-miller-visits-deployed-airmen-detroit-michigan) · [Example negative by Bruce Morgan](https://www.dvidshub.net/image/9221790/george-washington-conducts-fueling-sea-with-badlands-trader)

## Controlled comparison

- The pinned RF-DETR Nano produces the first-pass seeds. The second pass tests budgets of three and six score-ranked windows and six ranked windows plus two spatially diverse fallback windows. Every window comes from detector output and is clipped to the frame. A common, bounded candidate pool retains raw-score and repeat-supported proposals. Proposal ceilings use one-to-one reference matching at IoU 0.50.
- The five ranking methods use raw detector scores, repeat support across crops, metadata and pixel-supported barrel axis, DINOv2 crop embeddings alone, or a hybrid of the geometry and embedding features. Multinomial logistic classifiers learn four labels. Reviewed references and archived fit-only part reviews supply complete syringes, lookalikes and fragments. Additional detector matches, mined backgrounds and constructed distal crops provide weak supervision. Fitting weights balance each class by acquisition.
- The candidate identity policies compare conventional overlap, learned fragment veto, and a separate decision for two visually distinct but overlapping instances. Supplemental candidates only append boxes. The unchanged released predictions act as the control for every arm. Development selects score floors, crop budgets and identity policies with no lost matches, no new false positives per image, and no increase in per-image absolute count error. The fresh evaluation uses the frozen selection, with the archive as an additional guard.

## Development results

The three learned models used 1,583 examples. Class counts are 235 complete, 396 fragment, 163 lookalike and 789 background. Twenty-six fragments were visually reviewed and 339 were constructed from annotated axes or elongated reference boxes. A visual spot-check shows the distinction between complete and truncated assemblies is imperfect in some detector-derived fitting boxes. The examples include both hypodermic and dental devices. [Fitting contact sheet](../../../data/recrop-verifier/fitting-examples.jpg) · [Grouped supervision](../../../data/recrop-verifier/fit-supervision.json)

| Development ranking method | Best safe budget | Additional matched syringes | Added false positives |
|---|---:|---:|---:|
| RF-DETR score only | 3 | 0 | 0 |
| Cross-crop agreement | 8 | 3 | 0 |
| Detector metadata and pixel geometry | 3 | 2 | 0 |
| DINOv2 crop appearance | 3 | 1 | 0 |
| Geometry and appearance | 3 | 2 | 0 |

Selection compared 945 configurations. Cross-crop agreement with six primary seeds and two additional spatially diverse seeds wins, with 15 to 18 true positives and seven false positives in both runs. Exact image counts improve from 37 to 40 out of 60. Candidate coverage at IoU 0.50 rises from 39 to 41 out of 44 targets after merging the crop and first-pass banks. These are distinct controls. The coverage comparison uses raw RF-DETR proposals, and policy selection measures additions to the released detector. [Development grid, learned coefficients and metrics](../../../artifacts/detection_hillclimb/recrop_verifier_policy.json)

## Frozen evaluation

| Cohort | Matched syringes, control to selected | False positives, control to selected | Exact counts, control to selected |
|---|---:|---:|---:|
| Development | 15 → 18 / 44 | 7 → 7 | 37 → 40 / 60 |
| Previously examined archive | 4 → 4 / 20 | 4 → 5 | 12 → 11 / 18 |
| Fresh reserve | 3 → 3 / 5 | 2 → 2 | 8 → 8 / 8 |

The selected rule fails the release gate. The fresh precision, recall and F1 each remain at 60%. AP50 is 0.362 and AP50–95 is 0.134 in both runs. The archive produces one new false positive in a pipetting scene. Repeated crops from the same detector can reinforce the same distractor. The appearance-only challenger recovers one archival syringe and adds one false positive. The geometry and appearance hybrid recovers the same syringe and adds three archival false positives, including the previously studied needle, safety hub and vial region. It also adds two fragment detections on the fresh reserve. The geometry-only model adds a fresh false positive on a marker. The published runtime remains the control. [Frozen evaluation with per-photo additions](../../../artifacts/detection_hillclimb/recrop_verifier_results.json) · [Visual review of false detections](../../../data/recrop-verifier/false-positive-review.jpg)

All eight independent photos already had the right count. Two of the predicted boxes included too much surrounding material, yielding IoU 0.308 and 0.455. Reference-assisted inspection shows that first-pass RF-DETR already proposed an alternative near IoU 0.804 for each. The three-seed crop bank improves the best overlaps to 0.887 and 0.908. This identifies a count-preserving relocalization opportunity. The reference-assisted selection measures a ceiling, and a count-preserving automatic policy has yet to be qualified for these scenes. On the archival Chemyx image, the maximum budget supplies one usable box at IoU 0.521 and leaves the second at 0.444. The localized candidate scores only 0.152 for cross-crop agreement. [Post-evaluation localization audit](../../../artifacts/detection_hillclimb/recrop_verifier_error_audit.json)

The fresh median CPU crop pass costs 1.87 seconds for up to eight windows, measured with the unchanged detector running on the local XPU in parallel. This timing covers RF-DETR crops and their proposal extraction. Candidate encoding and scoring run separately. The split check finds zero overlaps of image ID, acquisition group, and exact checksum across fitting, development, and reserve. A second evaluation of the frozen policy and proposal banks reproduces the result JSON byte for byte. The full local test suite passes 146 tests including crop nesting, view support, supervision boundaries, additive fusion and extreme-width crop preprocessing.

## Count-preserving refinement preflight

The prior frozen relocalization rule was rerun on the eight previously evaluated photos with its original RF-DETR inputs, the three-crop bank, and their union. The original tiled inputs make zero changes. Both recrop variants change one existing box, from IoU 0.455 to 0.908, preserving all eight counts. Matches rise from three to four at IoU 0.50 and from one to two at IoU 0.75. The second candidate reaches IoU 0.887 and lies entirely inside its incumbent box. Its score of 0.299 falls below the old rule's 0.401 floor. The first-pass and second-pass candidate boxes agree with each other at IoU 0.778 and 0.761 on the two target cases. This is a retrospective replay of already inspected scenes. [Frozen rule replay and individual comparisons](../../../artifacts/detection_hillclimb/recrop_relocalization_preflight.json)

The follow-up trains a replacement policy on acquisition-separated pairs. It evaluates gains and target preservation at IoU 0.50 and 0.75 on a new public reserve with narrow single assemblies and overlapping neighbors. Any accepted crop change must feed the marking and liquid analyses during application integration. [Count-preserving relocalization study](count_preserving_relocalization.md)

## Reproduction

Use the existing CPU RF-DETR environment, local project environment and XPU released-detector environment. Restore the exact public preview bytes from the asset URLs in the manifest before reproducing the reference, and retain the public input archives from the preceding assembly study. The commands that freeze supervision or policy deliberately fail when the corresponding artifact already exists.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.experiments.relocalization.recrop_reserve freeze
data/rfdetr/env/bin/python -m scripts.experiments.relocalization.recrop_proposal_bank fit development archive
.venv/bin/python -m scripts.experiments.relocalization.recrop_verifier learn
data/rfdetr/env/bin/python -m scripts.experiments.relocalization.recrop_proposal_bank reserve
data/sam3/xpu-env/bin/python -m scripts.experiments.relocalization.recrop_reserve baseline
.venv/bin/python -m scripts.experiments.relocalization.recrop_verifier evaluate
.venv/bin/python -m scripts.experiments.relocalization.recrop_error_audit
data/rfdetr/env/bin/python -m scripts.experiments.relocalization.recrop_relocalization_preflight collect
.venv/bin/python -m scripts.experiments.relocalization.recrop_relocalization_preflight score
.venv/bin/python -m pytest -q
~~~
