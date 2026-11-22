# Syringe part geometry and appearance verification

The previous recovery policy missed a tray of ten prefilled syringes and confused two pipette photos. Whole-object SAM proposals were available for the tray. This study tests whether anatomy and crop appearance can improve acceptance of the same proposal pool.

## Annotation and separation

Seventeen public fitting images have manually marked centers for the visible barrel, finger flange, plunger rod, and rubber seal. The review identifies 17 barrel centers, eight flanges, 11 rods and 11 seals. Hidden or ambiguous pieces have null labels. A separate clinic image has four anchors on the sharp foreground syringe, while the three blurred devices retain only whole-object boxes. The new archive includes a CC-licensed Pasteur pipette photo and existing fitting negatives with detached needles, pipettes and pen-like tools. Subject identification uses visible physical geometry and the source record. No clinical quantities were inferred. [Frozen reference data](../../../artifacts/detection_hillclimb/part_verifier_references.json) · [Part review examples](../../../data/part-verifier/annotated-landmarks-1.jpg) · [Pasteur pipette source](https://commons.wikimedia.org/wiki/File:Pasteur_pipettes.jpg)

The ten-syringe tray, its paired pipette negatives, and all 24 prior crowd references remain regression data. Two new public lookalike images admitted under the author, source URL, byte hashing, and transformed visual matching checks constitute the negative prospective controls. Their capture methods remain unverified. Three separate source photographs already have a matching contributor somewhere in project history. Those photographs are reported as a transfer challenge. Their distinct rasters show eight syringes and three pipettes across three photographs. An additional official public-domain image of two syringe assemblies is frozen before its separate run. Fil Rouge appears in the historical exclusion fingerprints and never enters the fit set. [Clinic photo](https://commons.wikimedia.org/wiki/File:Syringes_rest_on_a_clean_table_in_a_clinic.jpg) · [Earlier-source syringe lineup](https://commons.wikimedia.org/wiki/File:Syringe_with_needle_(2).JPG) · [Additional public-domain acquisition](https://www.dvidshub.net/image/6479379/wright-patt-phase-1-covid-vaccinations)

## Experimental design

SAM 3 ran locally on Intel Arc at 756 pixels. The body candidate stream merges the syringe and prefilled-syringe prompts with mask suppression. Each candidate receives the same geometry and DINOv2 crop features. The anatomical rule looks for evidence linking the barrel and the plunger rod and for a finger flange between them. Missing observed points never count as labeled negative anatomy.

Three logistic verifiers use prompt and relative geometry features, frozen DINOv2-Small crop embeddings, or both. Four acquisition-grouped folds set a conservative operating point for each model before any new prediction is collected. There are 36 fitting photos spanning 18 contributing groups with proposals, with 15 positive and 27 negative proposals. Four, four and three true proposals respectively survive at each out-of-fold zero-false-positive operating point. These are low-coverage fitting estimates. A separate geometric rule and the unverified SAM stream provide controls on the same proposals. The recovery adds to scenes with zero or one released detection and preserves all released objects. [Verifier coefficients, sources and checksums](../../../artifacts/detection_hillclimb/part_verifier_freeze.json)

## Evaluation

| Method | Matched / 91 | False positives | Exact counts / 27 |
|---|---:|---:|---:|
| Released engine | 34 | 1 | 12 |
| Unverified SAM recovery | 76 | 13 | 15 |
| Anatomical association rule | 66 | 6 | 13 |
| Learned part features | 56 | 5 | 13 |
| DINOv2 appearance | 34 | 1 | 12 |
| Combined arm | 34 | 1 | 12 |

The anatomical rule counts nine of ten on the earlier prefilled tray and adds nothing on the two earlier pipette cases. Across all regression scenes it still adds five false positives. The learned geometry arm recovers 22 matched instances, preserves all earlier matches, and adds four false positives. Appearance-only and combined heads leave the released instances unchanged. On the two fresh public lookalikes the anatomical rule and unverified SAM each add one false positive. The three fitted heads preserve the released outputs, including one existing false positive. On the known-contributor transfer the released detector recognizes six of eight. It already accepts two on the missed clinic scene, outside the current routing condition. [Full scene counts and failures](../../../artifacts/detection_hillclimb/part_verifier_results.json)

A separate exploratory appearance run adds 28 annotated positive crops and nine annotated distractor crops to the fitting candidate pool. The four grouped folds exclude every auxiliary crop from the validation acquisition. Three fitting proposals survive the zero-false-positive operating point, and no additional detection is accepted in the measured cohorts. [Reference-crop fit](../../../artifacts/detection_hillclimb/part_reference_crop_freeze.json) · [Replay](../../../artifacts/detection_hillclimb/part_reference_crop_results.json)

On the additional public-domain photo, both partly occluded syringes fall below successful localization at IoU 0.50. The released engine emits one unmatched box, which the additive variants preserve. Every tested learned verifier therefore yields zero recovered matches on this photo. Taken together, the new acquisition and hard-negative evidence fail the automatic release conditions. [New acquisition results](../../../artifacts/detection_hillclimb/part_verifier_new_acquisition_results.json)

## Plunger localization

At the fixed part score of 0.50, full-frame analysis places four of 11 seal centers within 10% of the syringe-box diagonal. Running the same prompts on manually cropped syringes increases barrel-center matches from eight to ten of 17 and reduces seal-center matches from four to zero of 11. On the separate clinic foreground syringe both views fail the seal-center criterion. A reliable seal position for automatic volume intervals remains unverified. [Matched landmark audit](../../../artifacts/detection_hillclimb/part_crop_diagnostic.json)

The full project test suite passed 124 tests. It includes checks for the barrel-flange-rod order and for unrelated parts outside a candidate box.

## Reproduce

~~~sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.experiments.parts.part_verifier_report
.venv/bin/python -m scripts.experiments.parts.part_reference_crop_study measure
data/sam3/xpu-env/bin/python -m scripts.experiments.parts.part_verifier_new_acquisition released
data/sam3/xpu-env/bin/python -m scripts.experiments.parts.part_verifier_new_acquisition measure
data/sam3/xpu-env/bin/python -m scripts.experiments.parts.part_crop_diagnostic
.venv/bin/python -m pytest -q
~~~

The fitting replay uses the published source and feature checksums. The report entry point converts NumPy landmark counts to plain JSON integers and reuses the frozen inference and metric implementations. To reconstruct the feature bank in a new workspace, render the fitting contact sheets with `python -m scripts.experiments.parts.part_verifier_data --sheets data/part-verifier`, preserve the manual landmark list, and run collection and fitting on a separate copy before scoring the held roles. A rehearsal reproduced all three contact sheets and their position maps byte for byte.
