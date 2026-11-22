# Object verification for syringe recovery

The previous recovery study found two failure modes after RF-DETR and overlap ranking. A mechanical pipettor was accepted, and the released crop veto rejected a correct dental cartridge syringe during regression replay. The verifier in this study can reject only newly proposed boxes. The released detections and their scores are copied intact. [Earlier recovery study](overlap_ranking.md#fixed-appearance-veto-diagnostic)

## Protocol

All arms share the same RF-DETR views, recrops, proposal ranking and duplicate policy. The calibrated recovery route uses the earlier contrastive rank cutoff of 0.97. The active probe uses its previously recorded cutoff of 0.75. Each route is evaluated with unfiltered additions, the issued SigLIP 2 prompt veto, a targeted SigLIP 2 prompt bank, a binary DINOv2 head, a typed DINOv2 head, and a typed head combining both feature sources.

The typed models distinguish a syringe, a pipette, a specimen tube or holder, a loose needle, and other objects. Each fitted head uses regularized logistic regression with acquisition and class weights. Tight and context crops use the same frozen local DINOv2 and SigLIP 2 weights across arms. Typed scores are the syringe logit minus the highest alternative logit. The targeted prompt arm uses the best syringe margin over alternative categories across the two crops.

Training samples consist of reviewed reference crops and RF-DETR proposals with unambiguous matching. A proposal must achieve IoU of at least 0.50 with a syringe and lead the competing instance by at least 0.08 to receive a positive label. Uncertain proposals with syringe overlap of at least 0.12 are skipped. A sufficiently overlapping reviewed lookalike gives a typed negative label. The other category includes background and miscellaneous objects. For supplemented scenes with partial lookalike labels, only explicitly annotated crops enter training.

The existing fit and development splits retain their previously assigned acquisition groups. Two newly reviewed negative acquisitions supply labeled crops. The new transfer cohort contains eight public image groups from distinct contributors. Three include a single observable syringe, including a dental cartridge syringe and small hand-held syringes in vaccination scenes. Five show pipettes or sample tubes. References, provenance, and file hashes were fixed before running either detector on these scenes. Known contributor, URL, exact byte, and close perceptual matches to earlier datasets were excluded. Bounding boxes were produced by one visual reviewer. An additional cohort replays eleven previously evaluated photos. Fil Rouge data were kept out of training, calibration and augmentation. [Source credits, screening exclusions and frozen references](../../../artifacts/detection_hillclimb/syringe_object_verifier_references.json)

Thresholds and the one eligible challenger are selected on development groups. A cutoff may discard a candidate only if all matches available without the veto survive at IoU 0.50 and 0.75 on development. Among eligible cutoffs, fewer false detections take priority. Promotion requires a recovered syringe on the new cohort and zero per-image false-positive or count regressions against the release in both cohorts. Rejection of an otherwise correct recovery at either IoU is also a failure. The holdout results cannot change the cutoffs.

## Fitting and development

Three logistic heads were fitted locally using 457 crops from 145 images in 118 acquisition groups. Of these, 310 are detector proposals, 116 are annotated syringe crops, and 31 are annotated lookalikes. The typed labels total 285 syringes, 8 pipettes, 4 tubes, 3 loose needles, and 157 other crops. The rare classes represent only four pipette, three tube, and two loose-needle acquisitions. This limits the evidence for broad class transfer. The automated partition audit found zero overlaps in reference IDs, acquisition groups, image hashes or source URLs across fit, development and reserve.

The 60-image development set contains 44 reference syringes. The released detector returned 15 true positives and 7 false positives. Calibrated recovery improved this to 17 true positives and 7 false positives. Its two added boxes were valid, so the chosen cutoff for each new verifier left both in place. The active route without a veto returned 19 true positives and 12 false positives. At fixed recall, the binary and typed DINOv2 heads dropped two false positives each, and the hybrid dropped four. The remaining additional false positive blocked selection. The issued SigLIP 2 veto on the active route returned 18 true positives and 10 false positives. Neither route produced a new verifier eligible for promotion. [Fitted weights, partition audit and calibration trials](../../../artifacts/detection_hillclimb/syringe_object_verifier_policy.json)

The issued SigLIP 2 veto was reproduced on the prior study's 18 exposed candidates. All 18 accept and reject decisions agreed with the earlier implementation. The maximum absolute margin difference was below 0.00000004. [Earlier veto margins](../../../data/overlap-ranking/appearance-gate-diagnostic.json)

## Independent transfer and regression

The table reports one-to-one matches at IoU 0.50. The replay cohort had 19 reference syringes in 11 images and was already exposed in earlier experiments. It contains dense and overlapping scenes. The eight-image reserve has three isolated reference syringes. It provides evidence on these acquisitions. It has limited power to estimate generalization and does not test crowd transfer independently.

| Cohort | Policy | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50–95 | Exact count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| New public | Released detector | 0 | 1 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4/8 |
| New public | Calibrated recovery with any verifier | 1 | 2 | 2 | 0.333 | 0.333 | 0.333 | 0.112 | 0.090 | 6/8 |
| New public | Active recovery with issued veto | 1 | 3 | 2 | 0.250 | 0.333 | 0.286 | 0.112 | 0.090 | 5/8 |
| New public | Active recovery with typed hybrid | 1 | 4 | 2 | 0.200 | 0.333 | 0.250 | 0.112 | 0.090 | 5/8 |
| Exposed | Released detector | 6 | 1 | 13 | 0.857 | 0.316 | 0.462 | 0.287 | 0.165 | 6/11 |
| Exposed | Active recovery without veto | 12 | 13 | 7 | 0.480 | 0.632 | 0.545 | 0.444 | 0.231 | 3/11 |
| Exposed | Active recovery with issued veto | 11 | 6 | 8 | 0.647 | 0.579 | 0.611 | 0.456 | 0.235 | 6/11 |
| Exposed | Active recovery with typed hybrid | 10 | 7 | 9 | 0.588 | 0.526 | 0.556 | 0.421 | 0.224 | 5/11 |

The calibrated route recovered the dental cartridge syringe at IoU 0.875 and added one localization error on the Stennis vaccination image. On the active route, the binary and typed DINOv2 heads removed the correct dental recovery. The hybrid preserved it and removed zero false positives. The released detector itself already accepted a pipette in the wound-lab image. That false detection lies outside the scope of a verifier for additions.

The candidate bank contained a valid box for each reference syringe in the reserve at IoU 0.50. The best overlaps were 0.875, 0.541 and 0.894. The released routes admitted at most one valid box. The Stennis proposals that passed recovery ranked at IoU 0.440 and 0.363, and the third reference received zero offers. These two misses occurred before object verification. [Candidate-pool ceilings](../../../data/syringe-object-verifier/proposal-ceiling.json)

In the exposed replay, the hybrid removed six false positives and two valid overlapping syringes in the Aviano scene relative to active recovery without a veto. The issued veto removed seven false positives and the valid dental syringe. The NCI pipettor still passed both vetoes. The hybrid margin was 1.689 for the NCI pipettor and 1.491 for the correct dental syringe in the new reserve. On this observed pair, the ordering prevents a single cutoff from removing the pipettor while preserving that dental recovery.

**Status.** The comparison was implemented and executed locally. None of the tested rules met the deployment criterion. The frozen policy and both cohort evaluations reproduced byte for byte from the saved input features. The test suite passed 177 tests, including five checks for typed labels, preservation of incumbent detections, calibration, localization and acquisition leakage. The findings apply to syringe detection and counting. [Full metrics, eligible matches, false positives, missed boxes and veto reasons](../../../artifacts/detection_hillclimb/syringe_object_verifier_results.json) · [Comparison figure](../../../data/syringe-object-verifier/comparison.html)

## Ranking trace

A subsequent local replay reproduced every saved recovery decision on development and on the eight exposed reserve images. It traces each proposal through the rank cutoff, conflict with an incumbent, candidate budget, and competition with other proposals. On development, the released detector missed 29 reference syringes. Under calibrated recovery, the bank contained at least one usable proposal for 25 of those misses. The rule recovered two. For each of the other 23, the box with highest reference IoU was below the ranking cutoff. Under active recovery, four were recovered. The best box for 19 remaining misses fell below the cutoff and for two others lost a conflict. Four misses in each route had no valid box in the bank.

The two missed vaccinations show different errors. In the Stennis image, a candidate at IoU 0.539 reaches rank 0.992 and passes the calibrated cutoff. A competing box at IoU 0.440 wins the ordering and suppresses it. In the defense headquarters image, the best candidate reaches IoU 0.894 and rank 0.393. All usable candidates in that frame fall below both frozen cutoffs. These observations support testing a separate estimate of box completeness alongside syringe identity. The former should order alternate views of the same proposed instance. The latter should control whether that instance is counted. A new experiment needs new acquisition groups for independent evaluation because the reserve above was used for this diagnosis. [Per-candidate trace](../../../data/syringe-object-verifier/ranking-trace.json)

## Reproduction

Run from the repository root with the existing RF-DETR proposal bank, overlap-ranking artifacts and locally pinned weights. The reference manifest specifies the exact download URL and expected SHA-256 for each accepted photo. Restore the source rasters before running inference. For a complete reference freeze, also restore the rejected reviewed source images listed in the reference script. The freeze commands expect new output paths. Existing frozen results can be rescored.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.syringe_object_references
.venv/bin/python -m scripts.syringe_object_features fit development exposed
.venv/bin/python -m scripts.syringe_object_verifier freeze
data/rfdetr/env/bin/python -m scripts.syringe_object_reserve collect
data/sam3/xpu-env/bin/python -m scripts.syringe_object_reserve baseline
.venv/bin/python -m scripts.syringe_object_features fresh
.venv/bin/python -m scripts.syringe_object_verifier score
.venv/bin/python -m scripts.syringe_object_chart
.venv/bin/python -m scripts.syringe_ranking_trace
.venv/bin/python -m pytest -q
~~~
