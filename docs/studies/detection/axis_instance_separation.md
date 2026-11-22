# Separating overlapping syringe instances

The [relocalization study](count_preserving_relocalization.md) found two misses on a six-syringe photo with usable first-pass proposals. This experiment tests whether a pair classifier can allow an overlapping addition while preserving every incumbent detection.

## Method and data

The pipeline uses the pinned RF-DETR Nano proposal bank at a bounded candidate score floor. The classifier compares the existing pair verifier with a new verifier trained on body-axis evidence. It estimates the dominant axis from image gradients, samples the source crop along and across it, and looks for opposing longitudinal edges over several candidate barrel widths. The pair model uses centerline separation, edge support, orientation and the earlier geometry and pixel features. It learns four labels for the same assembly, its attached fragment, a different syringe and a distractor. The pixel profile estimates continuous edges. It does not identify a named plunger landmark.

Training used 1,382 fitting pairs. There are 698 same-assembly pairs, 318 fragments, 303 distractors and 63 distinct pairs from seven acquisitions. Existing visual fragment reviews are retained. Additional fitting pairs come from detector matches to unambiguous training references, synthetic oriented fragments and mined background. Acquisition and class weighting limits repeated evidence. Fil Rouge remains outside fitting and selection.

The first development pass allowed disjoint additions as well as overlap exceptions. It found no operating point with a positive gain under the per-image guard. Before running independent inference, the study restricted the learned policies to overlap exceptions. Conventional overlap suppression stays as the comparison. An overlap exception requires distinct-instance support against every overlapping incumbent and a fragment veto; near-identical boxes remain blocked. The grid selects on the existing 60 development photos using zero added false positives, no lost reference and no worse count per image. A separate fixed-score diagnostic compares active policies. [Fitted coefficients and development grid](../../../artifacts/detection_hillclimb/instance_separation_policy.json)

The independent reserve contains five public acquisitions and nine visually annotated syringes. It includes three parallel devices, a crossing tray of five, one syringe held in gloved hands, and two pipetting negatives. The crossing tray was visually screened in an earlier study and receives its first detector evaluation here. Its instance count is traceable by seal and barrel paths. Box edges are approximate where caps and rods overlap. One reviewer placed the boxes. The manifest binds authors, exact image bytes, source links and notes. Exact and transformed perceptual fingerprints, source metadata and contributors were checked against previous evaluation and training history. Stock photos are local evaluation assets. [Reserve references](../../../artifacts/detection_hillclimb/instance_separation_reserve.json) · [Crossing tray by Jan van der Wolf](https://www.pexels.com/photo/disposable-syringes-in-a-medical-container-33893209/) · [Parallel syringes by Olia Danilevich](https://www.pexels.com/photo/close-up-shot-of-syringes-5973354/) · [Pexels license](https://www.pexels.com/license/)

## Measured outcomes

No development operating point gains a match while meeting the release constraints. All safe finalists abstain. The fixed active probe runs only with at least two released detections. No fresh frame meets that condition. The released detector finds two of nine syringes with zero false positives and three exact counts out of five. A reference-assisted match finds four of five tray syringes and all three aligned syringes in the raw candidate bank. An exploratory one-incumbent replay using the corresponding development grid settings yields two true additions and no false positives for the conventional overlap arm. The learned overlap-only arms add no boxes. The exploratory result cannot qualify a policy that failed development selection. [Frozen comparison](../../../artifacts/detection_hillclimb/instance_separation_results.json) · [Proposal ceilings and activation replay](../../../artifacts/detection_hillclimb/instance_separation_diagnostic.json)

| Six-syringe diagnostic | Correctly matched | Added false positives | Reported count |
|---|---:|---:|---:|
| Released detector | 4 | 0 | 4 |
| Conventional overlap probe | 4 | 0 | 4 |
| Earlier identity probe | 4 | 1 | 5 |
| Body-axis identity probe | 5 | 1 | 6 |
| Body-axis probe with one annotation-identified error removed | 6 | 0 | 6 |

For the highest-ranked candidates for the two missing entities, distinct-instance scores rise from 0.269 and 0.331 to 0.801 and 0.900. The active axis verifier admits one true box and an incomplete proposal with detector score 0.402. The incomplete proposal is ranked ahead of the better-localized second syringe at 0.352, and suppresses it during sequential admission. Removing the known error from the proposal list allows both valid boxes through with the same rule. The final table row uses the annotations to remove that candidate and measures a diagnostic ceiling. This isolates complete-body ranking before suppression as a separate bottleneck. The active probes preserve the 18-image archive outcomes. [Per-image decisions and counterfactual](../../../artifacts/detection_hillclimb/instance_separation_diagnostic.json)

A separate SAM 3 probe produces plunger-rod prompt boxes with scores 0.621 and 0.616. Each lies almost entirely inside one of the two best missed-syringe proposals. Several other rod prompt boxes lie inside the erroneous fragment. Some high-scoring boxes also overlap incumbents. This is candidate-level evidence on one exposed scene and does not identify a validated plunger-to-instance assignment. [Plunger prompt evidence](../../../artifacts/detection_hillclimb/instance_separation_plunger_probe.json)

The full suite passes 153 tests. New tests cover paired edge response, the fragment veto, abstention on missing evidence and preservation of existing boxes. The cross-role identity, group, and image checks returned zero collisions. Repeated evaluation reproduced the frozen results byte for byte. The model stays experimental. [Follow-up on completeness and joint selection](complete_syringe_selection.md)

## Reproduction

Restore the exact preview bytes from the reference manifest and the archived RF and released outputs from the earlier studies. Run locally from the repository root.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.experiments.ranking.instance_separation train
.venv/bin/python -m scripts.experiments.ranking.instance_separation_reserve freeze
data/rfdetr/env/bin/python -m scripts.experiments.ranking.instance_separation_reserve collect
data/sam3/xpu-env/bin/python -m scripts.experiments.ranking.instance_separation_reserve baseline
.venv/bin/python -m scripts.experiments.ranking.instance_separation score
.venv/bin/python -m scripts.experiments.ranking.instance_separation_diagnostic
data/sam3/xpu-env/bin/python -m scripts.experiments.ranking.instance_separation_part_probe
.venv/bin/python -m pytest -q
~~~
