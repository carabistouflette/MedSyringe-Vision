# Learning when to add, replace or reject a syringe box

The previous completeness experiment added an unmatched box to a photo that already had one valid detection. The count became correct by coincidence. This experiment separates the decision to add a physical instance from the decision to improve localization of an existing one. [Previous calibration](completeness_calibration.md)

## Protocol

The study reuses the frozen RF-DETR Nano proposal generator and the released hybrid detector. RF-DETR provides full-image, tiled and bounded recrop proposals. Two acquisition-weighted multinomial logistic heads share the same proposal pool. The geometry head sees detector scores, crop scale, cross-view agreement and the proposal's relationship to existing boxes. The visual head also sees image-derived axis and body profiles, including differences between the proposed and incumbent crops.

Fit supervision uses 143 images from 116 public groups. A proposal represents a reference when its IoU is at least 0.50 and leads the competing reference by at least 0.08. It receives the addition label when that identity is missing from the simulated incumbent set. It receives the replacement label when it improves the box assigned to that identity by at least 0.08 IoU. Positive labels also require preservation of observable reviewed landmarks where supplied. Remaining proposals receive the rejection label. The states mix archived detector boxes, omitted incumbents, and controlled geometric perturbations. The 60,529 actions comprise 426 additions, 2,187 replacements and 57,916 rejections. Repeated states share their parent acquisition. They do not constitute independent photos. Original Fil Rouge images and their derivatives stay outside fit and calibration. [Serialized heads, acquisition counts and development grid](../../../artifacts/detection_hillclimb/syringe_action_verifier_policy.json)

Calibration uses 60 separate public development photos from 23 groups. The 572 configurations cover add-only, replace-only and combined modes, with two duplicate-handling rules and bounded action floors. The acceptance guard preserves prior matched instance identities at both IoU 0.50 and 0.75, and prevents increases in false positives and absolute count error in any image. A replacement preserves the incumbent slot and count, while an addition appends a new slot. The research evaluator applies box changes only. An application integration would also need to regenerate attributes from the new crop.

Seven public acquisition groups were reserved for transfer. They contain six visible syringes over four positive images and three labware negatives. Two positive views contain adjacent devices. One reviewer drew the visible extents before running either detector on the cohort. One photo was visually screened during a previous study and its predictions were first collected here. Contributor, source URL, exact hash and perceptual overlap checks passed against the project history. Known author aliases were treated as conflicts. The earlier calibration and dense-syringe cohorts are exposed replays. [Frozen references and exclusions](../../../artifacts/detection_hillclimb/syringe_action_verifier_reserve.json) · [Tray photo by cottonbro studio](https://www.pexels.com/photo/silver-tray-with-syringes-7581089/) · [Labware photo by Jonathan Borba](https://www.pexels.com/photo/scientist-with-a-test-tube-and-glass-container-in-her-hands-15831834/) · [Pexels license](https://www.pexels.com/license/)

## Development results

| Active policies at IoU 0.50 | True positives | False positives | Exact count / 60 | IoU 0.75 matches gained / lost |
|---|---:|---:|---:|---:|
| Released detector | 15 | 7 | 37 | 0 / 0 |
| Geometry add-only | 20 | 20 | 36 | 0 / 0 |
| Geometry replace-only | 15 | 7 | 37 | 1 / 2 |
| Geometry combined | 20 | 20 | 36 | 1 / 2 |
| Visual add-only | 19 | 12 | 42 | 0 / 0 |
| Visual replace-only | 15 | 7 | 37 | 1 / 1 |
| Visual combined | 19 | 12 | 42 | 1 / 1 |

All 560 development configurations that made a change failed at least one guard. The visual features improved the trade-off for additions and reduced losses at 0.75 during replacement. The six active diagnostic policies were selected by development F1 with localization and action-count tie breaks, then bound to the public reserve before inference. They are exploratory because they had already failed the development gate. [Frozen diagnostic operating points](../../../artifacts/detection_hillclimb/syringe_action_diagnostic_policy.json)

## Transfer results

| Seven new acquisitions, IoU 0.50 | TP / 6 | FP | Precision | Recall | F1 | AP50 | AP50–95 | Exact count / 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Released | 4 | 0 | 100.0% | 66.7% | 80.0% | 66.3% | 56.1% | 6 |
| Geometry add-only | 5 | 3 | 62.5% | 83.3% | 71.4% | 78.4% | 63.3% | 6 |
| Geometry replace-only | 4 | 0 | 100.0% | 66.7% | 80.0% | 66.3% | 41.3% | 6 |
| Geometry combined | 5 | 3 | 62.5% | 83.3% | 71.4% | 78.4% | 47.9% | 6 |
| Visual add-only | 5 | 2 | 71.4% | 83.3% | 76.9% | 78.4% | 63.3% | 6 |
| Visual replace-only | 4 | 0 | 100.0% | 66.7% | 80.0% | 66.3% | 42.1% | 6 |
| Visual combined | 5 | 2 | 71.4% | 83.3% | 76.9% | 78.4% | 48.7% | 6 |

The released system detects neither device in the medical-instrument scene. The active visual policies add two boxes on that image. One localizes the dental cartridge syringe at 0.783 IoU. The other reaches just 0.339 against the same reference. The correct image-level count conceals a box that mixes the side of an already localized syringe with nearby tools and a second missed syringe. The latter has an RF-DETR proposal with 0.666 IoU and a visual addition score of 0.823. The false partial box scores 0.993, above the recovered correct box at 0.987. No single floor on this score could retain the correct box and reject the partial box in that scene. An additional false positive lands on labware in a negative acquisition. This trades one corrected count for one degraded count across the reserve. [Diagnostic action trace and missed-reference alternatives](../../../artifacts/detection_hillclimb/syringe_action_diagnostic_results.json) · [Medical instrument photo by Jimmy Liao](https://www.pexels.com/photo/table-with-prepared-syringes-and-other-medical-instruments-17112252/)

The active visual replacement policy changes three localizations on the reserve. IoU increases from 0.940 to 0.953 on one photo and from 0.889 to 0.936 on another. A third drops from 0.969 to 0.684 and loses its strict match. This leaves precision, recall and count at IoU 0.50 unchanged while degrading higher-IoU coverage. On each of the two exposed cohorts, every active additive arm adds one false positive and recovers zero matches. The release qualification is false. The frozen development-safe finalists all abstain. [Qualification and exposed-cohort results](../../../artifacts/detection_hillclimb/syringe_action_verifier_results.json)

The main limitation is action supervision generated by perturbing known boxes. It yields few independent examples of clashing physical instances. The reserve contains seven visually reviewed acquisitions and sparse crowded coverage, so the results describe observed failure modes with broad transfer uncertainty. Candidate availability is established for both misses in the new reserve. Better selection requires evidence that the box spans the visible assembly and that its axis belongs to a distinct device. Audit labels could distinguish a whole instance, a fragment, and boxes that mix parts from different syringes. In this study the existing visual features only partially reduce that error. AP uses the inherited detector score ordering after policy selection, with unchanged scores on moved boxes.

Both fitted heads, all 572 development trials and the SHA-256 of the fit ledger reproduced exactly on the same environment. Both transfer result files reproduced byte for byte from the frozen evidence. The local suite passed all 163 tests. Six new tests cover fit labels, duplicate resolution, slot preservation, and localized regressions hidden by matching counts. RF-DETR proposal inference ran on the local CPU. The release predictions used the Intel XPU.

## Reproduction

Restore the downloaded public assets and the prior proposal, release and reference caches. The commands run from the repository root. The freeze stages require fresh output paths. Their existing records reject changes to inputs and code. Use the existing locks when replaying scoring on this checkout.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.experiments.ranking.syringe_action_verifier freeze
.venv/bin/python -m scripts.experiments.ranking.syringe_action_diagnostic freeze
.venv/bin/python -m scripts.experiments.ranking.syringe_action_reserve freeze
data/rfdetr/env/bin/python -m scripts.experiments.ranking.syringe_action_reserve collect
data/sam3/xpu-env/bin/python -m scripts.experiments.ranking.syringe_action_reserve baseline
.venv/bin/python -m scripts.experiments.ranking.syringe_action_verifier score
.venv/bin/python -m scripts.experiments.ranking.syringe_action_diagnostic score
.venv/bin/python -m pytest -q
~~~
