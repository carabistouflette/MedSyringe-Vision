# Calibrating the completeness score

The [prior joint-selection study](complete_syringe_selection.md) recovered the six visible syringes on an already exposed photo when candidate prioritization used only the body-completeness score. This follow-up tests whether the same score can support an automatic operating point on public acquisitions.

## Design

Both arms reuse the same learned head and RF-DETR candidate bank. The raw detector confidence supplies the same configurable candidate prefilter in both arms. The pure arm uses the completeness score for ranking and acceptance. The blended arm uses 70% completeness and 30% detector confidence. The arms reuse greedy selection, the same bounded candidate list, the original incumbent predictions and either conventional overlap or the frozen axis-aware conflict rule. The active diagnostic uses an identical rule for both scores. This keeps evaluation geometry, candidate generation and instance suppression fixed while measuring the scoring change.

The grid contains 612 configurations from 60 public development photos. The release guard requires zero per-image increases in false positives or absolute count error and preserves matched syringe identities. Both score families choose conventional overlap and the same raw prefilter of 0.12. The selected pure and blended rules add two matched syringes on development, retaining the same seven false positives. Scores are on different scales, so their frozen acceptance floors are 0.85 and 0.65. The blended arm wins the predeclared tie break. [Policy and full grid](../../../artifacts/detection_hillclimb/completeness_calibration_policy.json)

The seven newly reviewed reserve acquisitions have five annotated syringes and three pipetting or labware negatives. The reserve includes two partially occluded syringes held in one glove. It has limited coverage of dense crossing scenes. The earlier crossing and six-syringe photos serve as separate exposed checks. One reviewer annotated the new photos before inference. The exact previews, contributors and source URLs are recorded. Source metadata, exact hashes and rotated or mirrored perceptual hashes were checked against earlier development and evaluation photos. Fil Rouge was excluded from fitting and calibration. [Reserve annotations and asset hashes](../../../artifacts/detection_hillclimb/completeness_calibration_reserve.json) · [Gloved pair by Laura Beauty Designer](https://www.pexels.com/photo/black-and-white-close-up-of-syringes-in-gloved-hand-36675186/) · [Pipette photo by Jorge Sepúlveda](https://www.pexels.com/photo/anonymous-lab-worker-using-pipette-for-injecting-sample-into-flask-4299424/) · [Pexels license](https://www.pexels.com/license/)

## Frozen outcomes

| New reserve at IoU 0.50 | Correctly matched / 5 | False positives | Exact counts / 7 |
|---|---:|---:|---:|
| Released detector | 2 | 2 | 6 |
| Calibrated pure | 2 | 2 | 6 |
| Calibrated blend | 2 | 3 | 7 |
| Active pure probe | 3 | 2 | 7 |
| Active blended probe | 3 | 2 | 7 |

The calibrated blend adds one wrong box in the two-syringe photo. The reported count becomes exact while the matched count stays unchanged. The pure rule abstains. Both active probes recover the same valid second syringe with no added false positives in the new reserve. The probes also recover one syringe on the preceding five-image cohort. They fail the historical guard. On the seven exposed acquisitions, the pure probe adds two true detections and three false positives, while the blended probe adds one true detection and four false positives. On the older 18-image archive, each adds two false positives. No operating point meets the release requirements. [Frozen evaluation](../../../artifacts/detection_hillclimb/completeness_calibration_results.json) · [Older archive guard and candidate trace](../../../artifacts/detection_hillclimb/completeness_calibration_diagnostic.json)

The raw RF-DETR bank contains IoU 0.50 alternatives for all three missed references. The highest-scoring valid pure candidate for the missed neighbor scores 0.593, below the frozen pure floor of 0.85. The other two missed matches are on images where the released detector already has one box. These are localization errors with a correct count. All 612 development decisions and the two frozen development operating points reproduced exactly. The local suite passed 157 tests. The added test verifies that both scoring modes preserve detector scores and the raw prefilter while allowing a different order.

## Reproduction

Restore the public previews specified in the frozen manifest and the earlier proposal caches. Run from the repository root.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.completeness_calibration freeze
.venv/bin/python -m scripts.completeness_calibration_reserve freeze
data/rfdetr/env/bin/python -m scripts.completeness_calibration_reserve collect
data/sam3/xpu-env/bin/python -m scripts.completeness_calibration_reserve baseline
.venv/bin/python -m scripts.completeness_calibration score
.venv/bin/python -m scripts.completeness_calibration_diagnostic
.venv/bin/python -m pytest -q
~~~
