# Appearance verification before syringe recropping

The previous visual pair model restored three missed localization matches on a replay and changed boxes on two lookalike photos. This study compares whether a fixed appearance model can veto those edits while preserving useful recrops. [Prior replacement and appearance findings](pair_relocalization.md#appearance-diagnostic)

## Experimental design

All variants share the released incumbent boxes, RF-DETR proposals, frozen pair model and ownership scores. A veto can prevent a proposal from replacing a box. It preserves ownership evidence for competing slots so it cannot resolve an ambiguous association by discarding a competing claim. The number, IDs and detection scores of existing instances are preserved.

The fixed controls use the previous active visual pair setting. They apply no veto, the issued SigLIP 2 veto, or a zero margin for the targeted prompts on the proposed crop. The targeted prompts contrast syringes with pipettes, tubes, loose needles and other lookalikes. They summarize the best margin over two crop extents. The development sweep compares this margin on the proposed crop, the lower margin of the incumbent and proposed crops, and the earlier typed DINOv2 plus SigLIP 2 model. The sweep varies the pair and appearance cutoffs. It uses the same cached scores across comparisons. There is no additional fitting of the appearance encoders.

A release candidate must preserve counts and all existing correct reference matches at IoU 0.50 and 0.75. It must avoid edits on negative scenes, instance switches and localization decreases larger than 0.05. It must also keep the reference gains of unfiltered replacements at the same pair cutoff. Development selects eligible rules on 60 photos. A separate active comparison per method is recorded from the same grid. An independent gain and preservation on replays are required for deployment.

Eight public photos were reviewed before inference. Four contain one syringe each and four contain pipettes or sample tubes. They use distinct acquisition groups and were checked against earlier source URLs, photographers, file hashes and perceptual fingerprints. The review was performed by one reviewer. Fil Rouge remains outside fitting and calibration. The prospective experiment tests isolated objects. The 32 exposed replays include adjacent and crowded syringes. [Source manifest and reviewed boxes](../../../artifacts/detection_hillclimb/replacement_appearance_reference.json)

For an additional crop-level diagnostic, the experiment reports the RF-DETR proposal with highest reference IoU per annotated category in each image. A syringe crop needs IoU 0.50 and a lookalike crop needs IoU 0.35 against its type box. These reference-assisted examples are computed separately from the replacement eligibility and measure verification of localized crops. They provide no estimate for all detected objects.

## Development

The local encoder scored 61 distinct crops on the 60 development images. The unfiltered prior visual pair rule performs six replacements, three on negative photos. A valid syringe drops from IoU 0.828 to 0.588. The issued SigLIP 2 veto and the zero-margin targeted veto both block all three negative edits and preserve the incorrect localization change. The active targeted settings at margin 0.020 each perform one harmless edit and gain zero reference matches. The typed hybrid's active setting retains the localization regression.

No rule in the 125-setting sweep meets the full development criterion. Acquisition IDs, image IDs, file hashes and source URLs have zero observed overlap across fit, development and reserve. The fixed comparators and the active alternatives remain frozen for transfer. [Development decisions and data audit](../../../artifacts/detection_hillclimb/replacement_appearance_policy.json)

## Transfer and replay

The released system counts all eight new images correctly and matches all four syringes at IoU 0.50. None of the six tested rules changes reference matches at IoU 0.50 or 0.75, detections per image, or AP50–95, which is 0.714 throughout. Three existing boxes can move under the unfiltered, issued veto, zero-margin targeted and typed settings. The two stricter targeted settings abstain. The controlled images have zero false-positive incumbents, so they provide zero opportunities to demonstrate the prevention of a false replacement in the full pipeline.

The independent crop diagnostic has four localized syringe proposals and five lookalike proposals across the eight images. The issued SigLIP 2 cutoff keeps all four syringes and rejects all five lookalikes. The zero-margin targeted cutoff also keeps four and accepts one pipette. The targeted cutoff selected for active development rejects all four syringe examples as well as all five lookalikes. The typed model's active cutoff keeps all four and rejects all five lookalikes in this diagnostic. These counts are conditional on reference-assisted localization and include two categories in the same image.

| Rule on 32 exposed photos | Edits | Gains at IoU 0.50 | Gains at IoU 0.75 | Edits on negatives |
| --- | ---: | ---: | ---: | ---: |
| Previous visual pair rule | 6 | 3 | 1 | 2 |
| Issued SigLIP 2 veto | 6 | 3 | 1 | 2 |
| Targeted SigLIP 2 at zero | 5 | 3 | 1 | 1 |
| Targeted proposed crop | 0 | 0 | 0 | 0 |
| Targeted both crops | 0 | 0 | 0 | 0 |
| Typed hybrid on proposal | 3 | 2 | 1 | 1 |

The zero-margin targeted veto prevents the specimen-tube edit and retains the three useful replacements. It rejects the first pipette proposal, with margin −0.0024. The same scene still changes because the pair rule selects another crop of the pipette, with margin +0.0070. The stricter targeted settings discard all three useful replacements from the earlier rule. This is why a test of only the first chosen box overestimated the effect of the targeted veto.

**Status.** The study was implemented and executed locally. No variant qualified for deployment. The evaluator needed an iterator fix after feature extraction. The dependency hashes were regenerated, and the payload hashes confirm that annotations, scores and frozen choices stayed identical. The frozen development selection and full evaluation reproduced byte for byte. The test suite passed 190 tests, including five new checks covering the appearance gate and cohort scoring. [Frozen metrics and per-image edits](../../../artifacts/detection_hillclimb/replacement_appearance_results.json) · [Cache verification](../../../data/replacement-appearance/scoring-fix.json) · [Comparison figure](../../../data/replacement-appearance/comparison.html)

## Reproduction

Restore the source photos at the reference paths, the pinned local model files and the prior archives. The first three commands require unused output paths.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.experiments.relocalization.replacement_appearance_reserve reference
.venv/bin/python -m scripts.experiments.relocalization.replacement_appearance development
.venv/bin/python -m scripts.experiments.relocalization.replacement_appearance freeze
data/rfdetr/env/bin/python -m scripts.experiments.relocalization.replacement_appearance_reserve proposals
data/sam3/xpu-env/bin/python -m scripts.experiments.relocalization.replacement_appearance_reserve release
.venv/bin/python -m scripts.experiments.relocalization.replacement_appearance_reserve features
.venv/bin/python -m scripts.experiments.relocalization.replacement_appearance evaluation
.venv/bin/python -m scripts.experiments.relocalization.replacement_appearance score
.venv/bin/python -m scripts.experiments.relocalization.replacement_appearance_figure
.venv/bin/python -m pytest -q
~~~
