# Small and occluded syringes, and seal-front localization

The study compares three local interventions for errors observed on real public photographs. It separates the availability of usable proposals, the correctness of automatic count recovery, and the ability to place the liquid-facing edge of the seal. Fil Rouge remains a reference for the application and was excluded from fitting, calibration, and augmentation in this study.

## Experimental split

The RF-DETR adaptation uses 143 reviewed source rasters from 116 fitting groups. It adds 38 deterministic close views, for 181 total training rasters, 158 visible syringe boxes and 40 empty scenes. A crop is accepted only when every intersected syringe has its full reviewed box inside the retained field of view. The separate development cohort has 60 rasters, 23 groups, 44 syringe boxes and 24 empty scenes. One candidate was excluded because it shared a contributor and source group with a transfer image. Every COCO file and materialized training raster has a recorded SHA-256. The initial detector was the existing public-data RF-DETR Nano continuation. The comparison uses the same model size and spatial resolution for both weights. It tests one CPU epoch with the encoder frozen and four inference arms. This test changes the fitting distribution and performs additional updates, so it cannot attribute any difference solely to crop augmentation. [Input design](../../../artifacts/detection_hillclimb/occlusion_detector_plan.json) · [Training record](../../../artifacts/detection_hillclimb/occlusion_detector_training.json)

The held evaluation includes the previously exposed regression, known-contributor transfer, hard negatives and Oriez occlusion examples. Two additional DVIDS acquisitions were screened against indexed author, URL, byte and visual hashes and annotated before inference. Three syringes were reviewed across these two photographs. A third candidate shows a densely layered tray with unclear instance boundaries and was excluded from exact-count scoring. Evaluation photos do not set score floors. The detection counts use score-ordered, one-to-one matches at box IoU 0.50. A new stage must preserve all already matched instances, add zero false positives in each cohort and improve the new reserve. [Reserve references](../../../artifacts/detection_hillclimb/occlusion_detector_reserve.json) · [Goonan, Patrick SFB](https://www.dvidshub.net/image/8071796/920th-aeromedical-staging-squadron-provides-vaccine-during-drill-weekends) · [Breving, Smyrna](https://www.dvidshub.net/image/9584272/tennessee-national-guard-medics-conduct-trauma-validation-during-68w-recertification)

A separate replay recomputed rotated and mirrored difference hashes for 143 fitting, 60 development and 35 public evaluation rasters. The checks found no fit-to-development, fit-to-evaluation, or private-image collisions at Hamming radius four. This screens the local inputs. Training exposure in upstream foundation weights remains unknown. [Leakage audit](../../../artifacts/detection_hillclimb/occlusion_leakage_audit.json)

## Proposal-stage audit

On the Oriez example, there are two reviewed small syringes. The released system has one box and matches neither. SAM 3 in a single 756-pixel scene view supplies one match among six raw candidates at the fixed 0.18 floor. It still supplies the one match after raising the floor to 0.40 and suppressing overlapping masks, and the previous part and appearance verifiers select no new boxes. Four tiles at 66% of the source dimensions supply two distinct matches among six edge-filtered boxes. One match is fragile at IoU 0.5004. Four tiles at 50% supply one match. Privileged crops centered on each reviewed object recover only one of the two at strict IoU. The oracle maximizes distinct target matches before summed IoU. It measures candidate availability and uses reference boxes. [Stage metrics and target overlaps](../../../artifacts/detection_hillclimb/occlusion_proposal_audit.json) · [Visual comparison](../../../data/occlusion-study/oriez-proposal-comparison.png) · [Oriez photo](https://www.dvidshub.net/image/6479379/wright-patt-phase-1-covid-vaccinations)

The left syringe has a useful full-image proposal at confidence 0.419 and only weak raw tile support at 0.264. The best right-hand tile box reaches confidence 0.877 and still includes part of the vial, accounting for its borderline overlap. These first-stage observations separate confidence, geometric localization, and duplicate handling.

## RF-DETR train and inference comparison

| Weights and view | Development precision | Development recall | Development F1 | Development AP50 | Development AP50–95 | Median CPU on public evaluation |
|---|---:|---:|---:|---:|---:|---:|
| Initial, full | .327 | .386 | .354 | .261 | .121 | 254 ms |
| Initial, full and tiles | .462 | .409 | .434 | .371 | .166 | 1,274 ms |
| Adapted, full | .290 | .409 | .340 | .213 | .070 | 256 ms |
| Adapted, full and tiles | .441 | .341 | .385 | .300 | .091 | 1,251 ms |

The one-epoch run took 275 seconds. Confidence floors maximize F1 on development and are frozen for each arm. Both tiled arms localize two of three reserve instances when used alone, with one false positive and zero exact counts across the two new photos. The release localizes one of three, with one false positive and one exact count. The four additive variants all preserve the released reserve counts and localizations. The adapted full-frame arm raises regression recall from 34/91 to 48/91 and false positives from one to 17. Its exact counts decrease from 12/27 to 8/27. The initialization and adapted tiled arms each add two regression matches and five false positives. No RF-DETR arm reaches the release condition. [Frozen selection](../../../artifacts/detection_hillclimb/occlusion_detector_selection.json) · [Full results and errors](../../../artifacts/detection_hillclimb/occlusion_detector_results.json)

On the new Breving photograph, the released box matches the visible syringe at IoU 0.179. The strongest adapted tiled box reaches IoU 0.821, and the fixed overlap rule rejects it as a duplicate of the released box. The initial full-frame RF-DETR recovers one Oriez instance at IoU 0.501 and the same rule blocks it against a released box at IoU 0.420. A separately tested relocalization policy would need to distinguish a tighter box for one instance from a second object. This audit used the reference boxes after inference to identify the blocker. [Fusion diagnostic](../../../artifacts/detection_hillclimb/occlusion_fusion_diagnostic.json)

## SAM 3 recovery policies

| Automatic recovery | Regression matches / 91 | False positives | Exact counts / 27 |
|---|---:|---:|---:|
| Released system | 34 | 1 | 12 |
| SAM full-frame additions | 70 | 13 | 13 |
| SAM 66% tile additions | 50 | 10 | 14 |
| Tile agreement across views | 48 | 5 | 15 |
| Tiles with the frozen part verifier | 38 | 2 | 12 |

Routing applies to scenes with at most one released detection. The new Goonan scene has two oracle matches after feature extraction in both full-frame and tiled SAM streams. That stage includes a minimum confidence of 0.40. The two boxes are not added by the frozen count recovery rules. On the Breving scene, neither SAM view reaches a match. On Oriez only one match survives that stage for either view. Every SAM policy leaves the automatic result at one matched syringe out of three on the new reserve. All methods preserve the released matched identities. [Full SAM results](../../../artifacts/detection_hillclimb/occlusion_sam_tiling_results.json)

## Independent-model corroboration

The frozen hybrid accepts a SAM tile candidate only when a tiled RF-DETR proposal also covers it at IoU 0.50 or greater. The two RF-DETR sources use half their standalone development floor for corroboration. The candidate has to pass the SAM floor and the standard duplicate rule. With initial RF-DETR weights, this moves regression results from 34 to 41 matches and from one to four false positives. With adapted weights, it moves the result from 34 to 42 matches, keeps the one existing false positive, and improves exact counts from 12/27 to 13/27. The adapted hybrid preserves all released matches. Both hybrids leave the other cohorts unchanged, including one match out of three on the two new acquisitions. These results justify a prospective test of the corroboration rule and fall short of the automatic release gate. [Frozen hybrid policy](../../../artifacts/detection_hillclimb/occlusion_hybrid_policy.json) · [Hybrid replay](../../../artifacts/detection_hillclimb/occlusion_hybrid_results.json)

## Box refinement with count held fixed

A separate follow-up tests whether a tighter RF-DETR box can improve localization without changing the instance or the reported count. The rule considers scenes with one released box. The replacement must be smaller than and substantially contained in that box. A credible spatially distinct proposal under the same released box vetoes refinement. The separate addition rule checks occupancy against both the original and refined boxes so an overlapping region is not accidentally counted a second time. The experiments update evaluation geometry only. A future application integration would have to recompute markings, liquid evidence and all other crop-derived fields from the adopted crop.

Of 72 configurations, 27 made a change on the 60 archived development images without losing a match at either 0.50 or 0.75 IoU. All archived development baselines came from the same historical detector. Selection used the number of gained matches at each IoU, then the smallest number of changes. The frozen rule makes three changes on development and gains one IoU 0.50 match and two IoU 0.75 matches. It uses the original RF-DETR checkpoint in full-and-tiled mode, the development confidence floor, area ratio at most 0.80, and containment at least 0.75. These development results follow a parameter search and do not estimate generalization. [Configuration grid and frozen rule](../../../artifacts/detection_hillclimb/occlusion_relocalization_policy.json)

On the retrospective 35-photo replay, refinement makes one change. It converts the unmatched Breving box to a true match at IoU 0.50, preserving the count and all previously matched targets at 0.50 and 0.75. The earlier two-photo reserve moves from one to two matched syringes of three and from one to zero false-positive boxes. This photograph motivated the follow-up and is diagnostic evidence. On the 27-photo regression set, appending and combining RF-DETR boxes adds two true and five false detections. Those arms reduce the number of exact counts from 12 to eight. [Retrospective results](../../../artifacts/detection_hillclimb/occlusion_relocalization_results.json)

Five additional DVIDS photographs were selected, annotated and hash-bound after freezing the rule and before running either engine. They cover five distinct events, three positive scenes with 11 visible assemblies and two laboratory pipette negatives. One dense tray was excluded because exact instance boundaries were ambiguous. Some references in the nine-syringe scene are blurred or clipped. The prospective gate requires a new match, no lost matches at 0.50 or 0.75, no extra false-positive boxes on any photo and no worsened count. [Prospective annotations and original URLs](../../../artifacts/detection_hillclimb/occlusion_relocalization_reserve.json)

| Method | Prospective matches / 11 | False positives | Exact counts / 5 |
|---|---:|---:|---:|
| Released system | 1 | 2 | 4 |
| Refine one existing box | 1 | 2 | 4 |
| Add RF-DETR boxes | 2 | 4 | 3 |
| Refine and add | 2 | 4 | 3 |

Refinement fixes one localization and loses one on a different image. Both changes preserve the counts. Both addition arms introduce two false positives on a pipette photo. No arm passes the prospective gate. The evaluation is small, with nine of 11 targets in one scene. [Prospective replay](../../../artifacts/detection_hillclimb/occlusion_relocalization_prospective.json)

The nine-syringe photo exposes the count bottleneck. The released engine counts one, and the addition arm counts two with one correct box. The original tiled RF-DETR stream provides eight distinct oracle matches among post-NMS candidates before the frozen floor. Three survive the floor in the oracle. This identifies low-score candidate verification and crowded instance separation as the highest priority. The eight potential matches use annotations and therefore do not measure automatic detection. [Proposal-stage audit](../../../artifacts/detection_hillclimb/occlusion_relocalization_proposal_ceiling.json) · [Guachun-Chichay photo](https://www.dvidshub.net/image/8711851/mission-ready-56th-fw-rapid-flu-shot-pod)

## Seal-front and graduated volume

The manual points for this study mark the edge facing the liquid. The previous part experiment marked seal centers. The landmark model sees aligned cross-sections along a manually reviewed axis and width. It compares a signed transverse-contrast baseline with a small logistic profile classifier trained on six public acquisitions and a variant mixing those same groups with 36 deterministic rendered images. Acquisition leave-one-out folds set conservative score floors. The model also checks whether a competing profile peak is almost as strong. The tolerance is expressed in source pixels with an allowance related to width and visible axial length. Small or visibly occluded axes can cause abstention before the seal scorer runs. [Landmark and scale references](../../../artifacts/detection_hillclimb/occlusion_seal_references.json) · [Frozen models and grouped fits](../../../artifacts/detection_hillclimb/occlusion_seal_models.json)

The unaided highest-ranking candidate is inside tolerance in 3 of 6 grouped development crops for signed contrast, 2 of 6 for real-only learning, and 3 of 6 for real plus render. In the two newly reviewed crops with visible seal fronts, signed contrast emits two positions with one inside tolerance. Real-only learning emits one position outside tolerance. The real-plus-render variant abstains on both. All three abstain on the new obscured-axis case and on numeric volume for these new images. The known-contributor clinic case is included separately. Contrast locates its rim and both learned heads abstain. [Held landmark predictions and field-specific abstention](../../../artifacts/detection_hillclimb/occlusion_seal_results.json)

A separate public 2 mL image has a reviewed readable scale and a visual rim interval of 1.139 to 1.257 mL. Its source caption gives an approximate fill description. Signed contrast emits 1.265 to 1.383 mL, with a 24.1 px rim error against the reviewed point. Both learned heads abstain. Scale projection has a 0.60 px residual on this case. The seal localization error dominates. These are visual readings from a reviewed axis and graduations. The dataset has no independently dispensed measurement on this image. The result provides no calibrated interval coverage claim. [Volume probe](../../../artifacts/detection_hillclimb/occlusion_volume_probe_results.json) · [Fifo image and attribution](https://commons.wikimedia.org/wiki/File:Syringe_details_raw.jpeg)

## Reproduction

Run data preparation and freeze steps once in a workspace with the existing public assets and model weights. Keep the study's source bytes and manifests alongside the results. Use local workers with the documented locked environments.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p data/occlusion-study
curl -fL https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2310/8071796/1000w_q95.jpg -o data/occlusion-study/8071796.jpg
curl -fL https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/2603/9584272/1000w_q95.jpg -o data/occlusion-study/9584272.jpg
curl -fL https://upload.wikimedia.org/wikipedia/commons/1/16/Syringe_details_raw.jpeg -o data/occlusion-study/fifo-2ml.jpg
.venv/bin/python -m scripts.occlusion_detector prepare
.venv/bin/python -m scripts.occlusion_reserve freeze
.venv/bin/python -m scripts.occlusion_leakage_audit
.venv/bin/python -m scripts.occlusion_hybrid select
data/sam3/xpu-env/bin/python -m scripts.occlusion_proposals
data/sam3/xpu-env/bin/python -m scripts.occlusion_reserve released
data/rfdetr/env/bin/python -m scripts.occlusion_detector train
data/rfdetr/env/bin/python -m scripts.occlusion_detector select
data/rfdetr/env/bin/python -m scripts.occlusion_detector score
.venv/bin/python -m scripts.occlusion_fusion_audit
data/sam3/xpu-env/bin/python -m scripts.occlusion_sam_tiling
.venv/bin/python -m scripts.occlusion_hybrid score
.venv/bin/python -m scripts.occlusion_seal reference
.venv/bin/python -m scripts.occlusion_seal learn
.venv/bin/python -m scripts.occlusion_seal measure
.venv/bin/python -m scripts.occlusion_volume_probe
.venv/bin/python -m pytest -q
~~~

The release replay and SAM use Intel Arc. The one-epoch RF-DETR adaptation and scoring run on CPU. An offline metric replay from archived proposals reproduced all 45 RF-DETR metric bundles and verified both checkpoint and training-curve hashes. The five added automated tests cover unique instance matching, complete crop annotations, duplicate rejection, cross-window corroboration, and explicit volume abstention. The 129-test suite passed with one pre-existing Starlette deprecation notice.

The refinement follow-up uses the same installed environments. The raw source URLs and exact SHA-256 digests are in the prospective annotation manifest. Store those JPEGs in `data/occlusion-relocalization` as listed in the manifest.

~~~bash
.venv/bin/python -m scripts.occlusion_relocalization select
.venv/bin/python -m scripts.occlusion_relocalization measure
.venv/bin/python -m scripts.occlusion_relocalization_reserve freeze
data/sam3/xpu-env/bin/python -m scripts.occlusion_relocalization_reserve baseline
data/rfdetr/env/bin/python -m scripts.occlusion_relocalization_reserve collect
.venv/bin/python -m scripts.occlusion_relocalization_reserve measure
.venv/bin/python -m scripts.occlusion_relocalization_diagnostic
.venv/bin/python -m pytest -q
~~~

The frozen selection and retrospective metrics reproduced byte for byte. The full suite passed 132 tests, including new checks for instance identity preservation, the ambiguous-neighbor veto and duplicate prevention after refinement.

The next study evaluates a supervised verifier built on native-input crops and a separate overlapping-instance classifier. [Native-crop verification](native_candidate_verifier.md)
