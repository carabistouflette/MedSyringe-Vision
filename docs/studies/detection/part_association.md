# Assigning visible parts to individual syringes

The action verifier recovered a missed dental syringe but also admitted a box mixing that syringe with adjacent instruments. Another syringe already had a usable RF-DETR proposal and remained unmatched. This study tests whether evidence of the visible parts and their physical association can rank the existing proposals better. [Previous action-verifier analysis](syringe_action_verifier.md)

## Reviewed data and compute

A single reviewer added 57 visible part centers to 20 syringe instances in seven public fitting scenes. The same annotation workflow added 29 centers to the 12 reference instances in four separate development scenes. The reviewed labels are barrel center, finger flange, plunger rod and rubber seal. The reviewer assigned each point to its owning syringe and used null for unobserved parts. The fitting stage reuses the existing annotations from 17 other public images. Development annotations are used to audit part matching and the fitting annotations supervise the model. All source rasters remain unchanged. [Part annotations and image checksums](../../../artifacts/detection_hillclimb/part_association_reference.json)

The compute budget includes all reviewed crowded scenes, all fitting images with part labels, 12 distinct fitting background acquisitions and up to 12 additional groups per polarity on development. The selection ranks the group's first eligible image by RF-DETR maximum proposal score. This targets hard lookalikes and limits repeat captures. It yields 36 fitting frames and 24 development frames. The first local benchmark ran on Intel Arc at 756 pixels. The bounded run reuses ten completed frames with the same SAM checkpoint, views, row hashes and proposal seeds. The median incremental SAM pass was 5.19 seconds on fit and 5.72 seconds on development. Fil Rouge did not contribute to annotation, fitting, calibration or augmentation.

SAM 3 provides typed part proposals on the full frame and an optional RF-DETR-seeded zoom region. A detected part gets a training owner only when its center is sufficiently close to an observed reference point with a clear margin over nearby points. The classifier learns barrel-to-flange, barrel-to-rod and barrel-to-seal association from 189 same-syringe and 160 different-syringe proposal pairs, with acquisition-balanced weights. The same-syringe training pairs cover 14 groups. The different-syringe pairs cover five groups. Repeated proposals and combinatorial pairs reduce the number of statistically independent examples. The features include longitudinal and transverse separation in the estimated barrel frame, part scales, semantic part types and detector scores. [Learned coefficients, fit composition and timing](../../../artifacts/detection_hillclimb/part_association_policy.json)

## Matched ablation

All arms rank the same RF-DETR candidates and keep every released object. The box control uses the previously fitted appearance score. The unlinked mode also checks for a barrel and secondary part in each proposal. The linked mode adds the learned pair association and optionally prevents assigned parts from being reused by nearby boxes. A conventional overlap rule applies when reliable distinct part evidence is unavailable. Missing parts stay unresolved. None of the arms moves or deletes a released box. This experiment addresses detection and counting. The estimates of capacity and liquid properties are outside its evaluation.

Development has 22 reference syringes on 24 images. A candidate must preserve existing matched instances at IoU 0.50 and 0.75, add zero per-image false positives, and avoid increasing count error per image. All 208 policies were evaluated before running prospective predictions.

| Development policy | True positives | False positives | Exact counts / 24 |
|---|---:|---:|---:|
| Released detector | 6 | 7 | 14 |
| Box control | 6 | 7 | 14 |
| Unlinked part evidence | 8 | 7 | 15 |
| Linked part evidence | 8 | 7 | 15 |

The unlinked arm won the predefined tie break by using the higher acceptance floor after identical gain, strict IoU gain and additions. The scores in different arms are on different scales. The tie break does not establish greater model confidence. For the observable development pairs, the learned classifier accepted 23 of 28 correct links and 37 of 84 incorrect links at a score of 0.50. These counts use a descriptive classifier cutoff and are separate from operating-point selection. The two positive and two negative link sets came from the same two development groups. A score from the acquisition-weighted classifier does not represent a calibrated field probability. [Development grid and frozen rules](../../../artifacts/detection_hillclimb/part_association_policy.json)

## Prospective transfer

The frozen reserve was reviewed before released or experimental inference. It contains a crowded photo from an Aviano vaccination session and three hard-negative acquisitions. In the Aviano photo eight foreground assemblies have individually traceable parts. Strong blur affects the foreground boundaries and hides any additional identities in the far background. The evaluated count is the eight visually identifiable assemblies. The negatives show a serological pipette, a handheld pipettor and tubing with detached needles. Contributor, source URL, acquisition, exact-hash and transformed perceptual screening excluded previously exposed source photos. Several otherwise useful positive images were rejected. One independent positive scene limits generalization of the transfer result. [Frozen reserve and exclusions](../../../artifacts/detection_hillclimb/part_association_reserve.json) · [Aviano photo by Ericka Woolever](https://www.dvidshub.net/image/5841408/flu-shots-aviano) · [Serological pipette by MicrobiologyMarcus](https://commons.wikimedia.org/wiki/File:Pipetter_and_serological_pipette.jpg)

| Policy | TP / 8 | FP | Precision | Recall | F1 | AP50 | AP50–95 | Exact count / 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Released detector | 0 | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2 |
| Box control | 0 | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2 |
| Unlinked part evidence | 2 | 4 | 33.3% | 25.0% | 28.6% | 10.3% | 1.7% | 0 |
| Linked part evidence | 2 | 4 | 33.3% | 25.0% | 28.6% | 10.3% | 1.7% | 0 |
| Active box probe | 2 | 4 | 33.3% | 25.0% | 28.6% | 10.3% | 1.7% | 0 |
| Active unlinked probe | 2 | 5 | 28.6% | 25.0% | 26.7% | 10.3% | 1.7% | 0 |
| Active linked probe | 5 | 10 | 33.3% | 62.5% | 43.5% | 34.7% | 10.7% | 1 |

Both part policies select the same five boxes on the reserve. Two localize syringes at IoU 0.50 and fall short of IoU 0.75. The three newly introduced false positives consist of one partial assembly on the tray and one in each of two pipette scenes. The released detector already produces a false positive on detached medical equipment. The actively linked rule counts eight boxes on the tray, of which five match at IoU 0.50. It adds six further false positives on the negative images. This shows why a correct count alone can hide incorrect instances.

The shared RF-DETR candidate pool contains eight independently matchable tray boxes at IoU 0.50 in an annotation-assisted one-to-one oracle. For each of the eight targets, individual best overlap ranges from 0.626 to 0.912. Availability of candidates therefore exceeds automatic selection. The serological pipette scores 0.894 under the frozen linked rule, above both correct accepted tray proposals at scores 0.763 and 0.744. The part prompts can interpret long graduated pipette components as compatible syringe parts, so the current association scores are insufficient for object identity. The additional SAM observations took a median 4.95 seconds per image on this four-frame run.

No policy qualifies for deployment. The result is limited to one crowded positive acquisition and three negatives. The released runtime, its thresholds and its field estimates retain their prior configuration. [Prospective metrics and accepted boxes](../../../artifacts/detection_hillclimb/part_association_results.json)

## Exposed failure replay

The previously measured dental tray, six-syringe scene and labware negative are replayed without changing the policy. The calibrated operating points add zero detections on these three frames. The active linked probe adds one correct syringe on the dental tray with zero added false positives. The active box-only probe finds the same number of new syringes and adds two false positives. Neither probe recovers the second syringe on the dental tray or the two missed syringes in the six-syringe scene. The prospective data show that the exposed gain fails to generalize. [Fixed-policy replay and part observations](../../../artifacts/detection_hillclimb/part_association_exposed_replay.json)

The serialized fit and all 208 development trials reproduced byte for byte. The local suite passed 167 tests. Four new tests check the visible landmark gates, association ranking, defensive fallback when parts lack reliable evidence, and preservation of incumbent boxes.

## Reproduction

Run from the repository root in the local environments. Restore the study's frozen public inputs and the earlier RF-DETR and release caches. Existing locks can replay scoring directly. Regenerating the locks requires fresh output paths. The source links and hashes needed to restore the reserve are in its manifest.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.part_association_reference
data/sam3/xpu-env/bin/python -m scripts.part_association_observations fit development
.venv/bin/python -m scripts.part_association freeze
.venv/bin/python -m scripts.part_association_reserve freeze
data/rfdetr/env/bin/python -m scripts.part_association_reserve collect
data/sam3/xpu-env/bin/python -m scripts.part_association_reserve baseline
data/sam3/xpu-env/bin/python -m scripts.part_association_observations fresh
.venv/bin/python -m scripts.part_association score
data/sam3/xpu-env/bin/python -m scripts.part_association_replay
.venv/bin/python -m pytest -q
~~~
