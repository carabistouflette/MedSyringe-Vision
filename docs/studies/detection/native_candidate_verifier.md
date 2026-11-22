# Native-crop verification for crowded syringe scenes

This study addresses low-confidence proposals in crowded scenes. An earlier nine-syringe photo supplied eight possible RF-DETR matches in a candidate audit, while automatic fusion matched one. The current experiment evaluates whether crop appearance and a separate overlap classifier can recover individual instances at a fixed operating point.

## Data and model

The fitting pool is the 143 previously reviewed public rasters from 116 source groups in the occlusion protocol. It includes 40 empty or lookalike scenes and seven images with several syringes. Sixty different images across 23 groups provide development data. Additional prospectively annotated public photographs have their own reserve. Fil Rouge remains outside feature fitting, calibration, data mining and augmentation. The exact separation and raster checks follow the earlier audit. [Prior leakage audit](../../../artifacts/detection_hillclimb/occlusion_leakage_audit.json)

The fixed RF-DETR Nano checkpoint runs on CPU in a full-image view and four overlapping crops. The min-score floor is 0.025 with IoU 0.62 suppression and up to 28 boxes per scene. These parameters define candidate generation. They are independent of the gate that decides whether to increment the count.

Explicit reference crops provide syringe and lookalike supervision. Mined positives require IoU of at least 0.55 and clear separation from the next matching reference. Mined negatives have IoU below 0.18 to all syringe references. Other crops are ignored for training. Per-class weights equalize total acquisition influence, so repeated crops from one event receive less individual weight. Crops come from decoded archived rasters before image resizing. The reserve includes 1,000-pixel DVIDS previews and the original Wikimedia assets. Native crop refers to the resolution of those local input rasters.

Frozen DINOv2 Small embeddings feed logistic regressions. The comparison uses a tight crop, a tight crop plus wider scene context, a fixed combination of the two-crop decision and RF-DETR confidence, and RF-DETR confidence alone. A separate classifier learns whether two overlapping positive crops correspond to distinct annotated objects. It uses intersection, center and area relationships and the similarity of the two appearance encodings. Overlapping crops with ambiguous identity are omitted. Near-identical boxes remain suppressed. [Model coefficients and policy grid](../../../artifacts/detection_hillclimb/native_candidate_verifier_policy.json) · [Representative fitting crops](../../../data/native-candidate-verifier/native-training-crop-review.png)

Only development data choose the operating points. Each policy must preserve the already matched objects, keep the false-positive count from increasing on any photo and preserve or improve the count error on each photo. The historical development snapshots came from one earlier detector. The later regression and prospective evaluations replay against the current released engine. No automatic replacement of an existing released box is performed in this experiment.

## Fitting and development results

The crop heads fit on 1,328 examples, comprising 116 reviewed syringe crops, 28 reviewed lookalikes, 206 matched detector proposals and 978 mined detector negatives. The relation head fits 374 overlapping pairs from 81 contributing source groups. The 37 distinct-instance pairs come from four groups. This concentration limits how well the head can generalize to novel styles of occlusion. Each crop is padded to 168 pixels at the encoder input, so fine details in large source crops may be reduced.

| Decision score | Development matches / 44 | False positive boxes | Exact image counts / 60 |
|---|---:|---:|---:|
| Historical detector | 15 | 7 | 37 |
| RF-DETR score and box suppression | 17 | 7 | 38 |
| Tight crop features | 15 | 7 | 37 |
| Tight crop and surrounding context | 15 | 7 | 37 |
| Crop context fused with RF-DETR score | 16 | 7 | 38 |

The per-family policies that use the learned identity check yield the same development totals as the corresponding box-suppression policies. The development selection prefers the detector score and standard box suppression. The supervised crop and pair heads remain in the comparison with operating points fixed before prospective inference. [Decision grid and data counts](../../../artifacts/detection_hillclimb/native_candidate_verifier_policy.json)

## Prospective cohort

Seven annotated public photos span four positive events with six syringes and three pipette negatives. The positives include two defocused overlapping syringes, two thin syringes installed in a pump, one syringe held by a person and one unusual multichamber syringe. The references were fixed before running either detector on these photos. The admission check screens photographer, URL, SHA-256 and image difference hashes including rotations and mirroring against existing project inputs. The displayed photographs include source credits and license identifiers in the local manifest. [Prospective references and licenses](../../../artifacts/detection_hillclimb/native_candidate_verifier_reserve.json) · [Overlapping syringes, Matthew Damon](https://www.dvidshub.net/image/6892731/operation-fortified-guardian) · [Pump scene, SabrielxD](https://commons.wikimedia.org/wiki/File:Chemyx_SKE10.jpg) · [Multichamber syringe, Philcapawiki](https://commons.wikimedia.org/wiki/File:Capa_Valve_Multi-chamber_syringe.png)

## Held evaluation

| Method | Regression matches / 91 | Regression false positives | Prospective matches / 6 | Prospective false positives | Prospective exact counts / 7 |
|---|---:|---:|---:|---:|---:|
| Released system | 34 | 1 | 3 | 1 | 4 |
| RF-DETR-only control | 34 | 1 | 3 | 1 | 4 |
| Tight appearance and box suppression | 39 | 2 | 3 | 1 | 4 |
| Tight appearance and learned identity | 39 | 4 | 3 | 3 | 3 |
| Appearance and context with box suppression | 38 | 7 | 3 | 1 | 4 |
| Appearance and context with learned identity | 38 | 11 | 3 | 2 | 3 |
| RF-DETR and context with box suppression | 37 | 2 | 3 | 2 | 3 |
| RF-DETR and context with learned identity | 37 | 2 | 3 | 2 | 3 |

No fixed arm recovered a new matched instance on the seven prospective photos. The released and development-selected systems both recorded 0.75 precision, 0.50 recall, 0.60 F1 and four exact counts. The shared RF-DETR bank could match four of the six references with an oracle, including both on the defocused pair. Neither thin pump syringe had a sufficient candidate box. The extra prospective boxes all appear on a single-syringe scene. All three pipette controls preserve the released result. The automatic release condition remains unmet. [Prospective scores and boxes](../../../artifacts/detection_hillclimb/native_candidate_verifier_prospective.json)

On the earlier nine-syringe scene, the RF-DETR and context combination adds one correct box with ordinary overlap suppression and two with learned identity. The latter moves the inferred count from one to three while the number of correctly localized instances moves from zero to two. It introduces no additional false positives on that photo. On the 27-photo regression cohort, that same variant adds three correct boxes and one false positive. The false count on a new single-syringe scene illustrates why appearance classification and distinct-instance verification need separate supervision. [Retrospective scores](../../../artifacts/detection_hillclimb/native_candidate_verifier_results.json) · [Crowd recovery and false extra box](../../../data/native-candidate-verifier/native-crop-verification-examples.png) · [Older crowd photograph](https://www.dvidshub.net/image/8711851/mission-ready-56th-fw-rapid-flu-shot-pod) · [Single-syringe photograph](https://www.dvidshub.net/image/9190822/uss-gerald-r-ford-cvn-78-conducts-deployment-shot-exercise)

In the single-syringe error, the accepted crop covers the needle and hub of the existing assembly and part of the vial. It has IoU 0.165 with the released box and intersection over the smaller area of 0.491. The pair check only scores ambiguous overlaps above the standard duplicate criteria. The fragment therefore bypasses the relation head. The follow-up tests explicit fragment examples and adjacent proposals with low bounding-box overlap. [Assembly study](syringe_assembly_and_thin_views.md)

Four new automated tests check ambiguous proposal labels, acquisition balancing, duplicate protection, and preserved per-photo count quality. The full suite passed 136 tests with the existing Starlette deprecation warning. A second prospective replay reproduced the scores and decisions byte for byte.

## Reproduction

Use the existing RF-DETR and Intel Arc environments and the project Python environment. Restore the original bytes from each source URL in the prospective manifest to its listed location. The software verifies those bytes and the local model hashes before inference.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
data/rfdetr/env/bin/python -m scripts.native_candidate_bank fit
data/rfdetr/env/bin/python -m scripts.native_candidate_bank development
data/rfdetr/env/bin/python -m scripts.native_candidate_bank evaluation
.venv/bin/python -m scripts.native_candidate_verifier learn
.venv/bin/python -m scripts.native_candidate_verifier history
.venv/bin/python -m scripts.native_candidate_reserve freeze
data/sam3/xpu-env/bin/python -m scripts.native_candidate_reserve baseline
data/rfdetr/env/bin/python -m scripts.native_candidate_reserve proposals
.venv/bin/python -m scripts.native_candidate_reserve measure
.venv/bin/python -m pytest -q
~~~
