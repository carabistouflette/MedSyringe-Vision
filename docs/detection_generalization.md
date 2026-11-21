# Syringe detection across acquisition domains

The objective is to improve automatic detection in new photographs while preserving each previously correct localization. A localization is correct when one prediction matches one visible-instance reference at intersection over union (IoU) of at least 0.50. The automatic operating point is selected by the study and is fixed for the user. Object counts and box localization are evaluated separately. An image can have the correct count with an imprecise box.

## Evidence boundaries

| Cohort | Role in this study | Boundary |
| --- | --- | --- |
| 177 expanded Fil Rouge JPEGs | Same-session regression | All images are positive. Unknown reuse of physical specimens and one acquisition prevent an independent estimate of generalization. Original rasters are treated as read-only. |
| Zenodo simulated phlebotomy | Fit and development | 190 fit frames have 138 labeled boxes, 66 empty frames and 34 filename groups. Forty development frames have 28 boxes, 14 empty frames and three filename groups. Close views recurring across filename groups restrict the interpretation to this simulation. |
| Commons public development | Selection | Eleven scenes by 11 photographers, with nine visible syringes in six scenes and five negatives. |
| Commons public audit | Reserved | Twenty-five scenes by 25 photographers, with 15 visible syringes in 13 scenes and 12 negatives. The list and boxes were reviewed without predictions, and hashed before scoring. |
| Dental-instrument challenge | Reserved stress test | Twenty frames with an aspirating dental syringe in 12 and structurally similar instruments in eight. A single publisher setup and probably recurring tools make this a controlled challenge. |
| Previous public reserve | Historical regression | Earlier study exposure limits this cohort to regression evidence. |

The corpus builders record source URLs and groups, attribution, license, byte hashes, and perceptual hashes. Commons screening compares rotations and reflections. The Zenodo capture includes different angles at a repeating clinical simulator. Its available previously unused groups are negative and visually linked to known scenes. The primary external scope is the new public photograph cohort. Original web or foundation-model pretraining exposure is unknown. This challenge uses deliberate enrichment for difficult scenes, so precision is conditional on the mix of positives and lookalikes. [Corpus investigation](detection_corpus_research.md) · [Commons protocol](detection_public_benchmark.md) · [Zenodo fit and development manifest](../artifacts/detection_hillclimb/phlebotomy.json)

## Development experiments

The experiments use the Intel Core Ultra 7 258V laptop, its Arc 140V GPU and 30 GiB of reported host memory. Both supervised updates below ran on CPU. The XPU path was tested for open-vocabulary comparisons. No user data was uploaded. Reproduction retains exact local model files, code hashes, the evaluation cohorts and CPU/XPU records.

| Experiment | Execution and observation |
| --- | --- |
| Released hybrid | Grounding DINO Tiny supplies whole and selected regions. RF-DETR and SAM 3 can recover missed instances. A DINOv2 crop verifier audits weak uncorroborated detections. On phlebotomy development it yields 10 TP, 2 FP and 18 FN. On Commons development it yields 2 TP, 5 FP and 7 FN. Six of the seven Commons misses have a low-ranked proposal at IoU ≥0.50. |
| RF-DETR Seg Nano transfer | One epoch at 240 pixels with frozen encoder and public-only warm start. Native simulator development AP50 is 0.612, AP50–95 is 0.287 and mask AP50–95 is 0.241. Tiled inference and cross-window corroboration improve simulator candidate localization. The improvement transfers poorly to varied public photos. |
| D-FINE Nano transfer | Started from the COCO-only checkpoint at 384 pixels. The one-class decoder trained locally with a frozen backbone, and epoch 3 was resumed to epoch 6 from a saved optimizer and random generator state. Simulator development F1 rises from 0.07 to 0.66. The final development run yields 18 TP, 9 FP, 10 FN, AP50 0.509 and AP50–95 0.234. |
| Cross-source D-FINE | Used 128 simulator frames, 35 Fil Rouge fit images, and 21 earlier public fit photos. Three earlier public fit images contain syringes. Three more local epochs were executed. Commons development has 1 TP, 15 FP and 8 FN at epoch 2, the strongest checkpoint by the predeclared worst-domain ranking. This limited mix does not support a reporting or deployment improvement. |
| Open vocabulary | Grounding DINO Base and OWLv2 were run on full images and a centered square when available. Public development operating-point F1 is 0.316 for the strongest Base view and 0.400 for the strongest OWLv2 view. Values use thresholds optimized on Commons development for this diagnostic. Each arm still has many errors. An initial CPU probe was stopped after slow inference, and the public measurements were repeated on the XPU runtime. |
| Zero-shot SigLIP 2 crop semantics | The visual model compares two context sizes with independent text prompts for syringes and lookalikes. An exploratory public-only rule yields 6 TP, 0 FP and 3 FN. This rule was found by assessing 11 development scenes, and selection also requires the simulator and reserved checks. SigLIP 2 uses the checkpoint’s native `siglip` model type and fast tokenizer in the pinned Transformers environment. |

[RF-DETR training logs](../artifacts/detection_hillclimb/rfdetr_whole-frame_training.json) · [RF-DETR per-epoch metrics](../artifacts/detection_hillclimb/rfdetr_whole-frame_metrics.csv) · [D-FINE training history](../artifacts/detection_hillclimb/dfine_training.json) · [Cross-source history](../artifacts/detection_hillclimb/dfine_multidomain_training.json) · [Grounding DINO Base checkpoint](https://huggingface.co/IDEA-Research/grounding-dino-base) · [OWLv2 checkpoint](https://huggingface.co/google/owlv2-base-patch16) · [SigLIP 2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-224)

## Human verification

The application displays the source photo and highlights a selected uncertain proposal. An operator can add the original candidate box to a human review, or draw and edit a box themselves. The exported revision retains the original automatic output, the candidate index and the model’s original score and reason. Saving the same candidate twice or citing a different rectangle against the candidate index is rejected by the API. Editing a candidate’s geometry detaches the active link, and the earlier revision remains inspectable. The local Chromium smoke used a real inference on an original Fil Rouge JPEG and checked the reviewed object and exported history. [Browser evidence](../artifacts/detection_hillclimb/candidate_browser_smoke.json)

## Fil Rouge localization reference check

All 177 expanded Fil Rouge images already had the correct automatic count. Five replayed boxes failed the original 0.50 IoU check. Inspection on the oriented original rasters found displaced or overly broad draft boxes in those cases. A separate post-hoc replay with five image-based alternatives measured 177 of 177 matches. This sensitivity result uses the original predictions and changes only the copies of the reviewed boxes. The frozen dataset and previous 172/177 result remain available. The review concerns five selected frames from the same session, with known exposure to the model replay. [Five case records with both IoUs and source hashes](../artifacts/detection_hillclimb/fil_rouge_box_review.json)

## Local language-guided probes

Florence-2 base with two phrases reached 2 TP, 24 FP and 7 FN on the eleven public development photos. Center-crop inference added seven unmatched boxes and no new matches. Qwen3-VL 2B with the explicit syringe and lookalike definitions reached 2 TP, 8 FP and 7 FN. Two Qwen responses failed strict JSON validation and were recorded as abstentions. The smaller models mislocalized parts of objects and confused realistic lookalikes. Neither arm qualified as an automatic detector. Both ran offline on Intel XPU with revision-pinned safetensors. [Florence-2 model](https://huggingface.co/florence-community/Florence-2-base-ft) · [Qwen3-VL model](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## Cross-domain crop calibration

The two zero-shot semantic margins produced 640 development trials on the same fixed proposals. Applying instance-level preservation and the false-positive ceiling in both domains narrowed the best public-only variant to a stricter point. The resulting development counts are 3 TP, 1 FP and 6 FN on public photos, and 11 TP, 2 FP and 17 FN in the simulator. This reduces public false positives from five to one, recovers one instance in each domain and preserves all previously matched development instances. Independent qualification remains pending. [All trials and selected development rule](../artifacts/detection_hillclimb/crop_semantics_zero_shot.json)

## Additional real-photo data ablation

The added fit collection has 33 independent photographer groups, 18 positive scenes, 21 syringe boxes and 15 difficult negative scenes. A manual reviewer drew boxes from the photo and inspected the smallest cases at expanded resolution. Author profiles, source URLs, image bytes and transformed perceptual hashes were checked against the historical evidence and reserved cohorts. The fit set adds glass and metal barrels, individually packaged devices, marks viewed through transparent plastic, clipped tools and field procedures. The hard negatives include pipettes, pen needles and adhesive cartridges. [Fit set and provenance](../artifacts/detection_hillclimb/public_additional_fit.json)

Two D-FINE warm starts received 108 training samples, 64 positives, identical simulated frames, the same geometry and exposure augmentation, the same seed and three epochs. The enriched arm used 108 distinct rasters from 77 groups, compared with 102 distinct rasters from 55 groups in the control. Highest public AP50 rose from 0.013 to 0.030 and best development F1 rose from 0.046 to 0.092. The result shows limited useful transfer at this training scale. The enriched model still has a high false-positive burden and was excluded from the reserved automatic comparison. [Matched recipe](../artifacts/detection_hillclimb/dfine_photo_ablation_manifest.json) · [Both training histories](../artifacts/detection_hillclimb/dfine_photo_ablation_training.json)

## Auxiliary detector search

The fixed comparison contains 6,048 automatic policies and 576 bounded review-only alternatives. It compares full images, centered or tiled views, corroboration across views, signal agreement across models, candidate size limits and overlap control. Development rules preserve the exact set of already matched references and do not increase false positives on either cohort. The chosen automatic D-FINE rule only considers small whole-frame candidates with independent primary support when the baseline is empty. It recovers six missed simulation references at no added false-positive cost. Its public-development counts are unchanged. None of the additive review policies exposed another reference beyond the five existing review slots on development. [Frozen auxiliary comparison](../artifacts/detection_hillclimb/detector_selection.json) · [Model and view figure](../artifacts/figures/detection_generalization_development.png)

The sampled crop fitting set uses 54 photo groups and the primary detector’s mined hard negatives. The cross-domain search compares trained SigLIP 2, DINOv2 and joint embeddings against the original two zero-shot text margins. It records 4,645 settings, including regularization, the proposal-score floor and acceptance decisions. Since the supervised joint head improved recall and the text-margin rule rejected lookalikes, the study tested both rules together on the development data. The combination yielded 6 TP, 1 FP and 3 FN on Commons development and 12 TP, 2 FP and 16 FN on the simulation. It preserved every previously matched instance in both. The combined policy, input hashes, learned weights and competing decisions were frozen before the reserved images were scored. [Training and calibration comparison](../artifacts/detection_hillclimb/crop_semantics_development.json) · [Frozen combined policy](../artifacts/detection_hillclimb/crop_semantics_frozen.json)

## Additional Open Images reference cohort

The publisher’s V6 train box file contains 127 class records across 94 `Syringe` image IDs, and the matching publisher CSV provides all 94 attribution records. After checking source IDs, photographer metadata, source URLs, rotated and mirrored perceptual fingerprints, box tags and orientation, 69 photos and 83 boxes remained for visual review. The reviewer inspected each full frame with the publisher references, expanded eight difficult cases, rejected two ambiguous photos, corrected two duplicate-instance annotations, and reclassified seven lookalike scenes. Lookalikes include construction toys, a novelty pen, a pipette and vacuum blood-collection holders. This avoids training on publisher positives that conflict with the project’s object definition.

A separate, later study uses 57 fit photos, including six negative scenes, 61 syringes and 41 photographer groups. Ten photos, including one difficult negative and 11 reference syringes, form the reserved cohort. Acquisition groups stay together. The assignment sorts within positive/negative, partial and multi-object strata by fixed seeded hash, with explicit protection for pairs of groups in a small stratum. The new concentration of real syringe photographs extends the training inputs. The held-out cohort has too few negatives for a broad estimate of precision. [Open Images release and metadata](https://storage.googleapis.com/openimages/web/download_v7.html) · [Source review](../artifacts/detection_hillclimb/openimages_additional_reference_review.json) · [Frozen fit and audit assignments](../artifacts/detection_hillclimb/openimages_additional_manifest.json)

## Reproduction

Run the commands from the repository root after restoring the original capture, the published assets and their pinned local model folders. `docs/engineering.md` documents the application and worker environments. The standard environment is `.venv`. The RF-DETR environment is `data/rfdetr/env`. The Intel environment is `data/sam3/xpu-env`. A training or inference stage reports progress and saves its state in `data/detection_hillclimb`.

```bash
uv sync --locked --python 3.12
.venv/bin/python -m scripts.detection_public_benchmark restore
.venv/bin/python -m scripts.detection_public_benchmark verify
.venv/bin/python -m scripts.detection_hillclimb_corpus prepare
data/rfdetr/env/bin/python -m scripts.detection_hillclimb_train --epochs 1
.venv/bin/python -m scripts.detection_hillclimb_dfine train --epochs 3
.venv/bin/python -m scripts.detection_hillclimb_dfine train --epochs 6 --resume
.venv/bin/python -m scripts.detection_hillclimb_mix train --source "$FIL_ROUGE_SOURCE" --epochs 3
.venv/bin/python -m scripts.detection_real_photo_adaptation --source "$FIL_ROUGE_SOURCE" --epochs 3
```

The detector experiment records bind model proposals to image and checkpoint hashes. Collect baseline proposals on the Intel environment and RF-DETR proposals in the model’s training environment. Use the two `development` scopes before selecting either study. The SigLIP 2 and DINOv2 encoders are local safetensors published under the revisions and byte hashes in the respective lock records. Collect `public-development`, `phlebotomy-development` and `fit` crop features before running `compare`. The frozen `select` and `freeze` commands expect a new experiment directory without reserved predictions, or a copy of the delivered tree with the new execution’s output slots empty. Use the delivered JSON files to inspect the original results without retraining.

```bash
for scope in public-development phlebotomy-development; do
  data/sam3/xpu-env/bin/python -m scripts.detection_hillclimb_experiment collect \
    --scope "$scope" --backend baseline --device xpu
  for checkpoint in original-rf adapted-rf; do
    data/rfdetr/env/bin/python -m scripts.detection_hillclimb_experiment collect \
      --scope "$scope" --backend "$checkpoint"
  done
  .venv/bin/python -m scripts.detection_hillclimb_dfine collect --scope "$scope"
  .venv/bin/python -m scripts.detection_hillclimb_mix collect --scope "$scope"
  for checkpoint in dino-base owlv2; do
    data/sam3/xpu-env/bin/python -m scripts.detection_open_models \
      --scope "$scope" --backend "$checkpoint" --device xpu
  done
done
.venv/bin/python -m scripts.detection_crop_semantics collect --role public-development
.venv/bin/python -m scripts.detection_crop_semantics collect --role phlebotomy-development
.venv/bin/python -m scripts.detection_crop_semantics collect --role fit
.venv/bin/python -m scripts.detection_crop_semantics zero-shot
.venv/bin/python -m scripts.detection_crop_semantics compare
.venv/bin/python -m scripts.detection_hillclimb_rank select
.venv/bin/python -m scripts.detection_semantics_audit freeze
bash scripts/detection_hillclimb_audit.sh public-audit
bash scripts/detection_hillclimb_audit.sh previous-reserve
```

The clinical challenge uses the author-provided annotation ZIP pinned in its manifest. Restore the staged source using `scripts.detection_dental_challenge`. With the automatic policies frozen, run `scripts.detection_dental_evaluate collect --backend baseline` in the Intel environment and the selected auxiliary arms in the relevant environment. Once all three baseline caches are complete, collect the crop rule using `scripts.detection_semantics_audit collect --scope <scope>` and score it using `measure`. Figures are rebuilt using `scripts.detection_generalization_figures`. [SigLIP 2 lock](../artifacts/detection_hillclimb/siglip2_crop_backbone.json) · [DINOv2 lock](../artifacts/auxiliary_crop_backbone.json) · [Dental source](https://doi.org/10.17632/mh6zpc7y9g.1)

## Reserved auxiliary-detector result

On 25 newly reserved Commons photos, the released hybrid and the frozen additive D-FINE arm both returned 10 TP, 10 FP and 5 FN. Precision was 50.0%, recall 66.7%, F1 57.1%, AP50 59.5% and AP50–95 44.0%. Each returned the exact count on 12 of 25 photos and left 4 of 12 negative photos empty. The prior public regression was also unchanged at 4 TP, 2 FP and 3 FN. The simulation gain did not transfer to these public cohorts. Both per-instance checks passed and the additive candidate failed promotion because its reserved external results did not improve. [Reserved additive comparison and error records](../artifacts/detection_hillclimb/detector_audit.json)

The supplementary lookalike audit reserves 13 full photographs by distinct photographers. An image has an empty syringe reference only when the full scene is visually checked. The set contains two views of vacuum blood collection by different photographers, catheters, an empty infusion pump, a high-pressure instrument, clear laboratory tubes with and without colored liquid, and writing supplies. The negative-only set measures scene specificity and false alarms. [Frozen lookalike references](../artifacts/detection_hillclimb/external_lookalike_manifest.json)

Export the verified reference snapshots with `.venv/bin/python -m scripts.export_detection_references data/detection_hillclimb/coco`. The successful export preserves the number of images and syringe instances in all eight partitions. Relative `file_name` entries resolve under `data/`. Public attribution and grouped split information remain in the accompanying source manifests.

## Reserved crop-policy results

| Cohort | Released TP / FP / FN | Frozen crop TP / FP / FN | Released F1 | Frozen crop F1 |
| --- | ---: | ---: | ---: | ---: |
| Commons, 25 scenes by 25 photographers | 10 / 10 / 5 | 13 / 8 / 2 | 57.1% | 72.2% |
| Previous public regression, 21 scenes | 4 / 2 / 3 | 5 / 2 / 2 | 61.5% | 71.4% |
| Dental challenge, 20 scenes from one setup | 0 / 3 / 12 | 0 / 0 / 12 | 0.0% | 0.0% |

On reserved Commons photos, precision rose from 50.0% to 61.9%, recall from 66.7% to 86.7%, AP50 from 59.5% to 73.0% and AP50–95 from 44.0% to 53.7%. Exact counts rose from 12/25 to 19/25, and correct empty results from 4/12 to 8/12. No prior instance match was lost in any of the three cohorts. The crop candidate matched 2/3 small, 8/10 occluded or truncated, and 4/4 multi-object instances. A 2,000-replicate paired photographer bootstrap gives a 95% interval of 0 to +45.5 percentage points for the recall difference and −11.3 to +34.5 points for the precision difference. The improvement in point estimates is established on these images, and the precision interval covers zero across photographer resamples.

Clinical limitations remain visible in the audit. A vial near an occluded syringe can still be proposed independently, and the crop verifier still accepts one insulin pen, two automatic pipette scenes and a paste applicator. The metal dental aspirating syringes remain an unmet subtype. The candidate rejected all three false boxes in that setting while recovering zero of twelve target instances. Dataset diversity and absolute precision remain below the proposed acceptance target.

The crop branch’s added median runtime in this study was 5.71 seconds on the Commons cohort, 16.75 seconds on the historical regression, and 14.51 seconds on the dental images. Multiple local experiments shared the machine during these measurements. They characterize the run and should not be read as isolated interactive latency. [Full reserved crop audit](../artifacts/detection_hillclimb/crop_semantics_audit.json) · [Metric figure](../artifacts/figures/detection_generalization_audit.png)

Rebuild the visual case galleries with `.venv/bin/python -m scripts.detection_qualitative`. The generated public plates contain the original reference, the released prediction, the frozen candidate, and source attribution for ten scenes with recoveries or remaining errors. Scene selection is descriptive and occurs after calibration is closed.

## Qualification decision

The combined crop policy passed the three-cohort Commons, historical and dental comparison. Separate controls exposed limitations that prevent automatic deployment. Across all 177 Fil Rouge images it preserved the 172 matches against the original draft boxes and added eight false-positive boxes. Exact counts fell from 177/177 to 169/177. On the ten Open Images photos it preserved the eight previous matches and found two more. Its additional four false-positive boxes reduced precision from 80.0% to 62.5%, F1 from 76.2% to 74.1%, and exact counts from 8/10 to 6/10. The supplementary negative cohort gives 10/13 empty predictions and three false-alarm scenes with either policy. The released detector continues to use its existing automatic configuration. [Fil Rouge qualification](../artifacts/detection_hillclimb/crop_semantics_capture_regression.json) · [Open Images qualification](../artifacts/detection_hillclimb/openimages_crop_extension_audit.json) · [Lookalike audit](../artifacts/detection_hillclimb/external_lookalike_audit.json)

The larger training-only Open Images branch added 57 reviewed fitting rows and evaluated 768 trained-head settings on the two development cohorts. None preserved all development true instances and the false-positive limits of the earlier configuration. The independent evaluation compared only the released detector and the previous frozen crop policy. [Training-extension selection](../artifacts/detection_hillclimb/openimages_crop_extension_frozen.json)
