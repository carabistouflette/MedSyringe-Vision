# Ranking syringe proposals and resolving overlapping instances

On the earlier Aviano photo, the part-association bank contained eight usable proposals. Six references remained reachable after the selected acceptance floor and four after the existing conflict constraints. The actual selector matched two. This follow-up compares proposal ranking and pairwise conflict decisions under the same detector and proposal budget. [Previous part-association study](part_association.md)

## Acquisitions and review

The study reuses the 143 fitting photos from 116 acquisition groups and all 60 public development photos from 23 disjoint groups. Fil Rouge does not supply labels, crops, augmentations or calibration outcomes. Each image uses the first RF-DETR pass and up to six stored recrops. Eligible boxes score at least 0.075 in RF-DETR, pass a fixed high-overlap suppression of 0.88, and belong to a shared pool of at most 36 candidates. The same pool serves all arms.

A single visual reviewer checked 27 actual proposal pairs from public fitting scenes. The final labels distinguish nine same-instance pairs, five different-instance pairs and 13 ambiguous pairs. The ambiguous review records exclude those pairs from identity supervision. Several automatically proposed distinct pairs covered multiple nearby syringe bodies, which made a stable ownership assignment impossible. A separate visual audit of the 12 highest-scoring known-negative fitting proposals found tube holders, pipettes, detached components and dental tools. [Frozen pair review and box coordinates](../../../artifacts/detection_hillclimb/overlap_ranking_pair_review.json) · [Visual review contact sheet](../../../data/overlap-ranking/fit-pairs-0.jpg)

The pointwise classifier uses 260 unambiguous complete proposals and 1,976 fragments or backgrounds. A further 215 geometry-ambiguous proposals are excluded from the unary fit. The ranking head preserves the same absolute labels and adds 1,906 within-image comparisons. Of those, 1,772 contrast usable proposals against fragments or backgrounds, and 134 prefer a more complete localization of the same instance. Inputs are shared RF confidence, cross-view agreement, visual axis profiles and fixed DINOv2-Small embeddings from tight and contextual crops. DINOv2 encoding and classifier fitting run locally on CPU.

The pair classifier uses the same visual crops, two embedding similarities and a symmetric description of axis agreement, scale and overlap. It combines nine reviewed and 1,597 geometry-labeled same-instance pairs with five reviewed and 21 geometry-labeled different-instance pairs. The reviewed pairs receive fourfold sample weight before logistic fitting. Different-instance examples come from five groups. This is too narrow a positive-pair base to support broadly reliable overlap exceptions without independent evidence. The 27 visual decisions require additional review before they can support a consensus reference set. [Fit counts and serialized classifiers](../../../artifacts/detection_hillclimb/overlap_ranking_policy.json)

## Controlled comparison

The four arms cross pointwise or same-scene contrastive scoring with conventional or learned overlap conflicts. A learned conflict may permit a well-supported distinct neighbor and block a likely repeated instance at lower overlap. Near-identical boxes remain in conflict. Missing pair evidence falls back to conventional overlap. Greedy and bounded exact joint selection use the same 16-slot cap. Every arm keeps each released box with its existing score. The operating points are internal to the experiment.

The 192 development policies were all evaluated against the same released predictions. The gate preserves reference identities at IoU 0.50 and 0.75, prevents any per-photo increase in false positives and prevents any per-photo increase in absolute count error.

| Development policy | Matched / 44 | False positives | Exact count / 60 |
|---|---:|---:|---:|
| Released detector | 15 | 7 | 37 |
| Pointwise and conventional conflict | 19 | 7 | 39 |
| Contrastive and conventional conflict | 17 | 7 | 38 |
| Pointwise and learned conflict | 19 | 7 | 39 |
| Contrastive and learned conflict | 17 | 7 | 38 |

The policy selects pointwise scoring with conventional conflicts and greedy selection for prospective qualification. This arm ties its learned-conflict counterpart on development. The fixed iteration order retains the simpler conflict model. The separately frozen active probes maximize development F1 and can fail the release guard. [Development selection and full grid](../../../artifacts/detection_hillclimb/overlap_ranking_policy.json)

## Prospective and exposed evaluation

Four independent acquisition groups were visually labeled before inference. Two military-source photos show three syringes. One image contains two small syringes held at different depths. The other shows a single partly obscured syringe. Two labware negatives show realistic mechanical and serological pipettes. Perceptual, exact-hash, author and URL checks screened the repository's earlier sources and the private inventory. Candidate photos with unclear crowded extents were excluded. The dense Aviano tray, earlier six-syringe photo and previously evaluated dental instrument scene remain exposed replays. [Independent acquisition manifest](../../../artifacts/detection_hillclimb/overlap_ranking_reserve.json) · [Kuwait source by Noel Gerig](https://www.dvidshub.net/news/388657/covid-19-vaccine-rollout-ongoing-us-army-central-footprint) · [Minot source by Jessica Montano](https://www.dvidshub.net/image/2940421/vaccination-exercise-trains-medics) · [Pipetting photo by Diane A. Reid](https://commons.wikimedia.org/wiki/File:Pipetting.jpg)

| Frozen policy | TP / 3 | FP | Precision | Recall | F1 | AP50 | AP50–95 | Exact count / 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Released detector | 2 | 0 | 100.0% | 66.7% | 80.0% | 66.3% | 36.8% | 3 |
| Pointwise, conventional conflicts | 3 | 2 | 60.0% | 100.0% | 75.0% | 86.5% | 40.9% | 2 |
| Contrastive, conventional conflicts | 3 | 2 | 60.0% | 100.0% | 75.0% | 86.5% | 40.9% | 2 |
| Pointwise, learned conflicts | 3 | 2 | 60.0% | 100.0% | 75.0% | 86.5% | 40.9% | 2 |
| Contrastive, learned conflicts | 3 | 1 | 75.0% | 100.0% | 85.7% | 91.6% | 41.9% | 3 |

All four frozen arms recover the obscured rear syringe in the two-technician photograph. Both contrastive arms use the same scoring floor and greedy selection. Changing only the conflict rule blocks a repeat localization of the foreground syringe and preserves the valid new detection. The rejected box overlaps the incumbent at IoU 0.307. Conventional overlap leaves it eligible. The learned distinct-instance score is 0.0028, which triggers a duplicate veto. The paired ablation isolates one useful effect of the new conflict decision.

The remaining false positive is a mechanical pipettor in the NCI photograph. In the contrastive model it ranks slightly above the correctly added syringe. A single score floor cannot reject that pipettor while retaining the new syringe in this observed pair of cases. The best transfer F1 is 85.7%, calculated from three correct detections and one false positive. The precision falls from 100.0% to 75.0% and the exact count remains three of four. This small cohort covers two positive acquisitions and two negatives. It cannot establish performance on crowded scenes as a general population.

The pointwise rule selected on development adds six false positives and recovers zero confirmed instances in the exposed replay. The exploratory contrastive probe with learned conflicts recovers five matches across three already evaluated scenes. It counts all six correctly in the prior close-up with green plungers, then incurs nine added false positives across the full seven-image exposed cohort. This does not qualify as an independent or safe deployment result. None of the frozen arms passes the release gate across the two cohorts. The existing detector remains the application default. [Prospective and replay metrics, candidate boxes and proposal ceilings](../../../artifacts/detection_hillclimb/overlap_ranking_results.json)

The three fitted heads and all 192 development decisions reproduced byte for byte on the same features. The two transfer evaluations also reproduced. The full test suite passed 172 tests. The five new checks cover fragment identities, ambiguous matching, conditional duplicate suppression, preservation of incumbents and exact joint selection under a broad proposal.

## Fixed appearance veto diagnostic

A post-evaluation run applied the pinned release SigLIP 2 veto only to the new boxes. It retained both the recovered syringe and the NCI pipettor accepted by the frozen contrastive policy. On the seven exposed acquisitions, the active contrastive probe moved from nine matched instances and ten false positives to eight matched instances and four false positives. The lost match is the dental cartridge syringe. The original veto rule and incumbent boxes were held fixed. These cases motivate targeted object classification on the new proposal stream. [Fixed-rule decisions and margins](../../../data/overlap-ranking/appearance-gate-diagnostic.json)

## Reproduction

Run from the repository root. Restore the precise public image bytes, existing RF-DETR archives, and the locally pinned DINOv2 weights. Freeze commands expect fresh output paths. Existing locked results can be rescored directly. The published inputs verify the crop review, classifier parameters and candidate placement against the source raster.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.overlap_ranking_review queue
.venv/bin/python -m scripts.overlap_ranking_review freeze
.venv/bin/python -m scripts.overlap_candidate_features fit development
.venv/bin/python -m scripts.overlap_ranking freeze
.venv/bin/python -m scripts.overlap_ranking_reserve freeze
data/rfdetr/env/bin/python -m scripts.overlap_ranking_reserve collect
data/sam3/xpu-env/bin/python -m scripts.overlap_ranking_reserve baseline
.venv/bin/python -m scripts.overlap_candidate_features replay fresh
.venv/bin/python -m scripts.overlap_ranking score
.venv/bin/python -m pytest -q
~~~
