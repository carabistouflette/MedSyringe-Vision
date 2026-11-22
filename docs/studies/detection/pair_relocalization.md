# Pair-supervised, count-preserving syringe relocalization

The box quality study found usable alternatives for four poorly localized syringes that were already counted. Its follow-up showed that the best visual score could prefer a fragment among boxes associated with the incumbent. This study predicts the outcome of a directed replacement pair and compares it with the earlier count-preserving selector. [Box quality and identity-link diagnostics](syringe_box_quality.md#replacement-diagnostic)

## Fitting and protocol

Training uses real RF-DETR alternatives from the 143-image public fitting cohort. A directed example has a proposed incumbent and an alternative from the same image. A box needs adequate overlap with one reviewed instance and a lead over competing references to be assigned to that instance. Background and different-instance examples provide negative identity supervision. Ambiguous boxes are omitted. A useful replacement keeps the same reference, reaches IoU 0.50, improves IoU by at least 0.10, and retains the reviewed visible points when available. Poor final overlap, deterioration or loss of a visible point provide negative improvement supervision. Near-ties in measured improvement are excluded from that head.

The geometry score uses relative translation, scale, box overlap, cross-view evidence and local crop descriptors. The visual pair score also uses the dot products, distances and differences of two frozen DINOv2 embeddings. The fitted methods compare geometry-only joint prediction, visual joint prediction, and separate models for same-instance identity and improvement. The ownership classifier balances same-instance, different-instance and background examples across acquisition groups. Controls include the frozen historical relocalizer and the difference between the prior visual IoU estimates of the incumbent and proposed box.

Each proposal can replace one slot. Competing slot claims within 0.10 of one another trigger abstention for the joint and factorized methods. The rule preserves the old ID and detection score and checks for overlap with other output slots before applying the replacement. The same-instance and improvement cutoffs are chosen automatically on the 60 development photos.

Eligibility requires unchanged counts, no lost reference at IoU 0.50 or 0.75, and no pair that switches to another clearly assigned syringe. A reduction in reference IoU greater than 0.05 or any edit on a reviewed negative image also fails the gate. Deployment qualification requires a gain on the new public reserve and preservation on all replay cohorts.

The independent reserve was annotated before detector inference. Six photos come from different contributors. They show seven reference syringes across three positive scenes, with five adjacent syringes in one scene. The negative scenes contain specimen tubes or a dental handpiece. Public source provenance, file hashes and exclusion reasons are preserved. Candidate photos were screened against earlier public acquisitions and Fil Rouge fingerprints. Crowded and overlapping scenes from earlier experiments are replays. [Reserve and screening records](../../../artifacts/detection_hillclimb/pair_relocalization_reference.json)

## Development selection

The identity head uses 15,234 directed pairs from 115 acquisition groups. These include 3,254 same-instance pairs, 100 different-syringe pairs and 11,880 pairs with background. The improvement model uses 3,034 same-instance pairs across 72 groups, with 915 useful replacements. The proximity of references and the pair count both matter for interpretation. Only 100 pairs directly test confusion between two different syringes.

The split audit finds zero shared image IDs, acquisition groups, exact file hashes or source URLs across fit, development and reserve. Fil Rouge stays outside training, selection and augmentation. The historical rule gains one match at both IoU cutoffs on development. The four challengers have no nontrivial eligible rule. The development grid is retained, including rejected settings. [Fitted model, split audit and development trials](../../../artifacts/detection_hillclimb/pair_relocalization_policy.json)

Each fitted method that made any development edits also has an exploratory comparison. These policies were fixed from the development grid before reserve scoring. Their selection maximizes net matches at IoU 0.50 and 0.75 and uses gains, harmful edits and number of edits as tie breakers. All three fail the development preservation gate. The visual IoU difference rule makes zero development edits throughout its grid. [Active comparison policy](../../../artifacts/detection_hillclimb/pair_relocalization_probe_policy.json)

## Independent transfer and replay

The released detector matches six of seven reserve syringes at IoU 0.50 with zero false positives and exact counts on five of six images. The missed syringe has no incumbent slot. The historical rule and all challengers that satisfy the development gate make zero reserve edits. The three active probes preserve counts across the reserve and 30 replay photos.

| Active comparison on reserve | Box edits | New matches at IoU 0.50 | New matches at IoU 0.75 | Degraded boxes |
| --- | ---: | ---: | ---: | ---: |
| Geometry | 6 | 0 | 1 | 3 |
| Joint visual | 0 | 0 | 0 | 0 |
| Separate identity and improvement | 0 | 0 | 0 | 0 |

Geometry raises AP50–95 from 0.598 to 0.681 on the reserve. It also lowers the reference IoU of three already well-localized boxes by more than 0.05 in the adjacent-syringe scene. In the earlier crowded replay, it gains two matches at IoU 0.50 and loses two different matches at that cutoff. One incumbent switches to a neighboring syringe. These per-instance changes show why the acceptance gate checks identities and localization as well as aggregate scores.

On the earlier box quality cohort, the joint visual probe restores three of the four previously missed localization matches at IoU 0.50. It takes the cohort from three to six true positives and preserves the count in every image. It also moves a false detection on a specimen-tube photo. The factorized probe restores one of those four missed matches with zero harmful edits on that cohort. Its selected active cutoff still fails the development gate because it modifies a reviewed negative.

**Status.** The count-preserving pipeline and all comparisons were executed locally. Deployment qualification was false. Independent coverage includes one adjacent-syringe scene with a single reviewer. Training and evaluation reproduced byte for byte. All 185 tests passed, including new tests for directed supervision, slot identity, ambiguity and localization preservation. [Frozen principal results](../../../artifacts/detection_hillclimb/pair_relocalization_results.json) · [Exploratory per-image results](../../../artifacts/detection_hillclimb/pair_relocalization_probe_results.json) · [Comparison figure](../../../data/pair-relocalization/comparison.html)

## Appearance diagnostic

A local audit reran frozen appearance models on the three successful visual replacements and the two edited lookalikes from the earlier replay cohorts. The released SigLIP 2 veto accepted all five proposals at its existing cutoff. The typed hybrid at its earlier active cutoff retained all three valid replacements, rejected the specimen-tube crop and accepted the mechanical pipette. The targeted prompt margin on the proposed crop was positive for each successful replacement and negative for both lookalikes. The same margin was negative on the original box for one valid syringe. This gives a concrete reason to compare appearance evidence on the proposed crop with evidence from both boxes. The five exposed examples serve as diagnostics. A deployment decision requires acquisition-separated development and an independent test. [Frozen before and after appearance scores](../../../data/pair-relocalization/object-presence-diagnostic.json)

## Reproduction

Run from the repository root with the pinned DINOv2, RF-DETR, released detector and archived fitting and development proposals. Restore the reserve photo bytes from the reference source URLs and verify their hashes. The two freeze commands expect fresh output paths.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.pair_relocalization_reserve reference
.venv/bin/python -m scripts.pair_relocalization freeze
.venv/bin/python -m scripts.pair_relocalization_probe freeze
data/rfdetr/env/bin/python -m scripts.pair_relocalization_reserve proposals
data/sam3/xpu-env/bin/python -m scripts.pair_relocalization_reserve release
.venv/bin/python -m scripts.pair_relocalization_reserve features
.venv/bin/python -m scripts.pair_relocalization score
.venv/bin/python -m scripts.pair_relocalization_probe score
.venv/bin/python -m scripts.pair_relocalization_figure
.venv/bin/python -m pytest -q
~~~
