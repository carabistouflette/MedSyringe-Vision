# Count-preserving syringe relocalization

The [recrop study](automatic_recrop_verifier.md#count-preserving-refinement-preflight) found two correct syringe counts with poor box localization. This follow-up learns when an existing detection can take a better box. The accepted proposal retains the original detection slot and detection score. A replacement is rejected when another slot has a competing claim or when the candidate would duplicate an accepted replacement.

## Data and experiment

The public fitting and development cohorts contain 143 and 60 images, respectively. The fitting pairs come from 83 acquisition groups. A simulated incumbent changes the approved reference box by expanding, shrinking or shifting its borders. Real incumbents are detector proposals whose reference identity is unambiguous. Candidate boxes come from the first pass, automatic recrops and fit-only constructed fragments. The supervision requires a same-object match with IoU at least 0.50 and at least 0.08 IoU gain. If reviewed visible part points exist, including rod, barrel, flange or seal points, the proposal must preserve them within a small tolerance. Intermediate gains or losses with credible identity are excluded from the labels. This yielded 7,920 training pairs with 1,353 positives. The points are used only to supervise fitting labels. At inference the visual variant measures dominant axis, cross-axis contrast and edge support across the crop. It has no runtime part detector. [Fitting pairs](../../../data/relocalization-selector/fitting-pairs.json)

The experiment compares detector confidence plus cross-view overlap, a learned head using detector agreement and box geometry, and a learned head that also uses source pixels. The previous rule with three recrops is the historical comparator. A common screen checks area ratio, containment and separation from other incumbent boxes. All arms may abstain. The development grid varies the internal decision floor, the crop budget and eligibility for multi-instance scenes. The policy is selected using zero lost reference matches at IoU 0.50 and 0.75, unchanged counts, and maximum gains. Setting a higher score cutoff is part of the study protocol and introduces no user-facing control. The development winner is the visual head with first-pass inputs and single-instance replacements. It corrects one box and gains one match at both cutoffs. The head using detector agreement and geometry gains one match at both cutoffs with three recrops but updates two boxes. The score-only consensus arm selected abstention. [Weights and development grid](../../../artifacts/detection_hillclimb/relocalization_selector_policy.json)

Seven newly annotated acquisition groups supply 16 objects and two pipette negatives. The reserve includes a hand-held syringe in the field, a close pair, a group of six with overlapping occlusion, and five arranged syringes. Individual extents follow the visible assembly. Dense candidates with uncertain instance boundaries and a historically exposed source are excluded before inference. SHA-256, perceptual distance, contributor, capture group and URL are checked against project history. One reviewer annotated the reserve. The public stock photos are local evaluation assets and are obtained from their individual source pages. Fil Rouge and all exposed recrop cases are outside fitting and model selection. [Source and annotation manifest](../../../artifacts/detection_hillclimb/relocalization_selector_reserve.json) · [Overlapping syringes by Mikhail Nilov](https://www.pexels.com/photo/close-up-shot-of-a-person-holding-syringes-8670507/) · [DVIDS field image by Abigail Duell](https://www.dvidshub.net/image/9905634/csl-kismayo-medical-training) · [Pexels license](https://www.pexels.com/license/)

## Frozen results

| Variant | Development matches / 44 | Archive matches / 20 | Exposed recrop matches / 5 | New reserve matches / 16 |
|---|---:|---:|---:|---:|
| Released detector | 15 | 4 | 3 | 14 |
| Detector agreement | 15 | 4 | 3 | 14 |
| Learned views and geometry | 16 | 3 | 3 | 13 |
| Learned views, geometry and pixels | 16 | 4 | 3 | 14 |
| Prior rule with three recrops | 13 | 6 | 4 | 13 |

The release gate requires a fresh gain at IoU 0.50 or 0.75 and no lost match at either cutoff in any evaluated cohort. The visual head selected on development qualifies on preservation and fails on fresh gain. It makes no changes on the new photos or either historical cohort. On the fresh reserve, precision is 93.3%, recall 87.5%, F1 90.3%, AP50 0.855 and AP50–95 0.637 for both the released detector and this head. Exact counts are five of seven. The two missed syringes are in the six-syringe overlap scene. A glass transfer tool is the false detection. The remaining comparators have either zero gains or a regression. The learned rule and its comparators remain experimental. [Per-image and per-IoU results](../../../artifacts/detection_hillclimb/relocalization_selector_results.json)

The learned views head and the previous rule with three recrops accept the same truncated barrel on the fresh field image. IoU drops from 0.648 to 0.153 and the count stays at one. Inspecting candidate scores also explains what a blanket relaxation would miss. The visual head ranks the IoU 0.804 candidate correctly at 0.932 on one previously exposed thin syringe. It scores a worse candidate higher on the other exposed miss. Its frozen acceptance floor is 0.975. The next fitting design needs stronger evidence of the visible extent and harder complete-versus-truncated training pairs. The selected rule operates on frames with one released prediction. The fresh multi-object scenes verify that this restriction preserves their outputs. Active replacement in crowded scenes still requires validation. [Localization and candidate audit](../../../artifacts/detection_hillclimb/relocalization_selector_diagnostic.json)

The six-syringe scene has four released matches, while the raw first-pass RF-DETR bank can match all six at IoU 0.50 using the annotations. The crop pass adds no distinct reference match on this scene. The count-preserving operation cannot address a missing slot. [Multi-object proposal ceilings](../../../artifacts/detection_hillclimb/relocalization_selector_diagnostic.json) · [Follow-up on overlapping instance admission](axis_instance_separation.md)

A post-evaluation SAM 3 prompt probe examines the field failure and the false detection of a transfer tool. In the field image, the tighter box covers only 19.4% of the highest-scoring rod proposal and 0% of the second rod proposal near the visible upper end. The incumbent covers both. On the transfer tool, the barrel and rod prompts produce scores of 0.946 and 0.718. These two observations support testing named parts as evidence for retaining the extent of a confirmed device. The pipette mislabel limits use for recognizing new devices. The probe is diagnostic and is outside policy selection. [Named part boxes and containment](../../../artifacts/detection_hillclimb/relocalization_part_probe.json)

The full suite passed 150 tests. Four cover neighbor ownership, cross-view evidence, visible part supervision and one-to-one slot replacement. The cross-role group, identity and hash checks returned zero collisions. Repeating the frozen evaluation reproduced its JSON byte for byte.

## Reproduction

Restore the raster bytes from the URLs in the frozen manifest and the fit, development and archive proposal banks from the previous recrop experiment. The commands that freeze the policy and references expect unused output paths. CPU performs RF-DETR inference and head fitting. The Intel XPU environment runs the released detector.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.relocalization_selector train
.venv/bin/python -m scripts.relocalization_reserve freeze
data/rfdetr/env/bin/python -m scripts.relocalization_reserve collect
data/sam3/xpu-env/bin/python -m scripts.relocalization_reserve baseline
.venv/bin/python -m scripts.relocalization_selector score
.venv/bin/python -m scripts.relocalization_selector_diagnostic
data/sam3/xpu-env/bin/python -m scripts.relocalization_part_probe
.venv/bin/python -m pytest -q
~~~
