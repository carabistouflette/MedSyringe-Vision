# Syringe assembly identity and thin-object views

This experiment addresses two recorded failures of the native-crop verifier. A single syringe's needle and hub were counted again on an earlier holdout, and the proposal bank missed two thin assemblies installed in a syringe pump. The study separates instance association from proposal generation and tests both with frozen development operating points.

## Inputs and pair supervision

The previously audited 143 public fitting photos and 60 public development photos remain separated by acquisition. The fitting-only bank supplies 33 proposals with partial overlap and typed part annotations. Visual review identifies 26 attached fragments, three alternate views of a complete assembly and two distractors. Two ambiguous boxes remain outside this manual supervision. The reproduced review map connects every sample to its original reviewed coordinates. [Manual pair labels](../../../artifacts/detection_hillclimb/syringe_assembly_fit_review.json) · [Example review sheet](../../../data/syringe-assembly-study/fragment-review-0.png)

The two multinomial logistic heads use the same 1,120 training pairs. There are 457 same-assembly pairs, 318 fragment pairs, 42 distinct-assembly pairs and 303 distractors. Forty-two distinct pairs come from just seven acquisition groups, which constrains transfer. The supplementary sources include 292 cropped fragments constructed from visible annotated axes or elongated fit-only bounds, 192 jittered references, 262 high-overlap RF-DETR proposals and 301 mined background regions. Mined backgrounds and oriented crop fragments are weak labels. Group and class weights limit the influence of repeated views. [Model and supervision counts](../../../artifacts/detection_hillclimb/syringe_assembly_pair_model.json)

The box head uses overlap, size and displacement. The image head adds pixel-supported long-edge orientation, transverse disagreement, edge strength and color statistics from source crops. Manual barrel, seal, rod or flange points can guide fitting crop placement. At inference, the image head estimates axes from gradients with a reliability value. It does not require human landmarks or claim that individual anatomical parts have been localized. Pairs can be scored even when they fall below the standard duplicate overlap. The release comparison keeps all existing boxes and separately tests vetoes and eligibility for two overlapping boxes to count.

## Fixed window comparisons

All views use the same frozen RF-DETR Nano checkpoint and 320-pixel input. The control contains the full image and the four 66-percent views already archived. The challengers add nine square views of 43 percent, ten full-axis strips with 28 percent transverse extent, or both. Fixed tile-border filtering drops candidates truncated by an internal edge. The ranked pools share the 0.025 prefilter, 0.62 suppression IoU and 28-box cap. Higher zoom and preserved longitudinal context can therefore be tested against the same two-stage candidate budget. Extra CPU latencies are measured independently of the archived baseline.

The 60 development images select floors and model settings by requiring no lost IoU 0.50 matches, no per-image increase in false detections and no worsening of count error. The selected control is the standard five-view bank with ordinary overlap suppression. It gains one development match with zero added false positives. Every view and association family retains its best qualifying development operating point for comparison on the held evidence. [Frozen selection and grid](../../../artifacts/detection_hillclimb/syringe_assembly_selection.json)

## Reserve

Three newly admitted positive acquisitions show a syringe in an infusion pump and two gloved-hand devices. Three negative acquisitions show lab pipettes and test tubes. Each public image was annotated and hash-bound before prospective inference. A similar published syringe pump and a Hamilton syringe were rejected for known contributors and visual overlap. Another visually ambiguous glass assembly photo was excluded. Fil Rouge remains outside fitting, augmentation and selection. [New references and exclusions](../../../artifacts/detection_hillclimb/syringe_assembly_reserve.json) · [Infusion pump image](https://commons.wikimedia.org/wiki/File:Syringe_infusion_pump.jpg) · [Gloved-hand syringe](https://www.dvidshub.net/image/6900270/medical-team-provides-care-fort-bliss-dona-ana-complex) · [Glass pipette](https://commons.wikimedia.org/wiki/File:RBC_Pipette.jpg)

## Results and failure analysis

| Proposal bank | Development oracle / 44 | Previously seen oracle / 17 | New reserve oracle / 3 | Extra median CPU time in development |
|---|---:|---:|---:|---:|
| Standard | 39 | 14 | 3 | 0 s |
| Standard and fine squares | 38 | 14 | 3 | 2.05 s |
| Standard and strips | 39 | 15 | 3 | 2.28 s |
| Standard, fine squares and strips | 39 | 15 | 3 | Both passes |

Oracle counts use annotation-assisted one-to-one matching at IoU 0.50. The strips supply a ninth distinct candidate on the earlier nine-syringe photo. In development the fine square views displace an existing match at the fixed 28-box budget. The two thin Chemyx pump assemblies remain absent from the automatic candidate pools. [Proposal ceilings and timings](../../../artifacts/detection_hillclimb/syringe_assembly_tile_development.json) · [Per-image comparisons](../../../artifacts/detection_hillclimb/syringe_assembly_results.json)

Every frozen policy leaves the released results unchanged on the six new photos and 12 previously seen photos in the view comparison. The new reserve has zero matched syringes out of three, one false-positive box and four exact counts out of six. The three new true instances are represented in the standard raw proposal bank. The best-localized candidate for each has a low score, between 0.036 and 0.115. None of the usable candidates reaches the selected automatic confidence floor. The 35-photo secondary archive replay also preserves all existing results under the four standard-view policies. The prospective gain criterion therefore remains unmet. [Complete metrics and acceptance verdict](../../../artifacts/detection_hillclimb/syringe_assembly_results.json) · [Secondary archive guard](../../../artifacts/detection_hillclimb/syringe_assembly_archive_guard.json)

The targeted fragment replay starts from the exact previously accepted additions. Ordinary overlap misses the needle and hub fragment. The pixel head assigns 0.611 combined probability to same assembly or attached fragment and vetoes the extra box at the development operating point. The box-only head assigns 0.182 to the same two classes for this pair. In the earlier crowded photo, applying the veto policy with the conventional duplicate rule preserves one of the two previously recovered instances. The other is a genuine overlapping neighbor. The learned distinct-instance probability of that pair is 0.827 and falls short of the frozen exception. Both errors require joint coverage when the rule is evaluated for deployment. [Fragment and crowd replay](../../../artifacts/detection_hillclimb/syringe_assembly_results.json) · [Needle and hub photograph](https://www.dvidshub.net/image/9190822/uss-gerald-r-ford-cvn-78-conducts-deployment-shot-exercise)

Two manually located recrops of the earlier Chemyx image recover both pump assemblies in the raw RF-DETR pool with either tested orientation. In the tighter horizontal crop the best overlaps are 0.867 and 0.869, with scores of 0.050 and 0.010. This diagnostic indicates that automatic region selection and low-confidence verification both matter. Expert crops use prior scene knowledge and are evaluated separately from automatic results. [Error audit and crop probe](../../../artifacts/detection_hillclimb/syringe_assembly_error_audit.json) · [Chemyx photograph](https://commons.wikimedia.org/wiki/File:Chemyx_SKE10.jpg)

The independent replay reproduced the development selection and manual-review input map byte for byte. The full test suite passed 141 tests, including five checks for strip coverage, fragment sampling, weak-overlap veto, distinct-neighbor handling and grouped training weights.

## Automatic recrop trigger pilot

The pilot ranks the standard bank by confidence. Each seed is expanded by half its width and height on every side, then clipped to the source image. Geometric coverage requires the crop to contain at least 90% of the reference box and occupy no more than 40% of the frame. Of the 29 unmatched development references, the first 3, 6, 12 and 28 seeds cover 18, 22, 24 and 27 respectively. The annotations measure coverage after crop placement. [Seed ranks, input hashes and coverage](../../../artifacts/detection_hillclimb/syringe_recrop_trigger_audit.json)

The pinned RF-DETR checkpoint ran on those automatic windows for the previously exposed six-image reserve and the Chemyx photo. With up to six seeds, the raw union gains one Chemyx match at IoU 0.663. Its best-localized crop candidate scores 0.032. Increasing the budget to twelve leaves the second Chemyx assembly just below the matching threshold at IoU 0.487. The two gloved-hand syringes reappear in the three-seed crops, and the infusion-pump syringe reappears in the six-seed crops. All three already had a usable first-pass proposal. Their best-localized crop candidates score below the corresponding first-pass candidates. The three negative scenes yield hundreds of raw candidates at the exploratory detector floor. This replay measures proposal potential on known scenes. Acceptance, false-positive rates after filtering, and latency benefits remain unevaluated. [Per-window RF-DETR replay](../../../artifacts/detection_hillclimb/syringe_recrop_rf_replay.json)

The follow-up study compares fixed crop budgets with a spatially diverse fallback, and combines multi-view agreement with visible barrel evidence. Its acquisition groups separate fitting, selection and fresh evaluation. Existing released identities and per-image count error form the regression guard. [Automatic recrop verification](automatic_recrop_verifier.md)

## Reproduction

Use the existing project, RF-DETR and Intel Arc Python environments. Restore the original image bytes listed in the prospective manifest before running its freeze command. Preserve the candidate banks and historical public frames from the preceding study.

~~~bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.syringe_assembly_review
.venv/bin/python -m scripts.syringe_assembly_model
data/rfdetr/env/bin/python -m scripts.thin_syringe_tiles development
.venv/bin/python -m scripts.syringe_assembly_study choose
.venv/bin/python -m scripts.syringe_assembly_reserve freeze
data/sam3/xpu-env/bin/python -m scripts.syringe_assembly_reserve baseline
data/rfdetr/env/bin/python -m scripts.syringe_assembly_reserve proposals
data/rfdetr/env/bin/python -m scripts.thin_syringe_tiles history
data/rfdetr/env/bin/python -m scripts.thin_syringe_tiles prospective
.venv/bin/python -m scripts.syringe_assembly_study score
data/rfdetr/env/bin/python -m scripts.syringe_assembly_diagnostic
.venv/bin/python -m scripts.syringe_recrop_trigger_audit
data/rfdetr/env/bin/python -m scripts.syringe_recrop_trigger_audit --replay-rf
.venv/bin/python -m pytest -q
~~~
