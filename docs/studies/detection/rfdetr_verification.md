# Lookalike verification for RF-DETR additions

The locked RF-DETR hybrid recovered two syringes on reserved Zenodo frames. It also introduced three false positives on a Wikimedia Commons photo of hand-painted pens. The acquisition in that photo is a regression control for this experiment. Its crops are excluded from fitting and calibration. The verifier only sees supplemental boxes after the hybrid proposes them. Every v3 box and score passes through untouched. [RF-DETR transfer study](rfdetr_transfer.md) · [Original regression](../../../artifacts/rfdetr_evaluation_previous_external.json)

## Data and controls

| Partition | Images | Instances used for this experiment | Role |
| --- | ---: | ---: | --- |
| New pen training | 8 | 10 pen crops | Fit each binary crop classifier |
| New pen calibration | 4 | 6 pen crops | Compare pen rejection at a fixed recovery constraint |
| New pen reserve | 8 | 16 pen crops | Frozen crop classification and full-image negative control |
| New syringe reserve | 7 | 9 syringe boxes | Frozen real-scene evaluation |
| New Zenodo reserve | 3 | 3 syringe boxes | Additional clip-disjoint evaluation in the public simulation domain |
| Original hybrid calibration | 32 | 6 matched supplemental boxes | Require every initial hybrid recovery to survive |

The pen and syringe photos use manually drawn single-pass box references. Pen images contain no labeled syringe and provide selected pen crops for training or separate crop-level evaluation. The split excludes shared file identities, source URLs, known authors, acquisition groups, identical content and difference hashes within Hamming distance four. Licenses, source links and authors are recorded per photo. Zenodo clips represented in the previous public training or reserve are excluded. Source clip identifiers come from filenames and the remaining simulation frames share one reserved clip. The real-scene reserve adds different authors, backgrounds and viewpoints. [Pen metadata and boxes](../../../artifacts/rfdetr_pen_manifest.json) · [Real syringe reserve](../../../artifacts/rfdetr_syringe_reserve.json) · [Additional Zenodo reserve](../../../artifacts/rfdetr_verifier_reserve.json)

The positive crop pool combines original public polygons' enclosing boxes from the RF-DETR training split and the project adaptation references, including only one representative per Fil rouge burst. Each scene receives equal weight within its class and the two classes contribute equal total weight. The comparisons use a frozen MobileNetV3 Small crop encoder and a frozen DINOv2 Small encoder, each with either tight crops or 18% context padding and with logistic regularization C of 1 or 10. An older general-purpose crop verifier acts as a control at its previously fixed threshold. [TorchVision MobileNetV3 Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) · [Meta DINOv2](https://github.com/facebookresearch/dinov2) · [DINOv2 checkpoint revision and hashes](../../../artifacts/auxiliary_crop_backbone.json)

The selected classifier rejects as many pen calibration crops as feasible while preserving every matched hybrid reference box. Ties prioritize fewer detection errors, then the simpler configuration. Thresholds are derived from calibration score midpoints and locked with the head, code and data hashes. The reserve receives one frozen prediction pass. The release gate requires that every v3 and hybrid match survives on all seven cohorts, that no cohort gains false positives relative to v3 and that at least one new syringe is recovered on a fresh scene. Previous Zenodo and Commons results serve as regression controls.

## Second development selection

The first selection uses only the original hybrid calibration and the new pen calibration photos. A second selection treats the older Zenodo and Commons scores as development evidence for this follow-up. It freezes a new candidate before either verifier is scored on the new pen, syringe or Zenodo reserves. The older Commons false positives and the two older Zenodo recoveries can inform this development decision. The new reserves are excluded from model and threshold selection.

The follow-up compares the eight encoder, padding and regularization combinations using the same fitting data. It also tests pairs in which a supplemental detection must pass both encoders. Every existing hybrid match must survive in all three development cohorts. The primary ranking criterion is the number of false-positive detections on those cohorts. Calibration pen rejection and then the number of encoders break ties. [First classifier selection](../../../artifacts/rfdetr_verifier_selection.json) · [Refined selection](../../../artifacts/rfdetr_verifier_refined_selection.json)

## Measured results

The first fit used 169 positive source scenes and eight negative pen scenes. The original verifier rejected two of six pen calibration crops. The new tight MobileNetV3 classifier rejected six of six while retaining each calibration recovery. The first configuration eliminated two of the three additional Commons false positives and rejected 16 of 16 annotated crops on the independent pen reserve. The second development selection checked eight single encoders and sixteen two-encoder combinations. They reached the same development results at their respective operating points, so the simpler tight MobileNetV3 classifier was selected.

| Cohort | Images | v3 TP / FP / FN | Unverified hybrid TP / FP / FN | Refined verified hybrid TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| Earlier Zenodo, follow-up development | 26 | 1 / 2 / 8 | 3 / 2 / 6 | 3 / 2 / 6 |
| Earlier Commons, follow-up development | 13 | 10 / 3 / 2 | 10 / 6 / 2 | 10 / 3 / 2 |
| Fil rouge development regression | 74 | 70 / 0 / 4 | 70 / 0 / 4 | 70 / 0 / 4 |
| Historical Open Images | 10 | 9 / 0 / 6 | 9 / 0 / 6 | 9 / 0 / 6 |
| New reserved Zenodo clip | 3 | 0 / 0 / 3 | 1 / 0 / 2 | 1 / 0 / 2 |
| New public pen scenes | 8 | 0 / 8 / 0 | 0 / 8 / 0 | 0 / 8 / 0 |
| New public syringe scenes | 7 | 6 / 4 / 3 | 6 / 4 / 3 | 6 / 4 / 3 |

The refined hybrid passed the gate with zero lost previously matched instances and no new false positives in the seven observed cohorts. The new recovery was one of three syringes in one unused Zenodo clip. The new public syringe photos yielded 60.0% precision, 66.7% recall and 63.2% F1 with each detector. The added verifier cannot suppress false positives from the v3 detector by design. The eight inherited mistakes on the pen reserve expose this limitation. All four Fil rouge misses remain unresolved. [First release decision](../../../artifacts/rfdetr_verifier_release_decision.json) · [Refined release decision](../../../artifacts/rfdetr_refined_release_decision.json)

## Active local release

The serialized box-only runtime reproduced the audited TP, FP and FN counts and the matched reference instances on all 141 evaluated images. The base detection replay uses its frozen proposals. The exported RF-DETR graph and MobileNet crop verifier processed the applicable images locally. Added CPU latency was about 0.90 seconds at the median on the Zenodo cohorts. The Fil rouge median rounded to 0 ms because the runtime skips supplemental inference when the base search finds no qualifying weak proposal.

The API and Chromium both processed the fresh recovered syringe through the active release. The tests exercised upload, detection overlays, inventory, persisted corrections and report export. The API recorded the RF-DETR score and MobileNet verification margin alongside the main score. Thresholds are automatic for the same application workflow. [Runtime parity and timing](../../../artifacts/rfdetr_verified_runtime.json) · [Active release manifest](../../../artifacts/rfdetr_verified_release.json) · [Live API evidence](../../../artifacts/rfdetr_verified_app_smoke.json) · [Chromium smoke test](../../../artifacts/rfdetr_verified_browser_smoke.json)

## Reproduction

Prepare the local environments and complete the frozen hybrid comparison first. Restore the DINOv2 Small model files using the revision in the checkpoint manifest. Freeze both selections before scoring either on the new reserves. The pen and real-syringe manifests contain the licensed download URLs and image hashes. The earlier Zenodo ZIP supplies the remaining frames.

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
study_python=data/rfdetr/env/bin/python
.venv/bin/python -m scripts.rfdetr_verifier prepare-reserve
"$study_python" -m scripts.rfdetr_verifier fit
"$study_python" -m scripts.rfdetr_verifier_refine select
for cohort in fresh pen_reserve syringe_reserve; do
  .venv/bin/python -m scripts.rfdetr_verifier base --cohort "$cohort"
done
for cohort in previous_zenodo previous_external fil_rouge historical fresh pen_reserve syringe_reserve; do
  "$study_python" -m scripts.rfdetr_verifier audit --cohort "$cohort"
done
"$study_python" -m scripts.rfdetr_verifier summarize
for cohort in previous_zenodo previous_external fil_rouge historical fresh pen_reserve syringe_reserve; do
  "$study_python" -m scripts.rfdetr_verifier_refine audit --cohort "$cohort"
done
"$study_python" -m scripts.rfdetr_verifier_refine summarize
```

### Export the local runtime

The box-only TorchScript export uses the frozen segmentation-trained RF-DETR checkpoint. The runtime compares the exported model with the eager model's fixed predictions on the seven audit cohorts. A passing comparison writes the automatic release manifest. If there is no candidate that passes the audit gate, the export command reports the failed gate.

```bash
"$study_python" -m scripts.rfdetr_verified_release export
.venv/bin/python -m scripts.rfdetr_verified_release verify
```

For another machine, retain the trained `data/rfdetr/verified-recovery.pt` file and the matching release manifest and runtime evidence, or repeat the training, verification and export. The production loader uses the existing pinned PyTorch and TorchVision environment. It does not need the isolated RF-DETR research environment. [Training, proposals and system setup](rfdetr_transfer.md)
