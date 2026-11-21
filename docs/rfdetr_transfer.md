# RF-DETR segmentation transfer and hybrid detection

## Experimental design

This study compares three local RF-DETR Seg Nano checkpoints. The public-only arm learns syringe instance masks from the Zenodo simulated phlebotomy dataset, licensed CC BY 4.0. The direct arm fine-tunes the publisher checkpoint on project development boxes. The transfer arm starts from the public-only checkpoint and uses the same boxes, training settings and update budget as the direct arm. It disables the segmentation head during adaptation because the project reference annotations contain boxes. The public-only checkpoint retains its mask head. [Zenodo data and license](https://doi.org/10.5281/zenodo.16924786) · [Acquisition and annotation methods](https://arxiv.org/html/2602.04624v1) · [RF-DETR Seg Nano reference](https://rfdetr.roboflow.com/latest/reference/seg_nano/) · [Publisher checkpoints](https://rfdetr.roboflow.com/latest/learn/pretrained/)

Runs use a local Intel Core Ultra 7 258V CPU, four PyTorch threads, a 240-pixel square input, batch size four, seed 2602, learning rate 0.0002 and a frozen encoder. The public run trains for three epochs and 108 batches. Both adaptation arms train for two epochs and 46 batches. Resolution was reduced from the model's default 312 pixels to bound CPU training time. The native trainer selects the public checkpoint by mask AP and the two adaptation checkpoints by box AP. The comparison measures the combined effect of mask supervision, exposure to new images and extra updates. Each arm has one seed. Checkpoint inspection verified that adaptation from the public phase omitted exactly 35 mask head tensors. The 510 remaining tensor shapes matched and all 223 encoder tensors were identical.

| Partition | Images | Syringe boxes | Negative images | Use |
| --- | ---: | ---: | ---: | --- |
| Zenodo training | 144 | 117 | 33 | Genuine mask supervision |
| Zenodo validation | 15 | 8 | 7 | Checkpoint selection and small-object calibration on held-out filename-derived clips |
| Project development | 89 | 62 | 31 | Box adaptation, including 20 Fil rouge burst representatives |
| Project calibration | 17 | 6 | 11 | Adaptation checkpoint selection and operating points |
| Zenodo reserve | 26 | 9 | 17 | Locked evaluation on nine unused filename-derived clips |

The public training sample is a subset of the earlier verified 160-frame manifest. The exporter reads the original polygons from the ZIP, checks that each bounding rectangle matches the previous import, rejects degenerate geometry, excludes one frame with an ambiguous clip key and builds COCO data. Project references stay box supervised. The public training and validation clip sets are disjoint.

The Zenodo reserve uses only the archive's official test directory. The split supplied by the publisher mixes clip identifiers, so 1,697 frames from clips already represented in the earlier 160-frame sample were excluded. The selection additionally excludes unknown clip identifiers, repeated export parents, exact duplicates and perceptual hashes at Hamming distance ≤4 from consulted images or prior reserve selections. The reserve was generated and the supplied labels visually spot-checked before performance measurement. The photos share a phlebotomy simulation environment and several positives share a clip. The effective sample size is smaller than the frame count. Filename-derived clips are a proxy for acquisition grouping and provide no guarantee of independent operators, rooms or physical syringes. Generic pretraining contamination is unknown. The reserve evaluates transfer within Zenodo. Performance on a new Fil rouge session remains unmeasured. [Public training data manifest](../artifacts/rfdetr_public_data.json) · [Adaptation manifest](../artifacts/rfdetr_adapt_data.json) · [Reserve manifest](../artifacts/rfdetr_audit_manifest.json)

The hybrid preserves each accepted deployed v3 proposal and can append RF-DETR proposals. Calibration compares a full image with a full image plus four overlapping tiles, with and without corroboration from lower-scoring v3 boxes. The project calibration boxes have a median area of 45.54% of the image and the reserve boxes have a median area of 0.37%. Calibration pools the 17 project scenes and 15 Zenodo validation frames so that small objects influence the operating point. Neither subset contributes to supervised updates. Precision must hold in both calibration subsets. Thresholds and model choice are frozen before evaluating the reserve. The standalone threshold for each arm is selected on calibration F1. Audit cohorts are the reserved Zenodo clips, the 74 unique Fil rouge images and two previously used public regression cohorts. Instance matching uses one-to-one assignment at IoU ≥0.50. Each reference box matched by v3 must remain matched. A candidate must recover new instances while maintaining precision on calibration. The locked check requires zero lost matches, no lower precision in any cohort and at least one recovery on the reserve. The application exposes no experiment thresholds.

## Measured training and calibration results

| Checkpoint | Best epoch | Checkpoint-selection cohort | Box AP50 | Box AP50–95 | Mask AP50–95 | CPU wall time |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Zenodo masks | 3 | Zenodo validation | 75.25% | 40.22% | 26.15% | 623 s |
| Direct box adaptation | 2 | Project validation | 36.21% | 29.81% | — | 250 s |
| Mask-to-box transfer | 1 | Project validation | 68.62% | 50.83% | — | 246 s |

The second adaptation epoch reduced transfer box AP50–95 to 31.70%, so checkpoint selection retained epoch 1. Both adaptation arms used identical update budgets. The training comparison on project validation favored transfer by 21.02 percentage points of box AP50–95. The public-only checkpoint was selected for the hybrid, which optimizes recovery of v3 misses at a fixed operating point across both calibration domains. [Public curves](../artifacts/rfdetr_public_metrics.csv) · [Direct curves](../artifacts/rfdetr_direct_metrics.csv) · [Transfer curves](../artifacts/rfdetr_transfer_metrics.csv)

Among 336 calibration configurations, the selected full-image plus tiles setting appended six matched detections in Zenodo validation. It kept the project results and the nine existing false positives. The combined counts went from 6 TP / 9 FP / 8 FN to 12 TP / 9 FP / 2 FN, with no lost instance match. Precision rose from 40.0% to 57.1%, recall from 42.9% to 85.7% and F1 from 41.4% to 68.6%. The public-only model covered 6 of 8 v3 misses on whole-frame proposals and all 8 with tiling before the corroboration and score checks. Fine-tuned transfer covered 3 misses on whole-frame proposals and 5 with tiling. [Frozen selection and proposal diagnostics](../artifacts/rfdetr_selection.json)

## Reserved clips and error analysis

| Detector | TP | FP | FN | Precision | Recall | F1 | Box AP50 | Box AP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deployed v3 | 1 | 2 | 8 | 33.3% | 11.1% | 16.7% | 18.6% | 11.9% |
| Zenodo masks alone | 3 | 4 | 6 | 42.9% | 33.3% | 37.5% | 34.4% | 19.8% |
| Direct box adaptation alone | 0 | 1 | 9 | 0.0% | 0.0% | 0.0% | 2.2% | 0.7% |
| Mask-to-box transfer alone | 3 | 7 | 6 | 30.0% | 33.3% | 31.6% | 19.2% | 10.3% |
| Locked hybrid | 3 | 2 | 6 | 60.0% | 33.3% | 42.9% | 25.7% | 16.5% |

Both additional detections came from clip 0031. The two boxes occupy 0.37% and 2.88% of their respective images. The reserve provides evidence of recovery in one additional clip, with five positive clips represented across the reserve. The median measured cost of the public full-image plus tiling branch was 1.12 seconds per reserve image on the local CPU (95th percentile 1.27 seconds), excluding v3 and image loading. The supplemental branch gives accepted boxes the application threshold as their score. Its AP rankings consequently use ties at that threshold.

Of the six remaining false negatives, four had a matching RF-DETR box below the locked auxiliary cutoff. One matching box near gloved hands exceeded the auxiliary cutoff but lacked the required v3 corroboration. The final miss was a very small syringe near a gloved hand. The best RF-DETR proposal reached IoU 0.474, and v3's wider box reached IoU 0.192. The other legacy false positive was on a frame with no labeled syringe and localized to equipment on the workstation. [Reserve metrics and per-instance evidence](../artifacts/rfdetr_evaluation_fresh.json)

## Regression checks and release decision

| Cohort | Images | v3 TP / FP / FN | Hybrid TP / FP / FN | Previously matched instances lost |
| --- | ---: | --- | --- | ---: |
| Reserved Zenodo clips | 26 | 1 / 2 / 8 | 3 / 2 / 6 | 0 |
| Fil rouge development originals | 74 | 70 / 0 / 4 | 70 / 0 / 4 | 0 |
| Historical Open Images | 10 | 9 / 0 / 6 | 9 / 0 / 6 | 0 |
| Wikimedia Commons regression | 13 | 10 / 3 / 2 | 10 / 6 / 2 | 0 |

The locked hybrid fails the precision release check. Precision on the previously consulted Commons cohort falls from 76.9% to 62.5% and F1 from 80.0% to 71.4%. All three new false positives appear in one negative photo of hand-painted pens. They are detections of three separate pen-like objects. This confirms that the geometric corroboration is vulnerable to shared errors between both detectors. This unverified branch failed the release check. The follow-up study applies a separate gate to verified additions. [Lookalike verification study](rfdetr_verification.md) [Locked release decision](../artifacts/rfdetr_release_decision.json) · [Commons error records](../artifacts/rfdetr_evaluation_previous_external.json) · [Pen photograph by Rudolph.A.furtado, CC BY 4.0](https://commons.wikimedia.org/wiki/File:%22_Reve_%22_hand_painted_pens_at_%22_India_Pen_Show-_2026%22.jpg)

For Fil rouge, the evaluation covers all 74 unique images. The 75 source files and the byte-identical duplicate were verified against the original inventory. Live v3 outputs on two original HEIC files matched replayed proposals. Four draft references remain unmatched at IoU 0.50. On IMG_3229 and IMG_3235, RF-DETR proposed matching boxes at scores of 0.066 and 0.063, well below the calibrated auxiliary cutoff. The best auxiliary IoUs on IMG_3283 and IMG_3318 were 0.304 and 0.292. All Fil rouge images come from one acquisition and serve as a development regression. [Fil rouge per-instance evidence](../artifacts/rfdetr_evaluation_fil_rouge.json) · [Open Images regression](../artifacts/rfdetr_evaluation_historical.json)

The three training runs, grouped data checks, frozen comparison and four-cohort audits were executed locally. The delivered checkpoints and metrics support the observed recovery of small syringes within the Zenodo environment. Precision on syringe-like objects remains a constraint for activating the branch. The two recoveries in one clip, single training seed and draft Fil rouge references limit the scope of the measured result. [Training curves and audit figure](../artifacts/figures/rfdetr_transfer.png)

## Reproduction

Run from the repository root after preparing the original and public project sources. Restore the public Zenodo ZIP to `data/zenodo/phlebotomy.zip` and generate the earlier sample if needed with `scripts.public_transfer_study`. The experimental environment is separate from the application. The official initial checkpoint is checked against its recorded SHA-256 at the start of training. [Engineering guide](engineering.md)

```bash
uv sync --locked
uv venv --python .venv/bin/python data/rfdetr/env
uv pip sync --python data/rfdetr/env/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  scripts/rfdetr-requirements.txt
curl -fL https://repo.roboflow.com/rfdetr/rf-detr-seg-n-ft.pth \
  -o data/rfdetr/rf-detr-seg-nano.pt
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
study_python=data/rfdetr/env/bin/python
"$study_python" -m scripts.rfdetr_transfer prepare public
"$study_python" -m scripts.rfdetr_transfer prepare adapt
"$study_python" -m scripts.rfdetr_transfer prepare audit
"$study_python" -m scripts.rfdetr_transfer train public
"$study_python" -m scripts.rfdetr_transfer train adapt --run direct --epochs 2
"$study_python" -m scripts.rfdetr_transfer train adapt --run transfer --epochs 2 \
  --weights data/rfdetr/runs/public/checkpoint_best_total.pth
"$study_python" -m scripts.rfdetr_compare calibrate
.venv/bin/python -m scripts.rfdetr_compare baseline
for cohort in fresh fil_rouge historical previous_external; do
  "$study_python" -m scripts.rfdetr_compare audit --cohort "$cohort"
done
"$study_python" -m scripts.rfdetr_compare summarize
.venv/bin/python -m scripts.rfdetr_figures
```

Training saves epoch checkpoints with optimizer state, publisher configs, CSV metrics and an inference checkpoint under `data/rfdetr/runs/`. To continue an interrupted run, repeat the original command with `--resume` pointing to its last `checkpoint_<epoch>.ckpt`. The local data directory is excluded from git. Preserve the checkpoints and experiment outputs there when moving the study to another machine.
