# Local recovery, syringe markings and liquid appearance

## Detection recovery

The still-image checkpoint is Meta SAM 3 at revision `3c879f39826c281e95690f02c7821c4de09afae7`. The 3.44 GB safetensors file matches the published SHA-256. Authorization is used for the checkpoint download. Inference, training and evaluation run locally. The production hybrid invokes SAM only after the existing detector returns no accepted object. SAM proposes boxes using the English text query `syringe` on the full image and four overlapping tiles. The 504-pixel model rebuilds its positional embeddings for that grid and loads the official weights. The Intel Arc 140V runs the model through PyTorch XPU. [SAM 3 model and terms](https://huggingface.co/facebook/sam3) · [SAM 3.1 video release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) · [PyTorch XPU documentation](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)

The random-weight benchmark measured about 46 seconds per CPU pass at 1008 pixels and 13 seconds at 504. With real weights and a visible syringe, two 504-pixel CPU passes took 14.1 and 15.9 seconds. On the same control, Intel XPU took 8.4 seconds cold and 1.42 seconds warm, with essentially identical boxes and scores. The executable fallback uses five views and several seconds per image where it runs. [CPU 1008 architecture probe](../../../artifacts/sam3_cpu_probe.json) · [CPU 504 architecture probe](../../../artifacts/sam3_cpu_probe_504.json) · [Pretrained CPU control](../../../artifacts/sam3_measured_cpu_504.json) · [Pretrained Intel Arc control](../../../artifacts/sam3_measured_xpu_504.json)

| Cohort | Images | Existing TP / FP / FN | With SAM TP / FP / FN |
| --- | ---: | ---: | ---: |
| Fil rouge development capture | 74 | 70 / 0 / 4 | 74 / 0 / 0 |
| Previously consulted Open Images | 10 | 9 / 0 / 6 | 11 / 0 / 4 |
| Previously consulted Commons | 13 | 10 / 3 / 2 | 10 / 3 / 2 |
| Pen regression | 8 | 0 / 8 / 0 | 0 / 8 / 0 |
| New public reserve | 13 | 4 / 8 / 3 | 4 / 8 / 3 |

Calibration used 47 previously consulted development scenes, selected one additional match and introduced zero false positives. The 13-image public reserve was frozen before selection. It contains seven syringe photos and six syringe-negative photos of pens or pipettes, with file hashes, author groups and individually reviewed draft boxes. The fallback preserved every previously matched reference across all five audit cohorts. It recovered all four Fil rouge instances and two historical Open Images instances. The new public reserve measured no change. The five-cohort test establishes a gain in the supplied capture and avoids claiming general perfect detection. [Frozen settings](../../../artifacts/sam3_fallback_selection.json) · [Public reserve](../../../artifacts/sam3_reserve.json) · [All 74 Fil rouge cases](../../../artifacts/sam3_audit_fil_rouge.json) · [Local worker verification](../../../artifacts/sam3_runtime.json)

## Reading capacity and classifying the visible scale

The local PP-OCRv6 detector and recognizer inspect the detected syringe crop in all four quarter turns and add barrel-aligned views when long-edge geometry supports a rotation. The pipeline then re-reads small unit patches and higher-resolution crops of the likely terminal marking. Ordered graduations guide where to look. The application requires a visually supported number and mL evidence before it reports nominal capacity. It may report a scale type from coherent ordered graduations without assigning a capacity. Contradictory readings trigger abstention. The structured observation distinguishes nominal capacity from apparent liquid volume. [PP-OCRv6 documentation](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)

The frozen OCR operates on all 74 object crops from the verified detector. It reports 36 nominal capacities and 37 scale types, all consistent with the recorded references. Each of the 36 reported number-and-unit regions was visually inspected. The burst-disjoint subset has 20 scenes, with 11 read capacities and 11 scale types at zero observed errors among responses. All images share one capture session. The evaluation still lacks a separate physical syringe population with independent type annotations. The nine explicitly visible control markings produced six exact numeric hits with the selected tiny reader in the source orientation and eight after physically rotating the input photos by 180 degrees. Complete unit-and-number readings were five and seven, with no wrong accepted capacities. The medium control retrieved all nine numeric tokens and seven full capacities, with materially higher latency. The active reader uses the tiny mode with local patch refinement. [Reference and rotation controls](../../../artifacts/capacity_ocr_reference_results.json) · [All detected crops](../../../artifacts/capacity_ocr_detected_results.json) · [Worker orientation check](../../../artifacts/capacity_ocr_runtime.json)

## Measuring liquid appearance

The initial warm-color segmentation often picked up a hand through the clear barrel. The refined experiment fits a liquid-region head over frozen DINOv2 patch features using visually drawn centerline traces, with whole capture bursts assigned to fitting, calibration and holdout. It restricts the photometric sample to warm pixels supported by both the syringe silhouette and the learned interior. The visual estimate uses median CIELAB L* and the middle 50% of the sampled pixels. Categories use display-lightness bands below 35 for dark and at least 65 for light. Exposure and illumination affect this appearance measurement. The source data provides no absolute calibration of the liquid's color or transparency. [Liquid centerline annotations](../../../artifacts/liquid_region_scribbles.json) · [Grouped localization experiment](../../../artifacts/liquid_region_study.json)

The color-only baseline returned 73 measurements, of which 55 passed the final centerline check. The selected method returned 72 measurements, all passing that check, with a median absolute difference of 2.11 L* versus samples near the traces. Four misplaced development traces were corrected under magnification before the final fit and validation. The 13 photos from four held-out bursts were excluded from both fitting and calibration. All 13 yielded supported measurements, with a median difference of 1.44 L*. The location test checks that the bounding rectangle of the sampled region covers the midpoint of the reviewed trace within a fixed tolerance. The L* comparison tests consistency within the photograph. Calibration across lighting, cameras and different sources of liquid remains open. [Per-image validation](../../../artifacts/liquid_appearance_validation.json)

Relative opacity can be estimated from a registered clear-barrel reference of the same view, exposure and textured background. The function estimates luminance contrast attenuation and abstains when the paired data lacks sufficient agreement. It recovered the three synthetic attenuation levels of 0.15, 0.45 and 0.75. The original dataset contains no matching clear-barrel reference, so each of its real opacity fields is unknown. No fill volume is assigned from the nominal capacity. [Opacity and apparent lightness checks](../../../artifacts/liquid_appearance_validation.json)

## Reproduce and restore the local runtime

Use the existing root environment for the application and reproduce the optional workers in their own environments. Restore the SAM model only after the official Hugging Face gate is accepted. Use an authorized local read token for the download. Keep the official PP-OCRv6 detector and recognizer folders under `data/ocr-v6/models/`, as recorded in the release manifest. The worker makes local predictions with its model folders and offline settings.

```bash
uv venv --python .venv/bin/python data/sam3/xpu-env
uv pip sync --python data/sam3/xpu-env/bin/python \
  --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match \
  scripts/experiments/baselines/sam3-xpu-requirements.txt
uv venv --python .venv/bin/python data/ocr-v6/env
uv pip sync --python data/ocr-v6/env/bin/python scripts/experiments/baselines/ocr-v6-requirements.txt
data/sam3/xpu-env/bin/python -m scripts.experiments.expanded.sam3_study download
```

The prepared manifests preserve draft annotations, local sources and hashes. The reports record the exact calibration and measured outputs. The commands below replay the executable detection, marking and photometric release checks after restoring the frozen data and checkpoints.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.experiments.expanded.sam3_release
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.experiments.measurement.marking_release
HF_HUB_OFFLINE=1 data/sam3/xpu-env/bin/python -m scripts.experiments.measurement.liquid_fit
HF_HUB_OFFLINE=1 data/sam3/xpu-env/bin/python -m scripts.experiments.measurement.liquid_measure study detected
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.experiments.measurement.liquid_release
```

The combined cold API control analyzes the original `IMG_3229.HEIC`, recovers one syringe from SAM 3, reads 20 mL, measures 48.2 L* and leaves opacity unknown. Observation creation took 41.5 seconds including model startup and OCR on the tested machine. [Real API control](../../../artifacts/sam3_ocr_liquid_api_smoke.json) · [Chromium control](../../../artifacts/sam3_ocr_liquid_browser_smoke.json)

For a controlled opacity pair, crop equivalent pixel locations of the two registered inputs before running the following command.

```bash
.venv/bin/python -m scripts.experiments.measurement.liquid_measure opacity observed.png clear_reference.png --matched-setup
```
