# Local inference optimization

This work changes how the released models are scheduled and how the appearance verifier receives image data. The checkpoint weights, operating scores, source views, acceptance rules, review candidates, and numerical definitions of the measured fields are carried forward unchanged. Fil Rouge is a regression reference. The benchmark performs no fitting or threshold selection.

## Execution changes

- On Intel Arc, the CPU marking reader and the SAM liquid check can work on the same accepted crop in parallel. The volume estimator consumes both results in detection order. The CPU path runs the checks sequentially. Each optional worker retains its own failure state.
- The text-only appearance verifier can receive the exact padded RGB crops it needs in a lossless PNG. It uses the full image when the prepared crops would be larger. Cropping and padding use the existing coordinates, padding factors, image resampling and ordering.
- **Count only** is an explicit choice in the upload form and the image CLI. All detection and corroboration stages still run. Crop classification, appearance, markings and fill are skipped. The original record and ZIP identify this choice, and the reviewer can still inspect the full-resolution evidence and edit the observation.

The release manifests record the new source checksums. Historical evaluation records retain the original manifests and model identifiers. Versioned weight hashes and the prior acceptance evidence remain pinned. The new qualification uses fresh predictions with the revised runtime.

## Reproduction

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
capture=/path/to/Images_fil_rougev1
runtime=data/sam3/xpu-env/bin/python
"$runtime" -m scripts.benchmark_inference --device xpu --capture-root "$capture" \
  --first baseline --pairs 2 --output /tmp/inference-xpu.json
"$runtime" -m scripts.benchmark_inference --device xpu --capture-root "$capture" \
  --first optimized --cold-only --output /tmp/inference-xpu-cold.json
"$runtime" -m scripts.benchmark_inference --device xpu --capture-root "$capture" \
  --compare-count-only --pairs 1 --output /tmp/inference-xpu-count-only.json
```

For a CPU comparison use `.venv/bin/python` and `--device cpu`. Append `--case` one or more times to limit the run to named cases. Inference wall time includes model execution, internal IPC and liquid field enrichment. The timing excludes image hashing and decode, SQLite writes, browser rendering and initial construction of the engine. Engine setup and the first request in a new process are recorded separately. Warm runs alternate ordering so that both versions run first on different case and repetition combinations.

The comparison verifies all reported decisions and crop fields, with a tolerance of 0.0002 in normalized detector box coordinates and 0.001 in detector scores. Repetitions of the existing Arc path showed drift in both even before optimization. The test also records the exact output hashes and maximum differences in the paired results. The benchmark is local to an Intel Core Ultra 7 258V and its integrated Arc GPU. It is a timing and regression panel, with no statistical claim about transfer to unseen acquisition conditions.

## Inline Intel Arc comparison

These are medians of two warm runs per configuration and case. The baseline uses sequential crop checks and full-scene transfer to the appearance verifier.

| Case | Markings | Baseline | Optimized |
| --- | --- | ---: | ---: |
| Fil Rouge capacity | On | 14.14 s | 9.70 s |
| Fil Rouge fill | On | 8.95 s | 5.28 s |
| Native crowded scene | Off | 13.47 s | 12.46 s |
| Public yellow liquid | On | 7.45 s | 5.51 s |
| Public lookalike | Off | 3.88 s | 4.18 s |

The optimized marking cases saved 26–41% of elapsed time. The public lookalike showed no repeatable gain. The appearance input for a one-object 3024 × 4032 photo went from 12.19 million RGB pixels to 0.10 million pixels for two padded crop views. The bytes sent also depend on PNG content and compression.

Counts, accepted decisions and measured fields stayed the same across all five scenes. The strict full-report gate passed eight of ten pairs. Both failed pairs affected pending proposals on the Fil Rouge fill case. The largest coordinate difference among these proposals was 0.00036 and the largest score difference was 0.00188. The full reports for each request are preserved for inspection. [Paired measurements](../../artifacts/detection_hillclimb/inference_latency_xpu.json)

An isolation run kept the exact primary proposal list fixed for each of the five inputs while running the downstream verifier, markings and segmentation normally. All five output pairs then had identical hashes after removing elapsed time. This separates changes in downstream execution from the observed variation in the unchanged Arc proposer. The original live discrepancies remain recorded above. Reproduce the isolation by adding `--fixed-primary --pairs 1` to the Intel Arc command. [Isolation run](../../artifacts/detection_hillclimb/inference_fixed_primary_xpu.json) · [Latency figure](../../artifacts/figures/inference_latency.png)

## Count-only route

A separate run paired ordinary analysis against **Count only**. On the same machine, all three detection reports passed the numeric parity gate, including their pending review proposals and detector health evidence. The original count-only reports contain no crop measurements.

| Case | Ordinary analysis | Count only | Accepted instances |
| --- | ---: | ---: | ---: |
| Fil Rouge capacity | 11.34 s | 4.04 s | 1 |
| Fil Rouge fill | 6.22 s | 2.39 s | 1 |
| Native crowded scene | 11.88 s | 6.52 s | 3 |

These are single paired observations per case. The crowded image has six independently reviewed bodies. The unchanged count of three illustrates the remaining localization and recall deficit. On the Fil Rouge capacity case the first request after setup took 68.61 s with the baseline and 61.08 s with the optimization in separate fresh processes. The system page cache was left in place. First-use latency remains substantial. [Count-only run](../../artifacts/detection_hillclimb/inference_count_only_xpu.json)

## CPU comparison

Both paired CPU results had identical output hashes after removing elapsed time. The Fil Rouge capacity request took 27.51 s and 25.18 s with sequential checks in both configurations. The first crowded request took 72.97 s and started a separate CPU corroboration worker. The subsequent baseline request took 58.41 s with that worker resident. This pair cannot isolate the CPU effect of the transport change in the crowded scene. [CPU measurements](../../artifacts/detection_hillclimb/inference_latency_cpu.json)

## Restored runtime and native replay

The detached offline checkout restored all 43 pinned model files and three Python environments from the earlier runtime. It then switched to the optimized source. The optimized count-only engine repeated inference on the 23 sealed native public rasters. All prior counts and localized instances were preserved. The result was 39 true positives, four false positives and 20 exact counts. The warm median was 2.99 s. These images were already observed, so the experiment measures regression and runtime on the archive. The prospective 24-acquisition benchmark remains the more demanding test of generalization. [Native replay](../../artifacts/detection_hillclimb/inference_count_native_replay.json) · [Prospective crowd benchmark](../studies/fil-rouge/crowd_capacity_liquid_update.md)

The application adds safeguards for reviewed attributes and joint scale–color comparisons. These changes affect API and browser reports. The model code and bundle weights used in the native replay are the same in the final source. The 198 automated tests include redraw evidence, capacity abstention, partial joint inventory, transcription and OCR invalidation, and byte-identical source and reference export.

## Independent review

Two readers scored the browser-qualified candidate at revision `7a4bf5d` against the full specification. Their scores were 6/10 and 6.5/10. The principal evidence was the 40.7% recall on independent crowded scenes, limited independent attribute and abstention measurements, and the absence of measured operator time savings. The final application corrects the two reproduced defects involving incompatible partial inventory categories and superseded automatic inscriptions. Its later changes leave the detector metrics and the broader acceptance findings open. The review scores apply to the candidate they inspected.

## Delivery verification

Revision `305c3f6` passed the complete 198-test suite in the restored checkout. Both Chromium runs used the actual Intel Arc models. They exercised native image upload, detection, review and ZIP export. The full-analysis control exposed and reviewed piston and graduation evidence. The crowded count-only run retained the pending proposals and recorded the skipped attributes. A separate browser regression redrew a seeded object into a different region and verified that the old fields were withdrawn while the initial prediction was preserved. All temporary local servers and inference workers were closed after the tests. [Full analysis](../../artifacts/detection_hillclimb/inference_final_full_browser.json) · [Count only](../../artifacts/detection_hillclimb/inference_final_count_browser.json) · [Delivery checksums](../../artifacts/detection_hillclimb/inference_delivery.json)
