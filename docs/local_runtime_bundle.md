# Offline runtime bundle

The released code and evidence are versioned in Git. The offline bundle carries the selected detector weights, trained RF-DETR recovery checkpoint, verifier, SAM and OCR files, the base CPython distribution, and the three installed Python environments. Each model file is verified by SHA-256 against the corresponding tracked release. The bundle manifest records the Git revision, model hashes, and every installed Python package version in each environment.

```bash
python3 scripts/runtime_bundle.py pack /path/to/medsyringe-runtime
```

Run that command from a clean checkout on the machine whose working release is being captured. Only the files required by the released inference path are selected. The capture corpus, temporary data, and observation database stay outside the bundle. Retain the included model license files and upstream license terms when copying the weights. Access and sharing of the Meta checkpoint are governed by its license. [Model and dataset licenses](engineering.md)

Use a clean checkout at the exact revision recorded as `revision` inside the bundle's `runtime-bundle.json`.

```bash
python3 scripts/runtime_bundle.py restore /path/to/medsyringe-runtime
python3 scripts/runtime_bundle.py verify /path/to/medsyringe-runtime
HF_HUB_OFFLINE=1 .venv/bin/python -m uvicorn medsyringe.app:app --host 127.0.0.1 --port 8765
```

Restoration copies independent files, relinks the packaged interpreter and validates both model hashes and package versions. Existing runtime assets cause a restore to abort before any files are copied. For another operating-system distribution, use the standard worker setup instructions to reproduce a suitable environment. Drivers, glibc and system libraries remain requirements. The app can use the CPU primary detector. The shipped SAM configuration still expects compatible Intel XPU hardware. Upload analysis reports fallback and corroboration outages in the original observation and in the browser. [Worker setup](sam3_markings_liquid.md)

The bundle occupies several gigabytes. A reflink-capable filesystem can copy it efficiently. Keep it separately from the source repository alongside its manifest and the code revision it restores.

## Executed clean-checkout verification

The precision release at `c9e09fab9edea8c63f8144b3f987c68e4eb1b657` is packaged locally under `data/deliverables/medsyringe-runtime-c9e09fa`. It contains 43 checked files with 5,986,486,215 bytes of model assets and three Python environments. The packaged payload occupies approximately 15 GiB. A clean checkout restored the assets, verified their hashes and package versions, and passed all 58 automated tests offline. A later Chromium run verified the exact commit's real Intel inference with the precision worker available, candidate review, native crops and byte-identical export. Its first attempt exceeded the browser timeout during concurrent workloads. The replay used a configurable analysis timeout and explicit process-group cleanup. [Restored precision application](../artifacts/detection_hillclimb/restored_browser_c9e09fa.json)

The preceding bundle for `92a7c0466a4404524203db91144a0cbb8fbe2c0f` is stored at `data/deliverables/medsyringe-runtime`. Fifty-seven tests and two application checks passed in its clean clone. The offline CLI returned one syringe on `IMG_3316.jpeg`. Chromium on Intel Arc verified a native 713 × 3,054 pixel evidence crop and exact original bytes in the ZIP. [Earlier restore](../artifacts/detection_hillclimb/runtime_bundle_restore.json) · [Earlier browser check](../artifacts/detection_hillclimb/restored_native_browser.json)

## Portable evaluation evidence

The separate archive at `data/deliverables/precision-evidence` contains the exact public benchmark raster bytes, image attribution, frozen boxes, ranked inference and expected metrics for five cross-source cohorts. The local full archive also contains capture metadata for three Fil Rouge partitions. The optional capture metadata contains normalized labels, ranked boxes and recorded original hashes. The original photos and observation database stay outside the archive.

```bash
.venv/bin/python -m scripts.package_detection_evidence pack /path/to/public-evidence
.venv/bin/python -m scripts.package_detection_evidence replay /path/to/public-evidence
```

Add `--include-capture-metadata` to include the capture partitions in a local archive. Add `--original-capture /path/to/Images_fil_rougev1` to check the local originals at replay time. The replay was run from an empty directory outside the project, verified all 177 supplied image hashes and reproduced every archived cohort's operating metrics and ranked estimates. This verifies exact metric replay from saved model outputs. Fresh inference can use the separately restored application runtime. [Portable replay evidence](../artifacts/detection_hillclimb/portable_precision_replay.json)
