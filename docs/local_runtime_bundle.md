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

The bundle for commit `92a7c0466a4404524203db91144a0cbb8fbe2c0f` is stored locally at `data/deliverables/medsyringe-runtime`. A clean clone restored the model files, the packaged base interpreter and all three worker environments. The restore verified their hashes and environment package inventories. Fifty-seven tests passed in the restored clone, and the local offline CLI returned one syringe on `IMG_3316.jpeg`. Chromium exercised the same restored clone on Intel Arc, including the real upload, a native 713 × 3,054 pixel evidence crop, and exact original bytes in the ZIP. The browser server was shut down after the check. [Restore evidence](../artifacts/detection_hillclimb/runtime_bundle_restore.json) · [Browser evidence](../artifacts/detection_hillclimb/restored_native_browser.json)
