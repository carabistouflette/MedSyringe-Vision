# Product validation for counting and liquid fields

The [new public validation report](../studies/fil-rouge/crowd_capacity_liquid_update.md) records the expanded instance-level dataset, the prospective SAM comparison, the experimental small-scale reader, and the paired optical exposure study.

The [low-count recovery report](../studies/detection/low_count_recovery.md) records the local proposal verifier, the duplicate association fix, the failed prospective pilot, and OCR-independent color transfer on manually aligned public barrel crops.

The [syringe part verifier report](../studies/detection/part_verifier.md) records a controlled comparison of anatomical geometry, DINOv2 crop appearance, and their combination, with the earlier tray protected as a regression case.

The application reports the count of visible syringe bodies, nominal capacity, liquid presence, apparent color and lightness, apparent volume and relative opacity. Each field carries its own evidence and may remain indeterminate. Original and reviewed results are retained in observations and their ZIP exports. The automatic visual-scale families in the original release remain 20 mL and 60 mL. The numerical nominal-capacity field can also retain explicitly read or reviewed capacities outside those families.

| User requirement | Status | Evidence and scope |
|---|---|---|
| Collect at least 500 public photos | Implemented and validated at image level | 547 provider-resolution positives passed visual presence review and checksum verification. Another collection contains 742 visually verified photos and frames. The totals are reported separately and do not establish a single deduplicated collection of 1,289 acquisitions. Full instance and attribute annotation remains in review. [Dataset card](../datasets/public_syringe_dataset.md) · [Additional collection](../datasets/public_bulk_syringe_collection.md) |
| Keep evaluation images out of fitting | Implemented and validated for the documented sources | Fil Rouge remains a protected regression reference. New intake and partition checks lock out 229 public rows from projects already represented in evaluation, including a related project alias. [Public protocol](../datasets/public_training_protocol.md) · [Exposure audit](../../artifacts/detection_hillclimb/expanded_public_catalog_exposure.json) |
| Count overlapping devices without confusion | Implemented but not fully validated for release | The SAM mask experiment found all ten visible bodies in the supplied example. The fixed fusion improved the exact counts of six prospectively reviewed public photos from 4/6 to 5/6 and matched 8/9 visible instances with zero false positives. The released engine retains its original count. [Crowd study](../studies/detection/crowded_sam_evaluation.md) · [Prospective results](../../artifacts/detection_hillclimb/crowded_sam_prospective.json) |
| Read maximum capacity when supported | Implemented but not fully validated | The app exposes mL and the source of each reading. The rotation controls include correct upright and 180° reads of a visible 20 mL marking. The small public 5 mL marking still needs review. [OCR runtime](../../artifacts/fill/graduation_capacity_runtime.json) |
| Determine whether liquid is present | Partially implemented | A supported liquid-colored region or a positive lower fill bound supports automatic presence. The reviewer can confirm empty, present or indeterminate. The independent part-prompt trial produced no empty answer stable across both orientations. The automatic path abstains on the visually empty examples. [Field replay](../../artifacts/detection_hillclimb/liquid_field_capture_replay.json) · [Nozzle and piston prompt trial](../../artifacts/detection_hillclimb/liquid_presence_part_experiment.json) |
| Measure color and lightness | Implemented but not fully validated | Automatic color is currently limited to consistent localized warm regions. It emitted on 61 of 74 confirmed colored crops, including 9 of 13 in the held-out capture blocks. The latter had 2.20 L* mean absolute error against visual centerline samples. All 14 uncolored cases stayed indeterminate for presence. A native reviewer-selected sample also supports other hues. [Replay](../../artifacts/detection_hillclimb/liquid_field_capture_replay.json) |
| Estimate filled volume in mL | Implemented but not fully validated | Automatic volume uses supported 20 mL and 60 mL scales, readable numbered graduations and a visible piston. It emitted on 20 of 45 same-session colored controls. All 20 estimates were inside the reviewed brackets. The clear controls produced no volume. The replay evaluates visual brackets and contains no exact physical-volume reference. [Volume audit](../../artifacts/fill/audit.json) |
| Measure opacity | Partially implemented | The app accepts matched, registered crops of an observed barrel and a clear reference. It reports relative contrast attenuation. The fit requires a visible reference texture and matching setup. Controlled synthetic pairs passed, and there is no real paired reference for this corpus. Opacity is indeterminate on the single-image path. [Pair method and controlled evidence](../../artifacts/liquid_appearance_validation.json) |

## Review actions

Open an observation. A nominal capacity entered in the review is distinct from a volume measured using visible graduations. Use **Liquid presence** to record a review decision. Selecting **Empty** removes incompatible current appearance or positive-fill fields. The previous prediction remains in the original snapshot. A human-confirmed empty state may have an indeterminate mL value. A numeric zero is retained when a visible graduation and piston measurement supports it.

Redrawing clears the attached visual scale, barrel color, transcription, printed evidence, and crop measurements. Values selected again on the new box are recorded as reviewed. A visual-scale label alone leaves nominal capacity unresolved. A manually confirmed nominal capacity supersedes an incompatible visual-scale suggestion. Manual fill uses the reviewed landmarks and does not quantify the reviewer's measurement uncertainty.

Select **Visual scale and barrel color** to check joint inventory categories. Reference comparison includes the joint distribution alongside the individual categories. Reference exports include the exact uploaded reference bytes when the original upload is available.

Partial labels can cover only compatible expected slots. The report gives lower bounds on missing and extra objects even when some attributes are indeterminate. Editing a transcription withdraws the earlier automatic reading and apparent fill derived from those markings. An explicit capacity or a separate reviewed landmark measurement remains traceable to its own evidence.

Choose **Sample liquid color** and draw a small region over the visible liquid, clear of the markings and plunger. Save the review. The server reads the sample from the original bytes and calculates median sRGB and Lab. It stores the sample coordinates and marks the color result as reviewed. With a region of the same textured barrel available in two registered images, **Measure relative opacity** accepts those two crops and returns the attenuation estimate and fit quality.

In the four-photo live Intel Arc check, the app counted and localized all four syringes. On the Fil Rouge control it returned a nominal capacity of 20 mL, visible orange liquid and an estimated 16 mL with a reported 14–17 mL interval. The visual reference bracket was 15–18 mL. On the public yellow example, the automatic consistency guard abstained after an earlier approach measured a darker portion of the device. A native manual sample yielded yellow. On the public 5 mL example, the reviewer confirmed the capacity and visibly empty state. The app retained an indeterminate mL estimate. Both revised findings survived the ZIP round-trip with their initial predictions preserved. The two visually empty photos remained unresolved automatically. [Live API and exported field checks](../../artifacts/detection_hillclimb/liquid_field_live_evaluation.json)

## Reproduce

```bash
.venv/bin/python -m scripts.experiments.public_data.public_syringe_dataset --verify
.venv/bin/python -B -m scripts.experiments.public_data.public_bulk_syringe_collection verify
.venv/bin/python -m scripts.experiments.measurement.liquid_fields_audit replay
MEDSYRINGE_DEVICE=xpu HF_HUB_OFFLINE=1 data/sam3/xpu-env/bin/python -m scripts.experiments.measurement.liquid_fields_audit live
.venv/bin/python -m pytest -q
```

All model operations above run locally. Public photographs and Fil Rouge controls are recorded separately. The field test selects one already reviewed body per scene and counts a missed localization as an abstention. Uncolored photographs have unknown contents unless the visible piston and barrel establish a reference. Color and opacity are properties of the visible capture.
