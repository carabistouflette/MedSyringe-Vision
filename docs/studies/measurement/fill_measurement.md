# Apparent syringe fill

MedSyringe Vision estimates the apparent fill of a photographed 20 mL or 60 mL syringe. It reports the estimated mL and percentage of nominal capacity, a visual range, and the piston edge and printed marks supporting the estimate. The field uses the scale and shape visible in the image. A gravimetric measurement, opaque container, unreadable scale, or invisible contents require separate evidence. In particular, an uncolored image does not establish that the syringe is empty.

## Measurement

The estimator uses the rim of the piston closest to the syringe tip as the longitudinal reading point. The approximate liquid-column volume follows from the corresponding position on the printed graduations. Sources illustrating this measurement geometry and machine-vision volume estimation include FDA device instructions and a syringe vision study. [FDA syringe diagram](https://www.accessdata.fda.gov/drugsatfda_docs/label/2019/022505s012s013lbl.pdf) · [Machine-vision study](https://www.tandfonline.com/doi/full/10.1080/24699322.2025.2582020)

The local pipeline combines three signals. Guarded PP-OCRv6 gives the nominal capacity and labeled scale points. A linear fit maps the printed points to the barrel axis. When exactly two widely separated marks are readable, the fallback also checks for a repeating sequence of visible graduation strokes and widens the output range. Finally the installed SAM 3 and DINOv2 appearance worker supplies a colored-interior seed. The estimator follows the color along the barrel and checks for a sustained transition into the dark piston head across three parallel strips. If the original appearance pass abstains on an upside-down view, three consistent numbered labels can authorize one upright retry for the liquid region. The piston and marks are still reported in the original coordinates.

The visual range covers graduation-fit residuals, ambiguity in where the edge falls, and the rounding of the displayed estimate. Its width increases for a two-label fit. Perspective, the curved barrel, reflections, and air bubbles can affect the result. A substantially interrupted colored column, an ambiguous piston edge or missing scale evidence triggers abstention. The range has no calibrated confidence level.

## Field abstention and reporting

A localized syringe keeps its object ID and contributes to the count even when its type or volume is indeterminate. The review interface gives a concrete reason when the detector or reader supplies one. API observations and ZIP reports expose `field_status` for the current review and `original_field_status` for the original prediction. The supported type field records visible scale categories. Nominal capacity and estimated liquid volume are separate. An indeterminate volume has `ml: null`. A supported zero estimate has `ml: 0` with status `estimated`.

The field evaluator records coverage on observable references, errors among predictions issued, and emissions on indeterminate references. It excludes unreviewed attributes from those denominators. It records point estimates and intervals separately. Exact mL error requires an exact reference. A visual reference bracket instead supports point inclusion and interval overlap. The format can represent an interval-only result. The current estimator issues a point with an optional visual range or abstains. [Field evaluator](../../../scripts/evaluate_field_abstention.py) · [Field and zero-value tests](../../../tests/measurement/test_field_status.py)

Fresh full-field inference on three public photographs correctly localized two of eight reference objects. Both localized syringes had indeterminate type relative to the supported scales and no estimable volume. Neither produced an unsupported type or volume estimate. The other six targets were detection misses and are outside the matched-field denominator. Replaying the locked visual references gave point coverage of 18/35 for the Fil Rouge 20 mL group and 2/10 for the 60 mL group. Every emitted point fell inside its reference bracket, and every emitted range overlapped the reference bracket. No values were emitted on the 14 clear controls. The groups are from one capture and the brackets do not establish physical error in mL. [Observed field controls](../../../artifacts/detection_hillclimb/field_abstention_results.json)

## Validation

Photo references are visually read graduation brackets. They include uncertainty around curved marks and the thickness of the visible rim. There are no syringe weights, dispensed volumes or calibrated fill values for this capture. Thirty-one development images represent amber acquisition blocks. The orange 20 mL and 60 mL acquisition blocks were reserved for this task before selecting parameters, and the clear block supplies an abstention control. All blocks share the Fil rouge session. Prior projects used some of the same photos for generic detection and markings; this study uses them for a new task with separately recorded references. [Reference brackets](../../../artifacts/fill/visual_reference.tsv) · [Frozen selection and all 36 cases](../../../artifacts/fill/selection.json)

| Cohort | Photos | Visual estimates | Estimates within the visual reference bracket |
| --- | ---: | ---: | ---: |
| Amber development | 31 | 7 | 7 |
| Orange 20 mL, held out | 35 | 18 | 18 |
| Orange 60 mL, held out | 10 | 2 | 2 |
| Uncolored control | 14 | 0 | — |

The two-label tick check increased development coverage from six to seven at zero observed out-of-bracket readings. The 20 mL evaluation uses reviewed JPEG boxes. The 60 mL evaluation uses older, pinned detector crops of the corresponding photographs. The combined held-out coverage is 20 of 45 colored photographs. The result brackets are too coarse to establish exact milliliter error. [Development estimates](../../../artifacts/fill/development.json) · [Per-image held-out evidence](../../../artifacts/fill/audit.json)

The release check exercises the actual detection, OCR, appearance and fill path on original JPEGs and rasters rotated by 180 degrees. The first check abstained on the selected rotation control because the detector crop yielded no reliable nominal-capacity reading. Two additional rotation controls identified a separate cause of abstention in the segmented appearance region. The upright retry retained the 38 mL estimate on IMG_3203 and the 16 mL estimate on IMG_3316 across both orientations. The clear control abstained in both orientations. A fill measurement is retained through export and through visual review when the box and scale still agree with the underlying model evidence. A corrected box or scale removes the measurement from the current revision and preserves it in the original snapshot. [Initial runtime check](../../../artifacts/fill/runtime_first_attempt.json) · [Rotation check before the retry](../../../artifacts/fill/runtime_before_orientation_retry.json) · [Full-pipeline controls](../../../artifacts/fill/runtime.json) · [Chromium evidence](../../../artifacts/fill/browser_smoke.json)

## Review a fill estimate

Select **Mark fill on image** for a detected or reviewed syringe. Choose two numbered graduation strokes on the same side of the barrel, one on each side of the piston. Click the piston edge facing the liquid near those strokes, enter the nominal capacity and the two graduation values, and save the review. The server interpolates in source-image pixel coordinates. The ZIP preserves the human landmarks and original automatic estimate.

An experimental rectified graduation-strip fallback was evaluated on the same capture with fixed settings. It added no supported estimates. [Follow-up study](../fil-rouge/improvement_regression.md)

## Reproduce locally

Run the commands in a restored experiment copy with the staged JPEG dataset, cached markings and appearance results from the prior Fil rouge studies, and the pinned local backbones. The input file hashes define the tested experiment. The app automatically enables the estimator after the release check. The existing **Read visible markings** option supplies the OCR input.

```bash
source_dir=/path/to/Images_fil_rougev1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.fill_study select "$source_dir"
.venv/bin/python -m scripts.fill_study audit "$source_dir"
.venv/bin/python -m scripts.fill_release "$source_dir"
.venv/bin/python -m pytest -q
```
