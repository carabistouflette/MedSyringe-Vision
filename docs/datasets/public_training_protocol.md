# Public acquisition and augmentation protocol

Fil Rouge is a locked regression reference for the user’s expected detection and field outputs. New training initializes from documented publisher weights or from checkpoints whose fitting parents are public. The older hybrid has historical Fil Rouge-derived components, so its scores on that capture remain same-source regression evidence. The public intake checks the original and expanded Fil Rouge inventories, original and derived checksums, and transformed visual fingerprints. No new training derivative has a Fil Rouge parent.

## Admission

Harvest public metadata and candidate rasters in separate directories. Establish original URL, author, license, intact scene, visual references, and an acquisition group before admitting an image. Cross-collection duplicates and links to earlier data are screened by checksum, provenance, and orientation-aware difference hashes. Connected acquisition groups stay together. The candidate detector may prioritize review. Empty predictions remain eligible for review. A candidate cannot enter training while the review is incomplete.

Use `scripts.experiments.public_data.public_training_intake` to combine source manifests and `scripts.experiments.public_data.public_acquisition_partition` to freeze reviewed references. The partition prioritizes independent coverage of crowded, transparent, occluded, negative, readable-marking and visible-piston cases. Sparse strata are disclosed. Positive references cover identifiable syringe bodies. A loose component, a printed illustration, and a nonmedical applicator are distinct hard negatives. Keep unknown fields null. A printed capacity, a syringe’s type, the piston scale position, and the volume of visible liquid are separate fields. Paired reference marks and a visible piston may be recorded without a physical volume measurement.

The second public collection contains 742 presence-reviewed photos and frames. Its assigned group aliases link 156 records to the earlier Syringe Test evaluation and 73 to the previously evaluated bench project. Intake and reviewed partition build now block both groups from new fitting, including when a record was published under a different project name. Cross-catalog pixel and lineage screening is still required for the other records. [Acquisition exposure audit](../../artifacts/detection_hillclimb/expanded_public_catalog_exposure.json) · [Second collection](public_bulk_syringe_collection.md)

## Controlled training

The first controlled experiment uses 111 reviewed public fit photographs and adds one view from each of 40 selected acquisition groups. The control repeats the original image. The geometry arm uses seeded rotation, scale, projective perturbation and cropping. It updates boxes, polygons and landmarks and rejects a derivative if a target would become too ambiguous to label. The capture arm changes brightness, contrast, glare, blur and compression and masks attribute supervision. The combined arm uses the same geometry with capture changes. All arms have 151 examples, 122 box references and 44 negative examples, with the same initialization and one local CPU epoch. The real development photographs are separate from the fitting groups. The final audit remains reserved. [Matched design](../../artifacts/detection_hillclimb/public_augmentation_design.json)

All four checkpoints were evaluated on the same 43 real development photos, covering 36 syringes and 13 hard-negative scenes. At the fixed 0.20 confidence floor and IoU 0.50, the capture arm found six of the 20 syringes in the transparent slice with one false positive there. It found none in the other two slices and produced 17 false positives overall. The operating point is unsuitable for deployment. Ranked AP improves modestly in all augmented arms. The one-epoch result supports further testing of capture augmentation on the expanded corpus. It provides no evidence of a production recall improvement. [Per-domain results and checkpoint binding](../../artifacts/detection_hillclimb/public_augmentation_development.json)

| Training arm | AP50 | AP50–95 | Precision at 0.20 | Recall at 0.20 | F1 at 0.20 |
|---|---:|---:|---:|---:|---:|
| Repeated real control | 0.274 | 0.110 | 0.000 | 0.000 | 0.000 |
| Geometry | 0.279 | 0.123 | 0.000 | 0.000 | 0.000 |
| Capture | 0.316 | 0.127 | 0.261 | 0.167 | 0.203 |
| Geometry and capture | 0.306 | 0.127 | 0.000 | 0.000 | 0.000 |

On the same development photos, combining the capture model's whole-frame predictions with four overlapping tiles increased matches from 6 to 10 and false positives from 17 to 19. F1 rose from 0.203 to 0.308, while ranked AP50 fell from 0.316 to 0.273. All six previously matched references remained matched. The tiles cost a median of 1.03 extra CPU seconds per image. Results are for an auxiliary RF-DETR checkpoint, so the deployed hybrid still needs an independent advantage before including this branch. [Inference view comparison](../../artifacts/detection_hillclimb/public_capture_inference_development.json)

The photo-type and liquid targets require separate supervised examples. The local liquid prototype uses an original synthetic raster. It varies hue, saturation, visual contrast and rendered fill while keeping graduation and inscription pixels intact. Its ground truth describes the renderer. Transparent-liquid appearance depends on the light and background. A real colour or optical-opacity benchmark needs documented capture and a paired comparison or instrument. [Liquid augmentation and limits](public_liquid_augmentation.md)

## Evaluation

Measure detection precision, recall, F1, localization and exact counts on real photos, separately by acquisition and condition. For each attribute, distinguish observable, indeterminate and unreviewed references. Report coverage and errors conditional on emission, as well as the fraction of indeterminate observations on which a value was issued. Evaluate volume points and intervals separately using exact mL references where available and visual brackets otherwise. The clear and unreadable controls cannot support an assertion of physical zero. [Field evaluation](../studies/measurement/fill_measurement.md)
