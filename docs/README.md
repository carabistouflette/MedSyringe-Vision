# Documentation

Start with the [root README](../README.md) to run the application.

## Usage and engineering

- [Run and review locally](guides/local_usage.md)
- [Engineering and reproduction](guides/engineering.md)
- [Offline runtime bundle](guides/local_runtime_bundle.md)
- [Timed operator review protocol](guides/operator_review_protocol.md)
- [Local inference optimization](guides/inference_qualification.md)
- [Product validation for counting and liquid fields](guides/product_validation.md)

## Project reference

- [Release results](reference/release_results.md)
- [MedSyringe Vision](reference/report.md)
- [Specification status](reference/requirements.md)
- [Experimental protocol](reference/protocol.md)
- [Local vision for 75 syringe images](reference/research.md)
- [Presentation](presentation.pdf) · [Typst source](presentation.typ)

## Datasets and annotation

- [Public syringe instance annotation pilot](datasets/instance_source_review.md)
- [Local collection of public syringe image candidates](datasets/public_annotated_collection.md)
- [Bulk collection of public syringe photographs](datasets/public_bulk_syringe_collection.md)
- [Local liquid appearance augmentation prototype](datasets/public_liquid_augmentation.md)
- [Public syringe photograph corpus](datasets/public_syringe_dataset.md)
- [Public acquisition and augmentation protocol](datasets/public_training_protocol.md)
- [Public syringe photo collection](datasets/public_web_collection.md)
- [Fresh syringe detection benchmark](datasets/detection_corpus_research.md)
- [Independent public syringe counting reference](datasets/crowd_next_reference.md)
- [Crowd audit protocol](crowd_next_audit_protocol.md)

## Detection studies

- [Verifying automatic second-pass syringe detections](studies/detection/automatic_recrop_verifier.md)
- [Separating overlapping syringe instances](studies/detection/axis_instance_separation.md)
- [Ranking complete visible syringes before suppression](studies/detection/complete_syringe_selection.md)
- [Calibrating the completeness score](studies/detection/completeness_calibration.md)
- [Count-preserving syringe relocalization](studies/detection/count_preserving_relocalization.md)
- [Separating overlapping syringe bodies with SAM 3](studies/detection/crowded_sam_evaluation.md)
- [Syringe detection across acquisition domains](studies/detection/detection_generalization.md)
- [Independent Commons syringe stress benchmark](studies/detection/detection_public_benchmark.md)
- [Unified syringe detector optimization](studies/detection/detector_optimization.md)
- [Hybrid detector experiments without additional original photos](studies/detection/hybrid_experiments.md)
- [Hybrid detection for small transparent syringes](studies/detection/hybrid_research.md)
- [Low-count syringe recovery and OCR-independent chamber appearance](studies/detection/low_count_recovery.md)
- [Native-crop verification for crowded syringe scenes](studies/detection/native_candidate_verifier.md)
- [Small and occluded syringes, and seal-front localization](studies/detection/occlusion_and_seal_study.md)
- [Ranking syringe proposals and resolving overlapping instances](studies/detection/overlap_ranking.md)
- [Pair-supervised, count-preserving syringe relocalization](studies/detection/pair_relocalization.md)
- [Assigning visible parts to individual syringes](studies/detection/part_association.md)
- [Syringe part geometry and appearance verification](studies/detection/part_verifier.md)
- [Precision study against syringe lookalikes](studies/detection/precision_study.md)
- [Appearance verification before syringe recropping](studies/detection/replacement_appearance.md)
- [RF-DETR segmentation transfer and hybrid detection](studies/detection/rfdetr_transfer.md)
- [Lookalike verification for RF-DETR additions](studies/detection/rfdetr_verification.md)
- [Supervised detector and localization study](studies/detection/supervised_training.md)
- [Learning when to add, replace or reject a syringe box](studies/detection/syringe_action_verifier.md)
- [Syringe assembly identity and thin-object views](studies/detection/syringe_assembly_and_thin_views.md)
- [Separate syringe identity and box quality](studies/detection/syringe_box_quality.md)
- [Object verification for syringe recovery](studies/detection/syringe_object_verifier.md)

## Measurement studies

- [Apparent syringe fill](studies/measurement/fill_measurement.md)
- [Stroke-guided capacity, piston landmarks, and additive detection](studies/measurement/graduation_piston_recovery.md)
- [Local recovery, syringe markings and liquid appearance](studies/measurement/sam3_markings_liquid.md)

## Fil Rouge and release studies

- [Public instance audit, crowded counting, and small-barrel measurements](studies/fil-rouge/crowd_capacity_liquid_update.md)
- [Expanded collection experiment log](studies/fil-rouge/expanded_experiment_log.md)
- [MedSyringe Vision — Expanded Fil rouge evaluation](studies/fil-rouge/expanded_report.md)
- [Fil Rouge precision, fill review, and Intel runtime](studies/fil-rouge/improvement_regression.md)
