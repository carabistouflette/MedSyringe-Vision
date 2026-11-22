# Release results

The current release returned the exact count on all **177 Fil Rouge photos** and matched **172 of 177 reference boxes** at IoU 0.50. These photos come from one capture group. On **23 external native-resolution photographs**, it matched **39 of 48** instances, produced four false boxes and returned 20 exact counts. On the five crowded photographs in that source, it matched 24 of 30 instances and counted two scenes exactly. [Full live regression](../../artifacts/detection_hillclimb/sam_residency_full_live.json) · [Independent device audit](../../artifacts/detection_hillclimb/device_scene_audit_results.json)

The more varied 24-acquisition crowd challenge returned **33 of 81** matches and **10 of 24** exact counts, with one false detection. This is 97.1% precision and 40.7% recall at IoU 0.50. Missed instances remain the primary failure in dense scenes. [Independent crowd challenge](../studies/fil-rouge/crowd_capacity_liquid_update.md)

## Detection optimization

The app applies a fixed detection configuration. A local SigLIP 2 crop check sends weak accepted detections to human review. On independently sourced MedBin photos, the gate removed 15 false positives and preserved all 11 existing matches. On 26 additional negative scenes, correct empty results rose from 20 to 25. Recall on the fresh mixed Commons group remains 15/22. On Intel Arc the application shares one SAM worker across proposals, corroboration and liquid measurements. [Cross-domain detection report](../studies/detection/detection_generalization.md) · [Offline runtime qualification](../guides/local_runtime_bundle.md)

The expanded corpus contains 547 public photographs with visually reviewed syringe presence. Instance annotations remain in review. The dataset card records acquisition groups, license provenance, annotation audits and a local review export. [Public dataset card](../datasets/public_syringe_dataset.md)

The review and export report nominal capacity, liquid presence, apparent color and lightness, volume and opacity as separate fields. Use a native liquid sample for a reviewed color measurement. The paired crop tool measures relative contrast attenuation when a matching clear reference exists. [Validated scope and review workflow](../guides/product_validation.md)

Earlier alternatives and comparisons are described in the [detector optimization report](../studies/detection/detector_optimization.md), the [lookalike precision study](../studies/detection/precision_study.md), the [supervised training study](../studies/detection/supervised_training.md), the [hybrid and public-data experiments](../studies/detection/hybrid_experiments.md), and the [RF-DETR verification study](../studies/detection/rfdetr_verification.md).
