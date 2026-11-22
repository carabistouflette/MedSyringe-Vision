# Public syringe photograph corpus

The local manifest lists **547 provider-resolution photographs with visually confirmed syringe presence** from nine Roboflow Universe projects. Two catalogue snapshots supplied 1,417 checksummed positive candidates. Four visual review batches selected 566 photographs for original-file retrieval. A second check excluded 14 protected matches or perceptually similar photos and five unsupported rasters. The surviving primary images include JPEG and MPO files. Original bytes are retained and each row records its asset URL, project credit, declared license, project group, preview checksum and original checksum. [Local manifest](../../data/public-syringe-dataset/manifest.json)

| Project group | Photos |
|---|---:|
| GLOVES, PILL WRAPPER AND SYRINGE | 111 |
| Stage 1 | 110 |
| Instrument train model | 98 |
| My First Project | 98 |
| Omnifix tube | 69 |
| Syringe pjt | 43 |
| SafeMed syringe detection | 10 |
| Syringe cotton | 4 |
| Syringe Detection | 4 |

Publishers declare CC BY 4.0 for 428 photos and public-domain status for 119. A project credit may identify the uploader only. Confirm photographer and original rights before release or reuse when those facts are unresolved. Treat each project as a provisional group in downstream partitioning, and join known repeated specimens and sessions across projects. Multiple views within a shared acquisition do not establish independent transfer coverage. Fil Rouge remains the protected regression reference.

## Annotation quality

The 547 rows confirm presence at image level. They require review of each instance and its visible geometry before detection training. Volume, type, colour and opacity remain unreviewed unless supported by additional evidence. The publisher metadata may provide a label count or a single box enclosing several syringes.

A manual overlay audit examined 34 photos and 45 publisher syringe boxes. Nine of 22 Open Images photos had provisionally usable instance labels at the inspected scale. Six were depictions, two were montage variants, two needed separate instance boxes, two needed a taxonomy decision, and one was too occluded to verify. The 12 sampled dental photos contained two plausible positive boxes and ten plausible hard negatives. The dental observations come from the same dataset and do not establish quality across independent acquisition sessions. [Audited records](../../artifacts/detection_hillclimb/public_publisher_annotation_audit.json)

One Open Images photo illustrates the counting failure. The publisher supplied one box for the scene. A manual reading identifies ten distinguishable syringe bodies. In local Intel Arc inference, the deployed hybrid returned two detections on the cropped screenshot and four on the full public photograph, with five additional review proposals in each case. Some accepted boxes cover multiple barrels. Both inputs come from one acquisition and illustrate a failure of instance separation. [Numeric evidence](../../artifacts/detection_hillclimb/public_overlap_count_case.json) · [Visual comparison](../../data/public-syringe-dataset/overlap-count-comparison.jpg)

The local RF-DETR annotation pilot used 12 new photos with 14 manually marked syringe instances. Its first proposal per image matched 8 of the 14 references at IoU 0.50, with four unmatched proposals. Pooling the five best proposals per image covered all 14, with 46 unmatched proposals. The same acquisition can contribute multiple images. Earlier overlays were visible before writing the coarse manual references. Treat these results as a review-queue diagnostic. [Pilot metrics and references](../../artifacts/detection_hillclimb/public_annotation_pilot_review.json) · [COCO preview archive](../../data/public-syringe-dataset/annotation-pilot.zip)

## Verify and review locally

```bash
.venv/bin/python -m scripts.public_syringe_dataset --verify
HF_HUB_OFFLINE=1 data/rfdetr/env/bin/python -m scripts.public_annotation_queue \
    data/public-syringe-dataset/manifest.json data/public-syringe-dataset/review-batch.zip --limit 24
```

The first command verifies asset hashes and the bound catalogues and visual review files. The second exports proposals in COCO with an oriented raster per image and a separate prelabel ledger. Inspect each visible instance in the local annotation tool and correct or add its full visible body box. For overlapping devices, review the original pixels and body axes to verify identity. Reject group boxes, loose components and depicted syringes. Follow the acquisition partition and partial-supervision rules when approving the corrected labels. [Public fitting protocol](public_training_protocol.md)
