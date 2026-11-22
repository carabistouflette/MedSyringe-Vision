# Local collection of public syringe image candidates

Collected on November 21, 2026. The manifest contains full-scene thumbnails with a maximum long edge of 1,920 px. The collection targets tightly packed, overlapping, transparent and partially occluded syringes, readable detail views and difficult lookalikes. Images and contact sheets live under the Git-ignored local data directory. Collection uses local files and public HTTPS downloads. Fil Rouge data is read solely for the local collision screen. Every thumbnail has a public image parent. Local image processing applies EXIF orientation, proportional resizing and JPEG encoding.

## Collection and review counts

| Source | Metadata candidates | Local thumbnails | Positive leads | Lookalike leads | Provenance groups | Useful after visual screening |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mendeley Dental | 1,085 | 130 | 50 | 80 | 1 | 8 |
| Open Images | 106 | 22 | 22 | 0 | 21 | 15 |
| MedBin Universe | 142 | 75 | 34 | 41 | 1 | 22 |
| Wikimedia Commons | 626 | 212 | 122 | 90 | 113 | 26 |
| **Total** | **1,959** | **439** | **228** | **211** | **136** | **71** |

The collection has **152 images with 1,139 publisher boxes**, including the dental distractor classes, and **75 additional images with publisher class counts**. The Open Images subset retained 43 boxes, including two group boxes and 16 depiction boxes. Its explicit test negatives did not survive the provenance screen. Seven MedBin images have counts for multiple syringes. Metadata candidates count export members or catalog records. Dental augmentations can therefore inflate that denominator.

Output thumbnails represent 136 conservative provenance groups. The genuinely independent acquisition count is unknown. Commons uses photographer credits, Open Images uses Flickr profiles, and each of MedBin and Dental forms one unresolved study group. The 71 useful views span 39 groups. The qualitative screen covered 86 views selected for crowding, attribute legibility or lookalikes. It identified 38 syringe scenes, 31 difficult negative candidates, two detail-only scenes and 15 rejects. Seven reviewed useful views show closely packed or numerous syringes, three show overlap, ten show occlusion and 29 show transparent bodies. Tags can overlap. The 353 unscreened thumbnails and the supplier labels still need review.

The pre-download screen excluded 301 historical IDs or dental parent names. Of 605 examined assets, 21 collided with historical bytes or visual fingerprints, 19 matched another candidate, 98 were unavailable or outside raster limits, and 28 were excluded by the quota or shared a selected dental parent. The first Commons catalog pass reached an HTTP 429 rate limit. The run combined the completed public queries with the repository's previously acquired public catalog, stopping further catalog queries after the rate limit. The source snapshot records the failed query. A separate SHA-256 check against all 148 original and derived hashes in the Fil Rouge source inventory found zero matches.

| Qualitative attribute | Views with evidence | Unknown among 439 |
| --- | ---: | ---: |
| Syringe type | 35 | 404 |
| Printed markings readable | 13 | 426 |
| Graduations readable | 18 | 421 |
| Plunger rod or seal visible | 34 | 405 |
| Barrel interior visible | 22 | 417 |
| Liquid visible or supported by the image and publisher description | 2 | 437 |

Coverage counts a view when the attribute is supported for at least one identifiable syringe. This is a targeted screen of legible examples. An unknown value does not imply absence.

## Sources and terms

| Source | Access and annotations | License and credit |
| --- | --- | --- |
| Mendeley Dental v1, DOI 10.17632/mh6zpc7y9g.1 | The published archive contains the Roboflow **Dental Surgical Tools v1 YOLOv11** export at `Dental tools Datasets/Dental Surgical Tools.v1i.yolov11.zip`. Each row records its archive member, the filename before the `.rf.` suffix and the publisher's boxes and classes. The export stretches images to 512 × 512. Its README lists flips and rotations in the training derivatives. | **CC BY 4.0 International** for the distributed images and annotations. Credit the six named contributors. [Mendeley dataset](https://data.mendeley.com/datasets/mh6zpc7y9g/1) · [Roboflow export version](https://universe.roboflow.com/object-detection-hgn2k/dental-surgical-tools/dataset/1) |
| Open Images | Official Amazon S3 JPEGs, V6 train and V5 test boxes for `/m/012ysf`, 2018_04 image metadata, and V5 test image-level negative labels with confidence `0` where available. The manifest retains native flags for groups, occlusion, truncation, depictions and objects inside other objects. A nonzero or unknown publisher rotation flags the coordinates for review. | The CSV declares **CC BY 2.0 Generic** for these images and supplies Flickr attribution per row. Open Images asks reusers to verify each image license at its source. Google LLC annotations are **CC BY 4.0 International**. [Open Images downloads](https://storage.googleapis.com/openimages/web/download_v7.html) · [Open Images licenses](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses) |
| MedBin_dataset 2, Roboflow Universe | A public catalog snapshot already available in the repository supplies native JPEG URLs and per-image class counts. The earlier project run found that the COCO export required `version:export` permission. This collection has class counts only. The correspondence between this listing and the full paper corpus remains unresolved. | **CC BY 4.0 International** as declared on the Universe project. The project credit is `medical waste`. Individual photographers and capture sessions are unknown. [Universe project](https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2) · [Export version](https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2/dataset/1) · [MedBin paper](https://doi.org/10.1016/j.wasman.2025.114742) |
| Wikimedia Commons | Public MediaWiki API queries used detail titles, syringe categories and lookalike categories. File descriptions and categories are review leads. These rows have `annotations: null`. | Each file records the exact Creative Commons license URL, including the version and jurisdiction where applicable. Keep the file credit and document changes when reusing an image. Check the applicable ShareAlike terms before sharing adapted thumbnails. [Imageinfo API](https://www.mediawiki.org/wiki/API:Imageinfo) · [Commons reuse guide](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) |

The exact image-license totals are 223 CC BY 4.0 International, 47 CC BY 2.0 Generic, 3 CC BY 3.0 Unported, 1 CC BY 1.0 Generic, 84 CC BY-SA 4.0 International, 61 CC BY-SA 3.0 Unported, 4 CC BY-SA 2.5 Generic, 3 CC BY-SA 2.0 Generic and 13 CC0 1.0 Universal. The manifest provides the original license URL and declared credit per image.

The Mendeley public file API returned an empty list for SID-RAS v1. The Deol et al. study includes overlapping instruments and lists its data as available on request. The Roboflow project with barrel and piston-seal classes is a lead for a future authorized export. [SID-RAS public file API](https://data.mendeley.com/public-api/datasets/cyghvmjrt3/files?folder_id=root&version=1) · [Deol et al. data availability](https://doi.org/10.1186/s13037-024-00406-y) · [Universe syringe parts project](https://universe.roboflow.com/ang/syringe-odcgd)

## Reproduce and verify

~~~sh
.venv/bin/python -m scripts.public_annotated_collection collect
.venv/bin/python -m scripts.public_annotated_collection verify
.venv/bin/python -m scripts.public_annotated_collection apply-review
~~~

`collect` uses the local source snapshots and the historical exports in this checkout, ranks candidates with a fixed SHA-256 seed, and writes the images, manifest and contact sheets by source. If the manifest already exists, the command verifies the local collection. To rebuild, archive the previous output directory, copy its `inputs/` into a new collection directory at the original path, use the same historical references and run `collect`. Public image availability and JPEG encoder versions can affect rebuilt bytes. For byte-for-byte verification, retain the original collection and use `verify`.

The `rows` array in `manifest.json` holds paths relative to `data/`. `sha256` and the 64-bit `dhash` describe the local JPEG. `upstream_sha256` describes the exact bytes read from the source file or archive member. `source_url` points to a file page or dataset, `original_url` records the upstream photo or archive URL, and `download_url` records the asset actually retrieved. `author` is the declared credit. `publisher_partition` describes the provider's original partition. `group` records the available provenance for conservative grouping. Bounding boxes use normalized `[xmin, ymin, xmax, ymax]` coordinates. Provider classes, boxes and negative labels require review before training.

`apply-review` binds qualitative observations to the reviewed thumbnail's SHA-256 and recalculates coverage. Unknown attributes remain `null`. Review records visual legibility only. Physical dimensions, volume, fill level, concentration, metric piston position and dose are outside the recorded fields. Provider annotations and visual observations have separate fields.

## Priority views for review

| Review target | Local candidate | What to inspect |
| --- | --- | --- |
| Overlap, partial occlusion, transparent barrels | [oi-0089b699b364a570](../../data/public-annotated-collection/images/oi-0089b699b364a570.jpg) | Clear syringe bodies crossing on white fabric. Several scale ticks and dark seals are legible. The publisher supplies a group box. Draw one box per separately identifiable syringe. |
| Overlap, clutter and type ambiguity | [oi-0bdd01af13c161cb](../../data/public-annotated-collection/images/oi-0bdd01af13c161cb.jpg) | Mixed bin of bodies and plunger rods, with an oral syringe and overlapping components. Agree on oral-syringe scope and match rods to barrels before counting. |
| Severe crowding | [commons-113570809](../../data/public-annotated-collection/images/commons-113570809.jpg) | Layered syringes in a yellow sharps container. Define an ignore policy for ambiguous fragments before using the scene for instance counting. |
| Dense packaging | [commons-154040646](../../data/public-annotated-collection/images/commons-154040646.jpg) | Closely spaced prefilled syringes partly obscured by reflective blister packaging. Check whether the project scope includes syringes in packages. |
| Type, body print and graduations | [commons-26451792](../../data/public-annotated-collection/images/commons-26451792.jpg) | Three closely aligned bodies with legible printed scales and dark seals. Record the type and attribute evidence for each visible instance separately. |
| Plunger and internal contrast | [medbin-jkCg2amSXzYbH2fvFEku](../../data/public-annotated-collection/images/medbin-jkCg2amSXzYbH2fvFEku.jpg) · [commons-24354361](../../data/public-annotated-collection/images/commons-24354361.jpg) | Assembled side view on a dark mat, plus a disassembled barrel and rod for part review. |
| Liquid visibility | [oi-9b1a594e0cea3393](../../data/public-annotated-collection/images/oi-9b1a594e0cea3393.jpg) · [commons-971336](../../data/public-annotated-collection/images/commons-971336.jpg) | A yellow liquid is visible in the Open Images photo. The Commons publisher describes liquid in a clear barrel photographed against black. Confirm the attribution on the raw Commons file and compare its edited variants before use. |
| Difficult negatives | [medbin-8pP1S8iS2niRKQAqxQOD](../../data/public-annotated-collection/images/medbin-8pP1S8iS2niRKQAqxQOD.jpg) · [commons-77326661](../../data/public-annotated-collection/images/commons-77326661.jpg) | Transparent graduated tubes and a dense row of laboratory pipettes. Verify the full scenes for stray in-scope instances. |

## Comparison with the project history

| Existing source | Historical images and groups | How to interpret the new candidates |
| --- | ---: | --- |
| Open Images additional train pool | 67 reviewed images, 51 profile groups | This run extends the same publisher data. The group-of and depiction examples in the new pool explain why image-level review matters. Check Flickr aliases and group boxes before using them. [Historical manifest](../../artifacts/detection_hillclimb/openimages_additional_manifest.json) |
| Mendeley Dental reserve | 20 reviewed images, one source group | This run uses the same archive and excludes the named parents in the reserve. Original filenames capture the augmentation families, while tray, camera, illumination and instrument reuse create additional dependencies across filenames. Keep this domain paired with its historical acquisition. [Historical manifest](../../artifacts/detection_hillclimb/dental_manifest.json) |
| Native MedBin reserve | 25 reviewed images, one unresolved acquisition group | This run uses the same Universe project and screens against its previously downloaded IDs, bytes and fingerprints. The new images still show repeated specimens and backgrounds. Treat the entire project family as overlapping the reserve. [Historical manifest](../../artifacts/detection_hillclimb/medbin_held_out_manifest.json) |
| Independent Commons benchmark | 36 reviewed images, 36 photographer groups | The new pool broadens detail and clutter coverage. Photographer identity remains a grouping proxy and search categories retain off-topic files. [Historical benchmark](../studies/detection/detection_public_benchmark.md) |
| Simulated phlebotomy | 230 historical images across 37 clip groups | Existing evidence covers staged in-hand use and related frames. This collection adds public stills from other providers. [Historical manifest](../../artifacts/detection_hillclimb/phlebotomy.json) |

A new candidate may share an author, event, device or session with an earlier image. An exact normalized credit comparison found 32 new rows in 19 groups whose authors also appear in the historical Commons benchmark, additional fit pool, consensus audit, crowd reserve, precision-negative and lookalike pools or the Open Images additional pool. The script screens known IDs, normalized URLs, Commons titles, dental filename parents, exact SHA-256 and 64-bit dHash within Hamming distance four across rotations and reflections. Similar frames can survive when the background, camera angle or crop changes. This happened in two Open Images views of the same setup despite the hash screen. The visual review flags the second view. Check alias IDs, visually similar crops and group links across historical reserves before assigning roles. Model pretraining exposure is unknown.

Before training, review full scenes and enlarge small or overlapping objects. Confirm the syringe type and target scope for capped devices, tubing, oral applicators, insulin pens, needle-only views and disassembled parts. Draw one reference box per identifiable in-scope syringe using the repository's whole-visible-object convention. Check geometry against the oriented thumbnail, complete missing instances, resolve group boxes and inspect candidate negatives for unlabeled syringes. Keep attribute evidence attached to the corresponding instance when several devices are visible. MedBin class counts need new geometry, and Commons needs complete geometry. The parent task owns the final split. [Project annotation protocol](../reference/protocol.md)
