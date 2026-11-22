# Bulk collection of public syringe photographs

This collection uses public Roboflow Universe image catalogs and the CDC Public Health Image Library. The original media and audit records are stored locally under `data/public-bulk-syringe-collection/`. The target is at least 500 visually verified, distinct positive images. A positive has a visible, identifiable physical syringe barrel or assembled syringe. Camera frames from a single acquisition can contribute distinct views. A needle alone, an empty wrapper, a vial, a rendering, and a repeated view do not contribute.

## Executed results

On November 21, 2026, the collector saved **3,698 public rasters** and released **742 visually verified positives** after duplicate and burst screening. The positives span **9 conservative acquisition groups**. These groups include long, highly correlated capture sequences. The actual number of independent camera sessions is unknown. [Positive manifest](../../data/public-bulk-syringe-collection/manifest.json)

The public catalogs supplied 35,761 records, with 14,268 carrying a matching publisher detection class or image label. Those figures include undownloaded records and repeated or unusable images. Among the local rasters, the automatic screen excluded 145 exact-byte repeats, 952 within-collection dHash matches, 130 historical dHash matches and 20 historical URL matches. After that screen, the visual pass recorded 36 negatives, 503 suspected burst repeats, 149 ambiguous images and 54 other exclusions, including apparent transforms and images with unresolved lineage. Another 967 downloaded candidates remain unreviewed. [Duplicate log](../../data/public-bulk-syringe-collection/duplicate-log.json) · [Review inventory](../../data/public-bulk-syringe-collection/review-inventory.json)

| Source | Catalog records | Publisher-labeled positives | Downloaded | Verified positives | Verified negatives | Visual bursts | Automatic exclusions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cdc-phil | 5 | 0 | 5 | 4 | 0 | 0 | 0 |
| top-syringe | 3,292 | 2,929 | 359 | 152 | 0 | 86 | 73 |
| syringe-object | 1,572 | 1,544 | 359 | 119 | 0 | 81 | 154 |
| syringe-test | 453 | 453 | 358 | 156 | 0 | 144 | 36 |
| syringe-lines | 391 | 391 | 312 | 198 | 0 | 85 | 15 |
| syringe-project | 172 | 127 | 135 | 73 | 0 | 50 | 0 |
| syringe-needle | 1,392 | 102 | 116 | 32 | 0 | 16 | 68 |
| syringe-seg | 281 | 199 | 61 | 0 | 0 | 29 | 7 |
| syringe-det | 2,848 | 2,736 | 119 | 0 | 0 | 0 | 35 |
| biomedical-waste | 3,802 | 513 | 217 | 8 | 0 | 2 | 15 |
| pharmaceutical-waste | 5,800 | 791 | 217 | 0 | 36 | 2 | 39 |
| vials-syringe | 8,893 | 1,744 | 622 | 0 | 0 | 0 | 315 |
| syringe-vial | 447 | 168 | 56 | 0 | 0 | 0 | 3 |
| syringe-plus | 1,590 | 1,590 | 578 | 0 | 0 | 0 | 440 |
| mmt-waste | 573 | 162 | 61 | 0 | 0 | 0 | 2 |
| uea-waste | 251 | 129 | 61 | 0 | 0 | 8 | 22 |
| medicine-syringe | 200 | 200 | 62 | 0 | 0 | 0 | 23 |
| vial-counting | 3,413 | 104 | 0 | 0 | 0 | 0 | 0 |
| syringe-poly | 386 | 386 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **35,761** | **14,268** | **3,698** | **742** | **36** | **503** | **1,247** |

The nine positive groups include six Roboflow project or shared-provenance groups for device photographs, one shared biomedical-waste group, and two CDC groupings by credited photographer and year. Within one setup, the visual pass retained meaningfully changed views and held back apparent repeats with similar framing, pose, background and object placement. Small differences can still survive the screen. No group is designated as an independent benchmark.

## Source licensing and access

The Roboflow connector's public Universe image-search tool listed original image URLs and image-level class summaries. It was called serially, with 200 records per page and at least one second between completed calls. The collector saved the observed pages and fetched only the public URLs supplied in those responses. The connection lacked `version:export` permission for the COCO archives of syringe det v2 and Biomedical Waste Detection v1. The downloaded assets come from the separate public image catalogs and carry no version-archive split or box claims. Public catalog counts can differ from the project page and from export counts. [Roboflow Universe](https://universe.roboflow.com/) · [syringe det v2](https://universe.roboflow.com/syringe-project/syringe-det/dataset/2) · [Biomedical Waste Detection v1](https://universe.roboflow.com/biowastemanagement/biomedical-waste-detection-nv7fq-bpeeu/dataset/1)

Roboflow rows carry the name and license printed by the dataset publisher. The platform does not identify the photographer or recording for every image. The run held back obvious web and stock imagery whose uploader attribution could not establish its original rights. The official CDC records carry the file-specific credit and public-domain designation. The CDC's public display rendition is the exact asset stored locally. DVIDS discovery used public pages and would require an explicit per-image public-domain mark, credit and file ID. The paced fetch received HTTP 403 for those DVIDS pages. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [CDC PHIL terms and downloads](https://wwwn.cdc.gov/phil/faq.aspx) · [DVIDS rights](https://www.dvidshub.net/about/copyright)

The run used distinct projects from the existing annotated collection and the Commons/Openverse search collection. Public historical manifests supplied URLs and fingerprints for overlap screening. Fil Rouge files and records were outside the read and write paths. The run used public GET requests and the licensed public listing connector. It sent no image uploads and made no purchases. [Existing annotated collection](public_annotated_collection.md) · [Existing web collection](public_web_collection.md)

## Public datasets examined

| Dataset | Publisher and declared license | Evidence from the run |
| --- | --- | --- |
| Top Syringe Detection | Tals Demo, CC BY 4.0 | Overhead camera views around connected equipment. Related frames are grouped at the project level because the catalog does not identify the original recordings. [Project](https://universe.roboflow.com/tals-demo/top-syringe-detection) |
| Syringe Object Detection | Angels Workspace, CC BY 4.0 | Handheld syringes recorded against alternating pale and dark surfaces around one bench. [Project](https://universe.roboflow.com/angels-workspace-busje/syringe-object-detection-gg123) |
| syringe-test | test, CC BY 4.0 | Close and wide views of capped, color-marked syringes around one table and mat. [Project](https://universe.roboflow.com/test-uugy6/syringe-test) |
| Syringe Lines | Line Segmentation, CC BY 4.0 | Handheld and bench syringe-barrel photos. Identical scenes appear in the mixed syringe det, vials_syringe and syringe_plus catalogs. The projects therefore share a conservative group. [Project](https://universe.roboflow.com/line-segmentation/syringe-lines) |
| Syringe Project | Michaels Workspace, CC BY 4.0 | Needle and syringe-body labels, with a fixed bench and handheld views. The catalog is a different project from the earlier bench source by the same workspace. Both are treated as one group because the equipment and setup are similar. [Project](https://universe.roboflow.com/michaels-workspace-ldsqh/syringe-project) · [Historical bench source](https://universe.roboflow.com/michaels-workspace-ldsqh/syringe-detection-fnu4f) |
| Syringe | Syringe, CC BY 4.0 | Bench views of a few syringe types with many similar repeated positions. [Project](https://universe.roboflow.com/syringe-16nka/syringe-vyhfk) |
| Biomedical Waste Detection and Pharmaceutical&Biomedical Waste | Biowastemanagement, CC BY 4.0, and Medical Waste Disposal, publisher-marked Public Domain | The same blue-drape scene appears in both catalogs. The public original endpoints already include several tilted, black-cornered or monochrome derivatives. The Pharmaceutical catalog also labels obvious vials, needles, masks and other objects as syringe. [Biomedical project and augmented v3](https://universe.roboflow.com/biowastemanagement/biomedical-waste-detection-nv7fq-bpeeu/dataset/3) · [Pharmaceutical project](https://universe.roboflow.com/medical-waste-disposal/pharmaceutical-biomedical-waste) |
| syringe det, vials_syringe and syringe_plus | syringe project, vial and vialSyringe, CC BY 4.0 | The same staged images, web imagery, obvious stock watermarks, renders and repeated views occur across the catalogs. These catalogs have weaker per-image lineage than the controlled sources. [syringe det](https://universe.roboflow.com/syringe-project/syringe-det) · [vials_syringe](https://universe.roboflow.com/vial-0no6n/vials_syringe) · [syringe_plus](https://universe.roboflow.com/vialsyringe/syringe_plus) |
| Additional medical waste and device leads | Each source page declares CC BY 4.0 | The downloaded Medicine sample contains stock-style photos. Some MMT items show watermarks. The UEA sample contains many landfill overview photos whose putative syringes cannot be identified at review scale. The segmentation sample contains many blurry views and packaging. [Medicine](https://universe.roboflow.com/medicine-gi9mg/syringe-aj8no-rnagc) · [MMT](https://universe.roboflow.com/mmt-kcsv2/medical-waste-yanoy) · [UEA](https://universe.roboflow.com/uea-ltv8w/medical-waste-bahxi) · [Segmentation](https://universe.roboflow.com/ws-1jgjc/syringe-rpizw) |

TrashBox reports 507 syringe images but the original repository and paper did not establish per-image rights for its web collection. The Kaggle *Syringe images* publisher selected `other` and supplied no explicit license terms in the public metadata description. The CNR Medical-Waste-4.0 release does not list a syringe class. The Hydra Research Lab download terms limit use to noncommercial or educational purposes. The Washington and Uganda clinical preparation studies describe access by permission or on request. The previously staged Zenodo phlebotomy export is part of the project's existing evidence. These sources contributed no new positives. [TrashBox publisher](https://github.com/nikhilvenkatkumsetty/TrashBox) · [TrashBox paper](https://www.fruct.org/files/publications/volume-31/fruct31/Kum.pdf) · [Kaggle metadata](https://www.kaggle.com/api/v1/datasets/metadata/maryam14222/syringe-images) · [CNR categories](https://github.com/cnr-isti-silab/Medical-Waste-4.0-Dataset) · [Hydra download terms](https://hydralab.rmutsv.app/dataset/medical-waste-dataset/) · [Washington data availability](https://pmc.ncbi.nlm.nih.gov/articles/PMC11496812/#Sec13) · [Uganda data availability](https://pmc.ncbi.nlm.nih.gov/articles/PMC12837600/) · [Prior Zenodo work](../studies/detection/rfdetr_transfer.md)

## Reproduce and audit

```sh
.venv/bin/python -B -m scripts.public_bulk_syringe_collection stage
.venv/bin/python -B -m scripts.public_bulk_syringe_collection sheets
.venv/bin/python -B -m scripts.public_bulk_syringe_collection publish
.venv/bin/python -B -m scripts.public_bulk_syringe_collection verify
```

The commands above run offline with the saved catalog, downloaded images and recorded reviews. The `fetch` and `fetch-official` commands retrieve public rasters from the saved discovery inputs when needed. `fetch` deterministically ranks image IDs per source, mixes sources in a round-robin, uses the selected positive and control budgets, and resumes from cached bytes. The observed runs allowed at most four media request starts per second and four concurrent transfers. HTTP 429 and 503 stop the download run. The image files retain the bytes served by the provider. Catalog pagination and source totals are historical snapshots and may change.

Several download passes stopped after the sample review showed overlap or low yield. The file counts reflect the complete local inventory. The first interrupted download pass ended before it wrote a fetch-run log. Later passes logged cached, downloaded, failed and paused items. [Fetch-run logs](../../data/public-bulk-syringe-collection/fetch-runs/)

The manifest paths are relative to `data/`. `source_url` denotes the publisher project or individual government photo page. `original_url` identifies the exact public media asset. The collector records SHA-256 of the stored bytes and the project's EXIF-oriented 64-bit dHash. It compares original URLs and SHA-256, then compares identity and rotated or mirrored dHashes at Hamming distance ≤4. The comparison uses the pinned public fingerprints in `history-index.json`. The associated manifest paths and checksums appear in `duplicate-log.json`. Perceptual matching can incorrectly collide on simple backgrounds and can miss a strong crop or edit. Suspected burst frames identified in contact-sheet review are held back separately. Review records point to both a stable contact-sheet position and the exact SHA-256 of the image seen.

The `group` field is a conservative acquisition grouping for split control. Roboflow grouping is by project or shared provenance when capture sessions are unknown. The same group may contain multiple camera sessions. Treat all images from a group together and keep them out of independent evaluation if any belonged to model development. No training or split assignment was run.
