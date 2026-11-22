# Independent Commons syringe stress benchmark

The benchmark contains 36 real photographs with source and license provenance, one per normalized photographer group. Files are publisher-delivered Wikimedia Commons renditions bounded at 960 pixels in width with original photo URLs recorded. The actual evaluated raster dimensions and hashes are pinned. Images remain local.

| Split | Photos | Photographer groups | Positive scenes | Syringes | Lookalike scenes | Multi-syringe scenes | Partial syringe scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Public development | 11 | 11 | 6 | 9 | 5 | 1 | 6 |
| Public audit | 25 | 25 | 13 | 15 | 12 | 2 | 9 |

Candidate discovery used Wikimedia Commons file metadata. Candidate groups and rasters were checked against 628 prior IDs, 499 previous URLs, 178 normalized photographer names, 925 exact hashes and 654 perceptual hashes staged in the project. Recorded URLs and titles were normalized before comparison. A new raster was excluded if any rotation or reflection had a 64-bit difference-hash distance of at most four from a prior retained raster. These checks address observed provenance collisions. They have limited sensitivity to large crops, altered backgrounds, unknown aliases and visually different views of the same physical subject.

Reference review used the full scene and crops enlarged for small objects. It did not use model outputs. The photos include injections in progress, overlapping syringes, a capped medication syringe, syringes installed in pumps, writing tools, laboratory pipettes, an insulin pen and a thermal paste applicator. Uncertain identities, diagrams and scenes with uncertain instance enumeration were excluded. The annotation convention covers the entire visible syringe and attached visible components, including an attached visible needle. Boxes use orientation-corrected normalized coordinates. Lookalike boxes provide auxiliary evidence on negative scenes. The audit labels are single-reviewer annotations, and the scene prevalence reflects deliberate stress-set curation.

Assignments were frozen before scoring by sorting photographer groups with a fixed SHA-256 seed within positive and lookalike strata. Every third group went to development. The manifest records the review hash, dimensions, exact raster hash, author, source file identifier, source URL, rendition URL and license for each photo. The dominant licenses are CC BY and CC BY-SA variants, with some CC0 and Commons public-domain records. Use the file-level metadata for attribution. Generic model-pretraining overlap is unknown.

Verify the local benchmark with `.venv/bin/python -m scripts.experiments.public_data.detection_public_benchmark verify`. Use the `restore` stage for missing files. It downloads only pinned renditions and checks their SHA-256 before writing. [Frozen manifest](../../../artifacts/detection_hillclimb/public_manifest.json) · [Blind visual reviews](../../../artifacts/detection_hillclimb/public_reviews.json)
