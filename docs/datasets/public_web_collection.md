# Public syringe photo collection

The [thumbnail manifest](../../data/public-web-collection/manifest.json) provides candidates for annotation review. The five search strata target closely spaced or overlapping syringes, transparent barrels, partial occlusion, lookalikes, and views that reveal syringe type, markings, graduations, plunger, or barrel interior. A photograph can match several queries. The `stratum` field records its sampling bucket and `queries` records the searches that found it.

## Local reproduction

Run these commands from the repository root with the project's Python environment.

```bash
.venv/bin/python -B -m scripts.public_web_collection collect
.venv/bin/python -B -m scripts.public_web_collection verify
```

The collector uses anonymous, serial HTTPS GET requests. Commons provides search results and image renditions via `imageinfo` at a requested width of 800 px. Openverse provides the first page of each search and serves its thumbnails. The collector filters Wikimedia entries from Openverse to reduce cross-index duplication. It caches each successful query with a timestamp and SHA-256. The search plan is fixed and samples terms and strata in rotation. Each stratum can retain up to 48 Commons and 20 Openverse thumbnails, with at most five per group, API, and stratum. Search rankings and renditions can change between live runs. [MediaWiki Imageinfo API](https://www.mediawiki.org/wiki/API:Imageinfo) · [MediaWiki API etiquette](https://www.mediawiki.org/wiki/API:Etiquette/en) · [Openverse API](https://api.openverse.org/v1/) · [Openverse terms](https://docs.openverse.org/terms_of_service.html)

`collect --offline` repeats selection and comparison against the cached search responses, cached thumbnails, and available local history. A thumbnail that failed to download can appear as a cache miss on replay. `verify` checks image SHA-256, dHash and dimensions, plus the checksums of the API snapshots. `restore` retrieves missing thumbnails from the pinned URLs and checks their SHA-256 before writing. Keep a local copy of the collection to preserve the rasters if the provider replaces or removes them. `review` applies the observations in `visual_review.json` to matching image IDs and SHA-256 values.

In each manifest row, `image` is relative to `data/`, `source_url` is the provenance page and `original_url` is the direct URL supplied by the source or catalog. Openverse can reference an already resized provider file. `thumbnail_url` identifies the stored rendition. SHA-256 identifies the saved bytes and dHash describes the EXIF-oriented raster. The license filter accepts CC BY, CC BY-SA, CC0 and public-domain markings. Confirm rights and attribution on the original page before reusing any photograph outside this local review. [Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) · [Openverse terms](https://docs.openverse.org/terms_of_service.html)

## Historical comparison

The collector indexes metadata in `artifacts/`, previously staged public candidates, and selected historical raster directories. The manifest records the files and directories covered, index sizes, and read errors. It rejects confirmed overlaps and duplicates using Commons IDs, normalized URLs, and exact SHA-256. The URL normalization reconciles Commons file pages with thumbnail and media URLs and Flickr photo pages with media URLs. It flags historical and within-batch perceptual matches at Hamming distance ≤4 on the repository's 64-bit dHash, using eight orientations and reflections of the thumbnail. Historical author matches are recorded for group review. Simple backgrounds can cause false matches. Crops, different views of the same specimen, and creator aliases can evade the checks.

The previous independent Commons benchmark contained 36 photos, including three multi-syringe scenes. The five crowded native-audit photos share an installation and specimens, and MedBin is treated as one presumed acquisition. Further views of an acquisition that has already been used belong with that source's regression evidence. [Commons benchmark](../studies/detection/detection_public_benchmark.md) · [Native audit and MedBin](../studies/detection/detection_generalization.md) · [Prior Commons crowd reserve](../../artifacts/detection_hillclimb/commons_crowds_reserve.json)

Fil Rouge contributes only to the local historical comparison. The collector also rejects perceptual matches against the protected Fil Rouge index. All collected thumbnails came from public API renditions. No Fil Rouge image was used for fitting or augmentation, and this task ran no training. Requests contained public search terms or media identifiers. No user files were sent and no payment was made.

## Annotation review

1. Open the contact sheets in `sheets/` and use `review_index` to locate the full thumbnail. If the boundary or inscription is too small, retrieve a larger authorized rendition and record the new raster hashes under the same acquisition.
2. Assign two independent readers to crowded, overlapping, and partly occluded scenes. Draw one box per identifiable syringe around its visible parts, following the historical box convention. Record which objects occlude it and distinguish occlusion from frame truncation or reflections. Flag uncertain instances for arbitration. For each hard negative, check the entire frame for syringes and identify the lookalike.
3. Classify the visible type from the raster. Transcribe legible printing in its original language and separately record graduation ticks, plunger visibility, barrel transparency, the visible interior, and any apparent liquid feature. Unknown attributes stay `null`. Graduations can be visible while their numbers remain unreadable. For multi-object scenes, an attribute describes at least one identifiable syringe unless the note specifies more. These observations provide no measurements of physical volume, device capacity, fill fraction, or composition.
4. Review visual matches and verify provenance before any group is reserved. The provisional `group` is the normalized credited author. Images without an author have provisional individual groups. Bring together bursts, alternate crops, albums, event reports and recurring specimens even if their authors are formatted differently. Keep every whole acquisition and its derivatives together. Flag acquisitions previously exposed to model fitting or selection. Final split decisions belong to the parent.

## Collection results

The run retained **302** thumbnails in **195** provisional groups. All 79 planned API requests produced cached responses, with 973 result occurrences and 770 unique catalog entries that passed the metadata filter. Commons supplied 212 thumbnails. Openverse supplied 90, of which 83 were indexed from Flickr, five from Rawpixel, and two from iNaturalist. The manifest records both source URLs and a license or rights label for all 302 rows. An author is available for 278. Ten extra thumbnails from the first Commons pass remain in the cache. Use the 302 manifest rows as the review inventory.

An offline replay reproduced all 302 candidate IDs, SHA-256 values, dHashes, groups, and sampling strata. The integrity check also verified all raster dimensions and the checksums of the 79 API snapshots.

| Search stratum | Commons | Openverse | Total |
| --- | ---: | ---: | ---: |
| Crowded | 48 | 18 | 66 |
| Transparent | 44 | 17 | 61 |
| Occluded | 48 | 20 | 68 |
| Lookalike | 48 | 20 | 68 |
| Attributes | 24 | 15 | 39 |
| **Total** | **212** | **90** | **302** |

The historical index used 380 JSON files and 874 locally readable rasters, providing 1,907 distinct SHA-256 and 1,409 dHash values, including 203 protected Fil Rouge dHashes. The collector rejected 95 known sources, one within-batch URL duplicate, one within-batch exact byte duplicate, and five unavailable or invalid renditions. No thumbnail matched the protected Fil Rouge dHash index. Eight retained rows have a historical dHash flag and 74 have a historical author match. Ten rows have a within-batch dHash flag. The flags can coexist and require review before assigning acquisitions. [Search log](../../data/public-web-collection/queries.json) · [Rejection log](../../data/public-web-collection/rejected.json)

The first Openverse attempt used a page size rejected for anonymous access, then received HTTP 429. After pausing, anonymous requests with a page size of 16 succeeded. Shortfalls against the 340-thumbnail cap reflect the available search results, source overlap, group caps and thumbnail failures. The query counts report discovery intent. Some results depict packaging, illustrations or irrelevant objects, and publisher markings do not establish acquisition independence. Model pretraining exposure is unknown.

## Visible attribute coverage

A preliminary screen recorded observations on **42** shortlisted thumbnails, including six crowded views, three with explicit overlap, 18 with occlusion and 12 hard-negative candidates. Some tags overlap and one hard-negative candidate needs an identity decision. Counts below cover fields populated after looking at the thumbnail. Unreviewed items and unconfirmed values remain `null`.

| Attribute | Populated thumbnails | Best initial views by review index |
| --- | ---: | --- |
| Visually identifiable type | 22 | **73** plastic syringe, **81** disk filter, **282** prefilled presentation |
| Visible markings | 21 | **73** markings on clear plastic, **264** in-hand NIPRO printing, **282** wraparound label |
| Legible text transcribed | 12 | **73**, **81**, **264**, **282** |
| Visible graduations | 17 | **73** high contrast, **81** large ticks, **288** two overlapping scales |
| Visible plunger | 24 | **73** green seal, **81** black seal, **268** assembled metal plunger and **276** matching disassembled view |
| Visible barrel transparency | 22 | **77** blue stem through the clear barrel, **292** mixed bodies on a textured background |
| Visible barrel interior | 17 | **81**, **143**, **273** |
| Apparent liquid feature | 4 | **129**, **273**, **277** show colored columns from the same Ivan Radic setup. **301** has an uncertain backlit feature for second review. |

For liquid review, prioritize **273** for the close view of the visible colored columns and **129** for the wider view of the entire setup. Those views give one staged acquisition and no physical measurements. Clear-liquid scenes with a resolvable liquid boundary and an independent ground truth remain a coverage gap. For readings on a single syringe, start with **73** and **81**. Use **60** and **8** for independent counts of overlapping or occluded devices. Use **288** for a case with both overlap and legible graduations, subject to the historical author flag. [Visual screening and reserve proposals](../../data/public-web-collection/visual_review.json) · [Attribute contact sheet](../../data/public-web-collection/sheets/attributes-1.jpg)

## Whole acquisitions proposed for reserve review

| Group | Representative review indices | Reason to review together |
| --- | --- | --- |
| Tony Webster | **60**, **51** | The used-syringe scene has heavy overlap and a translucent container wall. Keep both collected group views together while checking whether they depict related acquisitions. |
| mikecogh | **8**, **4**, **269** | Discarded syringes with different occlusion and backgrounds. Check whether the scene with wrappers and the isolated views share a site or session. |
| Adoscam | **77**, **106** | Hand occlusion and clinical context. Confirm the relationship between the two Benin views. |
| boaventuravinicius | **212**, **239**, **247** | Three views of the same writing-tool series supply challenging negatives with transparent tubes, crowded tips and a low viewpoint. |
| MallardTV | **198** | One capped insulin pen with a visible dose window, suitable for hard-negative review. |

The 13 Ivan Radic rows include views of the colored-column setup. Keep the visible setup together and resolve the historical author matches before considering external evaluation. The two Jacek Halicki glass-syringe views (**268**, **276**) show the same device. One has a historical perceptual match, so treat both as review examples pending adjudication. Keep the Navy event report `230131-M-IP954` (**271**, **275**) together and check the historical Navy Medicine account aliases before proposing an independent reserve. None of these proposals assigns a split.
