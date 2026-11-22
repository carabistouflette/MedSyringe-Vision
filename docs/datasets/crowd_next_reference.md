# Independent public syringe counting reference

The frozen reference contains 25 photos from 25 conservatively grouped acquisitions. Twenty show multiple assembled syringes and five are hard negatives. Thirteen photos contain at least three syringes and seven positive scenes show visible overlap. Together the positive scenes contain 84 marked syringes. The negative photos show an insulin pen, pens and a mechanical pencil, laboratory pipettes in two distinct setups, and vaccine vials. [Frozen manifest](../../artifacts/detection_hillclimb/crowd_next_reference_manifest.json)

| Visible syringes | Photos |
| ---: | ---: |
| 0 | 5 |
| 2 | 7 |
| 3 | 3 |
| 4 | 5 |
| 7 | 3 |
| 9 | 1 |
| 11 | 1 |

The photos came from previously screened public candidate catalogs and targeted searches of Commons and rights records. The manifest records the credited creator, acquisition context, source page, exact download URL, license and SHA-256 for every image. The archive credits identify separate vaccination events. Commons and Flickr images are limited to one photo per credited creator. The local exposure screen compared source URLs, identifiers, creator aliases, exact bytes and the repository's rotated and mirrored dHash comparisons at Hamming distance ≤4. The screen includes protected Fil Rouge fingerprints for collision checking. The reference uses only licensed public photos, with no images from Fil Rouge, known fitting projects or prior evaluation groups. [Historical comparison and exclusions](../../artifacts/detection_hillclimb/crowd_next_reference_screen.json)

The reader marked each identifiable assembled syringe on the oriented full image and reread overlaid boxes. Boxes enclose visible attached components, including caps and visible needles. A truncated or partly hidden object is counted when its visible barrel and device features identify a distinct syringe. Detached components and packaging are outside the target class. The most ambiguous piles and depth-blurred scenes were withheld. The original reading came from one reviewer and used no candidate predictions. This challenge deliberately overrepresents multiple-syringe scenes and includes historical devices. Its counts describe performance on these acquisitions. Public foundation-model pretraining exposure is unknown.

## Reproduce locally

```sh
.venv/bin/python -B -m scripts.experiments.public_data.crowd_next_reference verify
.venv/bin/python -B -m scripts.experiments.public_data.crowd_next_reference restore
```

The manifest paths are relative to `data/`. `verify` checks the pinned manifest, the source screen, local bytes, oriented dimensions, perceptual hashes, group uniqueness and reference geometry. `restore` fetches missing files from the recorded public URLs and accepts only their pinned bytes. Keep a local copy of the rasters in `data/crowd-next-reference/images/` because providers can replace renditions. Box coordinates are normalized XYXY in the EXIF-oriented full raster.
