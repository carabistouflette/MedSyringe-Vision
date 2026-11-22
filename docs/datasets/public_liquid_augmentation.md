# Local liquid appearance augmentation prototype

The original synthetic renderer tests whether liquid appearance and piston position can vary while retaining all inscription pixels. The contact sheet shows nine controlled cases. [Contact sheet](../../data/public-liquid-augmentation/contact_sheet_en.png) · [Run manifest](../../data/public-liquid-augmentation/manifest.json)

## Reproduce

```bash
.venv/bin/python -B scripts/public_liquid_augmentation.py
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/datasets/test_public_liquid_augmentation.py
```

The generator builds its background, syringe, graduation marks, masks and piston from primitives. It reads no source photographs. The amber control and all its derivatives share one provenance group and must stay in the same partition. The manifest records the script hash and hashes of decoded RGB rasters.

The command writes to the local, Git-ignored `data/public-liquid-augmentation/` directory. The measured run used Python 3.12.14, Pillow 12.2.0 and NumPy 2.5.5. Two generations produced the same SHA-256 values for all 22 files. All 28 targeted tests passed. The prototype uses the local environment and leaves installed inference assets intact.

## Controls and measurements

Each variant changes one setting from the amber control. Hue and saturation describe the chosen mixing colour. The transmission parameter sets how much of the background contributes to the encoded RGB mixture. The cloudiness parameter applies Gaussian blur to the background under the liquid at the same transmission setting. A single control can affect several measurements on the rendered image.

The 40 × 20 pixel probe spans four stripes inside the liquid. Hue and saturation come from converting its average RGB to HSV. The contrast metric divides the absolute luma difference between dark and light stripes by the difference in a clear synthetic reference of the same scene, which is 44.0 weighted RGB code levels.

| Case | Graduation | Visible liquid pixels | HSV hue | HSV saturation | Contrast relative to the clear reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Amber control | 3 | 9,102 | 37.7° | 0.250 | 0.659 |
| Blue hue | 3 | 9,102 | 203.1° | 0.276 | 0.659 |
| Reduced saturation | 3 | 9,102 | 40.5° | 0.065 | 0.659 |
| More transparent appearance | 3 | 9,102 | 40.1° | 0.084 | 0.875 |
| Less transparent appearance | 3 | 9,102 | 37.2° | 0.523 | 0.289 |
| Cloudy appearance | 3 | 9,102 | 37.8° | 0.251 | 0.132 |
| Empty control | 0 | 0 | Unknown | Unknown | Unknown |
| Lower level | 1 | 3,034 | 37.7° | 0.250 | 0.659 |
| Higher level | 5 | 15,170 | 37.7° | 0.250 | 0.659 |

The diagram shows the cylinder from the side. Its piston face aligns with the selected graduation. The liquid area grows by 4,736 pixels per graduation, including 3,034 pixels visible through the inscription band. The rod and thumb rest move with the piston. The empty control has no liquid mask and the probe abstains.

The opaque inscription band covers 14,304 pixels and its protected border covers another 2,914. A separate check decoded the PNG outputs at native dimensions and compared their three RGB channels. All eight variants and the clear reference changed zero protected pixels and zero pixels outside the permitted region. Each photographic appearance variant changes 9,102 pixels inside the liquid mask. Graduations 0, 1 and 5 change 15,208, 10,694 and 10,694 pixels respectively from the parent. The allowed region for level changes is the union of liquid masks and piston footprints before and after movement. The binary PNG masks use white for the covered region.

## Limits

The contrast ratio depends on the chosen background pattern and stripe width. Colour mixing and background diffusion can both reduce contrast. The manifest leaves mL volume, nominal capacity, liquid identity, concentration, optical transmittance and NTU turbidity unknown.

The renderer uses a single orthographic cylinder and one background. Dead space, bubbles, refraction, lighting-dependent reflections and meniscus geometry are outside this experiment. Every variant shares the same syringe shape and printed scale. Training benefit and performance on real photographs remain unmeasured.
